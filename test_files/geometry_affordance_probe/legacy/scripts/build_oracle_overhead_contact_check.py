#!/usr/bin/env python3
"""Build a top-down oracle contact/goal-point review bundle.

This is an oracle-style sanity check, not a learned ORACLE-Grasp VLM runner.
It uses saved RLBench/AGNOSTOS overhead RGB, segmentation masks, depth, and
camera metadata to choose role-specific mask centroids from the initial frame,
then projects those 2D points into world XYZ and X-ICM voxel XYZ.

The intended question is: if a point-selection module could reliably choose
the right top-down object/goal masks, would the projected evidence be useful
enough for the final X-ICM prompt?
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6], dtype=np.float32)
VOXEL_SIZE = 100
DEFAULT_DEMOS = [
    ("turn_tap_episode161_0_71", "turn_tap", 161, "turn left tap"),
    ("close_jar_episode161_0_65", "close_jar", 161, "close the red jar"),
    ("light_bulb_in_episode161_0_37", "light_bulb_in", 161, "screw in the red light bulb"),
    ("slide_block_to_color_target_episode161_0_67", "slide_block_to_color_target", 161, "slide the block to green target"),
    ("sweep_to_dustpan_of_size_episode161_0_57", "sweep_to_dustpan_of_size", 161, "sweep dirt to the tall dustpan"),
    ("push_buttons_episode161_0_51", "push_buttons", 161, "push the maroon button"),
    ("put_groceries_in_cupboard_episode161_0_74", "put_groceries_in_cupboard", 161, "put the crackers in the cupboard"),
    ("put_money_in_safe_episode161_0_63", "put_money_in_safe", 161, "put the money away in the safe on the bottom shelf"),
    ("place_shape_in_shape_sorter_episode161_0_70", "place_shape_in_shape_sorter", 161, "put the cube in the shape sorter"),
    ("put_item_in_drawer_episode161_0_1", "put_item_in_drawer", 161, "put the item in the bottom drawer"),
    ("insert_onto_square_peg_episode161_0_63", "insert_onto_square_peg", 161, "put the ring on the red spoke"),
    ("open_drawer_episode161_0_62", "open_drawer", 161, "open the bottom drawer"),
]

BACKGROUND_TOKENS = {
    "panda",
    "link",
    "gripper",
    "floor",
    "table",
    "workspace",
    "boundary",
    "spawn",
    "resizablefloor",
    "diningtable",
    "visibleelement",
}

ROLE_RULES: dict[str, list[dict[str, Any]]] = {
    "turn_tap": [
        {
            "role": "manipulated_object_contact",
            "target_object": "tap",
            "target_part": "tap_handle",
            "keywords": [["left", "tap"], ["tap"]],
            "contact_mode": "twist",
        }
    ],
    "close_jar": [
        {
            "role": "manipulated_object_contact",
            "target_object": "jar_lid",
            "target_part": "lid",
            "keywords": [["jar_lid"], ["lid"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "jar",
            "target_part": "rim_or_body",
            "keywords": [["jar0"], ["jar"]],
            "contact_mode": "goal_region",
        },
    ],
    "light_bulb_in": [
        {
            "role": "manipulated_object_contact",
            "target_object": "bulb",
            "target_part": "bulb_body",
            "keywords": [["bulb1"], ["bulb"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "lamp",
            "target_part": "lamp_screw_or_socket",
            "keywords": [["lamp_screw"]],
            "contact_mode": "goal_region",
        },
    ],
    "slide_block_to_color_target": [
        {
            "role": "manipulated_object_contact",
            "target_object": "block",
            "target_part": "block_body",
            "keywords": [["block"], ["cube"], ["target"]],
            "contact_mode": "single_contact",
            "allow_multiple": True,
            "max_matches": 3,
            "note": "Overhead mask labels expose target1/target2/target3 but not a separate block label, so all target/block candidates are shown.",
        },
        {
            "role": "goal_region",
            "target_object": "color_target",
            "target_part": "target_surface",
            "keywords": [["target"]],
            "contact_mode": "goal_region",
            "allow_multiple": True,
            "max_matches": 3,
            "note": "Overhead mask labels do not encode target color, so all target candidates are shown.",
        },
    ],
    "sweep_to_dustpan_of_size": [
        {
            "role": "manipulated_object_contact",
            "target_object": "broom",
            "target_part": "broom_handle",
            "keywords": [["sweep_to_dustpan_broom"], ["broom_visual"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "dustpan",
            "target_part": "dustpan_opening",
            "keywords": [["dustpan_tall"], ["dustpan"]],
            "contact_mode": "goal_region",
        },
        {
            "role": "secondary_object_to_move",
            "target_object": "dirt",
            "target_part": "dirt_cluster",
            "keywords": [["dirt"]],
            "contact_mode": "sweep_target",
            "allow_multiple": True,
            "max_matches": 3,
        },
    ],
    "push_buttons": [
        {
            "role": "manipulated_object_contact",
            "target_object": "button",
            "target_part": "button_top",
            "keywords": [["push_buttons_target"], ["button"]],
            "contact_mode": "press_point",
            "allow_multiple": True,
            "max_matches": 3,
            "note": "Mask labels do not encode button color, so all visible button candidates are shown.",
        }
    ],
    "put_groceries_in_cupboard": [
        {
            "role": "manipulated_object_contact",
            "target_object": "groceries",
            "target_part": "object_body",
            "keywords": [["crackers"], ["grocery"], ["groceries"], ["box"], ["item"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "cupboard",
            "target_part": "cupboard_opening",
            "keywords": [["cupboard"], ["cabinet"]],
            "contact_mode": "goal_region",
        },
    ],
    "put_money_in_safe": [
        {
            "role": "manipulated_object_contact",
            "target_object": "money",
            "target_part": "money_stack",
            "keywords": [["dollar_stack"], ["money"], ["dollar"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "safe",
            "target_part": "safe_opening_or_shelf",
            "keywords": [["safe_body"], ["safe_door"], ["safe"]],
            "contact_mode": "goal_region",
        },
    ],
    "place_shape_in_shape_sorter": [
        {
            "role": "manipulated_object_contact",
            "target_object": "shape",
            "target_part": "shape_body",
            "keywords": [["cube"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "shape_sorter",
            "target_part": "matching_hole_region",
            "keywords": [["shape_sorter"]],
            "contact_mode": "goal_region",
        },
    ],
    "put_item_in_drawer": [
        {
            "role": "manipulated_object_contact",
            "target_object": "item",
            "target_part": "object_body",
            "keywords": [["item"], ["cube"], ["object"], ["block"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "drawer",
            "target_part": "drawer_opening",
            "keywords": [["drawer"]],
            "contact_mode": "goal_region",
        },
    ],
    "insert_onto_square_peg": [
        {
            "role": "manipulated_object_contact",
            "target_object": "square_ring",
            "target_part": "ring_body",
            "keywords": [["square_ring"], ["ring"]],
            "contact_mode": "grasp_pair",
        },
        {
            "role": "goal_region",
            "target_object": "square_peg",
            "target_part": "peg_or_spoke",
            "keywords": [["pillar"], ["peg"]],
            "contact_mode": "goal_region",
            "allow_multiple": True,
            "max_matches": 3,
            "note": "Mask labels do not encode peg color, so all visible pillar candidates are shown.",
        },
    ],
    "open_drawer": [
        {
            "role": "manipulated_object_contact",
            "target_object": "drawer",
            "target_part": "drawer_handle_or_front",
            "keywords": [["drawer_frame"], ["drawer"]],
            "contact_mode": "pull",
        }
    ],
}


class _LightweightDemo:
    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> Any:
        return self._observations[index]


class _LightweightObservation:
    pass


class _LowDimObsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "rlbench.demo" and name == "Demo":
            return _LightweightDemo
        if module == "rlbench.backend.observation" and name == "Observation":
            return _LightweightObservation
        return super().find_class(module, name)


def load_low_dim_obs(path: Path) -> Any:
    with path.open("rb") as handle:
        return _LowDimObsUnpickler(handle).load()


def decode_rgb_mask(mask_image: Image.Image) -> np.ndarray:
    rgb = np.array(mask_image, dtype=np.int64)
    if rgb.ndim == 2:
        return rgb
    return rgb[:, :, 0] + rgb[:, :, 1] * 256 + rgb[:, :, 2] * 256 * 256


def image_to_float_array(image: Image.Image, scale_factor: int) -> np.ndarray:
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    else:
        float_array = image_array.astype(np.float32)
    return float_array / scale_factor


def create_uniform_pixel_coords_image(resolution: tuple[int, int]) -> np.ndarray:
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1),
    ).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1),
    ).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    return np.concatenate((pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)


def transform(coords: np.ndarray, trans: np.ndarray) -> np.ndarray:
    height, width = coords.shape[:2]
    coords = np.reshape(coords, (height * width, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (height, width, -1))


def pixel_to_world_coords(pixel_coords: np.ndarray, cam_proj_mat_inv: np.ndarray) -> np.ndarray:
    height, width = pixel_coords.shape[:2]
    pixel_coords = np.concatenate([pixel_coords, np.ones((height, width, 1))], -1)
    world_coords = transform(pixel_coords, cam_proj_mat_inv)
    return np.concatenate([world_coords, np.ones((height, width, 1))], axis=-1)


def pointcloud_from_depth_and_camera_params(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    uniform_pixel_coords = create_uniform_pixel_coords_image(depth.shape)
    pixel_coords = uniform_pixel_coords * np.expand_dims(depth, -1)
    camera_position = np.expand_dims(extrinsics[:3, 3], 0).T
    rotation = extrinsics[:3, :3]
    rotation_inv = rotation.T
    rotation_inv_position = np.matmul(rotation_inv, camera_position)
    world_to_camera = np.concatenate((rotation_inv, -rotation_inv_position), -1)
    cam_proj_mat = np.matmul(intrinsics, world_to_camera)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(pixel_to_world_coords(pixel_coords, cam_proj_mat_inv), 0)
    return world_coords_homo[..., :-1][0]


def point_to_voxel_index(point: np.ndarray) -> np.ndarray:
    bb_mins = np.array(SCENE_BOUNDS[0:3])[None]
    bb_maxs = np.array(SCENE_BOUNDS[3:])[None]
    dims_m_one = np.array([VOXEL_SIZE] * 3)[None] - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([VOXEL_SIZE] * 3) + 1e-12)
    return np.minimum(np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one).reshape(point.shape)


def normalize_name(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def split_tokens(value: str) -> set[str]:
    return {token for token in normalize_name(value).split("_") if token}


def is_background_name(name: str) -> bool:
    tokens = split_tokens(name)
    return bool(tokens & BACKGROUND_TOKENS)


def mask_areas(mask: np.ndarray, mask_id_to_name: dict[Any, str]) -> dict[int, dict[str, Any]]:
    ids, counts = np.unique(mask, return_counts=True)
    name_by_id: dict[int, str] = {}
    for raw_id, raw_name in mask_id_to_name.items():
        try:
            name_by_id[int(raw_id)] = str(raw_name)
        except Exception:
            continue
    areas = {}
    for mask_id, count in zip(ids.astype(int), counts.astype(int)):
        name = name_by_id.get(mask_id)
        if not name or is_background_name(name):
            continue
        areas[mask_id] = {"mask_id": int(mask_id), "mask_name": name, "area_pixels": int(count)}
    return areas


def keyword_score(name: str, keyword_groups: list[list[str]]) -> int:
    normalized = normalize_name(name)
    tokens = split_tokens(name)
    best = 0
    for group in keyword_groups:
        group_score = 0
        for keyword in group:
            key = normalize_name(keyword)
            key_tokens = split_tokens(key)
            if key and key == normalized:
                group_score += 10
            elif key and key in normalized:
                group_score += 3
            elif key_tokens and key_tokens <= tokens:
                group_score += 2
            elif key_tokens & tokens:
                group_score += 1
        best = max(best, group_score)
    return best


def select_candidates(
    areas: dict[int, dict[str, Any]],
    *,
    keyword_groups: list[list[str]],
    allow_multiple: bool,
    max_matches: int,
) -> list[dict[str, Any]]:
    scored = []
    for item in areas.values():
        score = keyword_score(item["mask_name"], keyword_groups)
        if score <= 0:
            continue
        scored.append((score, item["area_pixels"], item))
    if not scored:
        fallback = sorted(areas.values(), key=lambda item: item["area_pixels"], reverse=True)[:1]
        return [{**item, "oracle_match_score": 0, "oracle_match_reason": "fallback_largest_named_non_background_mask"} for item in fallback]
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    selected = scored[: max_matches if allow_multiple else 1]
    return [
        {
            **item,
            "oracle_match_score": int(score),
            "oracle_match_reason": "keyword_mask_match",
        }
        for score, _, item in selected
    ]


def nearest_mask_pixel_to_centroid(mask: np.ndarray, mask_id: int) -> tuple[int, int, float, float]:
    ys, xs = np.where(mask == mask_id)
    if len(xs) == 0:
        raise ValueError(f"mask id {mask_id} has no pixels")
    mean_x = float(np.mean(xs))
    mean_y = float(np.mean(ys))
    idx = int(np.argmin((xs - mean_x) ** 2 + (ys - mean_y) ** 2))
    return int(xs[idx]), int(ys[idx]), mean_x, mean_y


def orientation_from_mask(mask: np.ndarray, mask_id: int) -> float | None:
    ys, xs = np.where(mask == mask_id)
    if len(xs) < 3:
        return None
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(coords, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vh[0]
    return float(math.atan2(float(axis[1]), float(axis[0])))


def local_world_point(
    mask: np.ndarray,
    point_cloud: np.ndarray,
    *,
    x: int,
    y: int,
    mask_id: int,
    window_radius: int,
) -> dict[str, Any]:
    height, width = mask.shape[:2]
    x0, x1 = max(0, x - window_radius), min(width - 1, x + window_radius)
    y0, y1 = max(0, y - window_radius), min(height - 1, y + window_radius)
    local_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    local_cloud = point_cloud[y0 : y1 + 1, x0 : x1 + 1]
    same_mask = local_mask == mask_id
    points = local_cloud[same_mask]
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points):
        return {
            "world_xyz": np.median(points, axis=0).astype(float).tolist(),
            "point_source": "median_same_mask_local_overhead_window",
            "num_points_used": int(len(points)),
        }
    point = point_cloud[y, x]
    if point.shape == (3,) and np.all(np.isfinite(point)):
        return {
            "world_xyz": point.astype(float).tolist(),
            "point_source": "exact_overhead_pixel_point_cloud",
            "num_points_used": 1,
        }
    return {"world_xyz": None, "point_source": "unavailable", "num_points_used": 0}


def load_camera_arrays(episode_path: Path, camera: str, frame_index: int) -> tuple[np.ndarray, np.ndarray, dict[Any, str]]:
    depth_scale = 2**24 - 1
    demo = load_low_dim_obs(episode_path / "low_dim_obs.pkl")
    obs = demo[frame_index]
    mask = decode_rgb_mask(Image.open(episode_path / f"{camera}_mask" / f"{frame_index}.png"))
    cam_depth = image_to_float_array(Image.open(episode_path / f"{camera}_depth" / f"{frame_index}.png"), depth_scale)
    near = obs.misc[f"{camera}_camera_near"]
    far = obs.misc[f"{camera}_camera_far"]
    cam_depth = (far - near) * cam_depth + near
    point_cloud = pointcloud_from_depth_and_camera_params(
        cam_depth,
        obs.misc[f"{camera}_camera_extrinsics"],
        obs.misc[f"{camera}_camera_intrinsics"],
    )
    return mask, point_cloud, obs.misc[f"{camera}_mask_id_to_name"]


def build_role_points(
    *,
    task: str,
    instruction: str,
    mask: np.ndarray,
    point_cloud: np.ndarray,
    mask_id_to_name: dict[Any, str],
    camera: str,
    window_radius: int,
) -> list[dict[str, Any]]:
    height, width = mask.shape[:2]
    areas = mask_areas(mask, mask_id_to_name)
    roles = ROLE_RULES.get(
        task,
        [
            {
                "role": "manipulated_object_contact",
                "target_object": task,
                "target_part": "largest_visible_task_object",
                "keywords": [[task]],
                "contact_mode": "unknown",
            }
        ],
    )
    records = []
    for role_rule in roles:
        selected = select_candidates(
            areas,
            keyword_groups=role_rule.get("keywords", [[task]]),
            allow_multiple=bool(role_rule.get("allow_multiple", False)),
            max_matches=int(role_rule.get("max_matches", 1)),
        )
        for candidate_index, candidate in enumerate(selected, start=1):
            x, y, centroid_x, centroid_y = nearest_mask_pixel_to_centroid(mask, int(candidate["mask_id"]))
            world = local_world_point(
                mask,
                point_cloud,
                x=x,
                y=y,
                mask_id=int(candidate["mask_id"]),
                window_radius=window_radius,
            )
            world_xyz = world["world_xyz"]
            voxel_xyz = None
            if world_xyz is not None:
                voxel_xyz = point_to_voxel_index(np.array(world_xyz, dtype=np.float32)).astype(int).tolist()
            records.append(
                {
                    "role": role_rule["role"],
                    "candidate_index_for_role": candidate_index,
                    "source_model": "mask_oracle_topdown_centroid",
                    "source_view": f"{camera}_rgb_initial",
                    "task_instruction": instruction,
                    "contact_mode": role_rule.get("contact_mode", "unknown"),
                    "target_object": role_rule.get("target_object", "unknown"),
                    "target_part": role_rule.get("target_part", "unknown"),
                    "point_2d_normalized_overhead": [
                        round(float(x) / float(width - 1), 6),
                        round(float(y) / float(height - 1), 6),
                    ],
                    "pixel_xy": [x, y],
                    "centroid_pixel_xy_float": [round(centroid_x, 3), round(centroid_y, 3)],
                    "mask_id": int(candidate["mask_id"]),
                    "mask_name": candidate["mask_name"],
                    "mask_area_pixels": int(candidate["area_pixels"]),
                    "oracle_match_score": int(candidate.get("oracle_match_score", 0)),
                    "oracle_match_reason": candidate.get("oracle_match_reason", ""),
                    "orientation_radians_topdown": orientation_from_mask(mask, int(candidate["mask_id"])),
                    "world_xyz": world_xyz,
                    "voxel_xyz": voxel_xyz,
                    "point_source": world["point_source"],
                    "num_points_used": world["num_points_used"],
                    "note": role_rule.get("note", ""),
                    "human_review": {
                        "correct_object": None,
                        "correct_part_or_goal_region": None,
                        "actionable_for_primitive": None,
                        "projection_plausible": None,
                        "notes": "",
                    },
                }
            )
    return records


def copy_image(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def draw_overlay(image_path: Path, output_path: Path, roles: list[dict[str, Any]]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "manipulated_object_contact": "red",
        "goal_region": "lime",
        "secondary_object_to_move": "deepskyblue",
        "constraint_region": "orange",
    }
    width, height = image.size

    def outlined_text(x: int, y: int, text: str, fill: str = "white") -> None:
        x = min(max(x, 2), max(width - 16, 2))
        y = min(max(y, 2), max(height - 14, 2))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), text, fill="black")
        draw.text((x, y), text, fill=fill)

    for idx, role in enumerate(roles, start=1):
        x, y = role["pixel_xy"]
        color = colors.get(role["role"], "yellow")
        radius = 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=2)
        draw.line((x - 10, y, x + 10, y), fill="black", width=1)
        draw.line((x, y - 10, x, y + 10), fill="black", width=1)
        theta = role.get("orientation_radians_topdown")
        if theta is not None:
            dx = math.cos(theta) * 18
            dy = math.sin(theta) * 18
            draw.line((x - dx, y - dy, x + dx, y + dy), fill=color, width=3)
        outlined_text(x + 8, y - 8, str(idx))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _role_key_lines(record: dict[str, Any]) -> list[tuple[str, str]]:
    colors = {
        "manipulated_object_contact": "red",
        "goal_region": "green",
        "secondary_object_to_move": "blue",
        "constraint_region": "orange",
    }
    out = []
    roles = record["oracle_contact_hints"]["interaction_roles"]
    for idx, role in enumerate(roles, start=1):
        color = colors.get(role["role"], "black")
        role_name = str(role["role"]).replace("_", " ")
        mask_name = str(role.get("mask_name", "unknown"))
        voxel = role.get("voxel_xyz")
        text = f"{idx}. {role_name}: {mask_name} | voxel {voxel}"
        out.append((color, text))
    return out


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path, thumb_width: int = 300) -> None:
    images = []
    for record in records:
        original_path = Path(record["overhead_image"])
        overlay_path = Path(record["overlay_image"])
        if not original_path.exists() or not overlay_path.exists():
            continue
        original = Image.open(original_path).convert("RGB")
        overlay = Image.open(overlay_path).convert("RGB")
        scale = thumb_width / original.width
        panel_height = int(original.height * scale)
        original = original.resize((thumb_width, panel_height))
        overlay = overlay.resize((thumb_width, panel_height))
        key_lines = _role_key_lines(record)
        line_height = 15
        key_height = 24 + max(1, len(key_lines)) * line_height
        gutter = 12
        tile_width = thumb_width * 2 + gutter
        tile = Image.new("RGB", (tile_width, panel_height + key_height), "white")
        tile.paste(original, (0, 0))
        tile.paste(overlay, (thumb_width + gutter, 0))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, thumb_width - 1, panel_height - 1), outline="black")
        draw.rectangle((thumb_width + gutter, 0, tile_width - 1, panel_height - 1), outline="black")
        draw.rectangle((0, 0, 58, 15), fill="white")
        draw.rectangle((thumb_width + gutter, 0, thumb_width + gutter + 72, 15), fill="white")
        draw.text((4, 2), "original", fill="black")
        draw.text((thumb_width + gutter + 4, 2), "with points", fill="black")
        title = f"{record['task']} / {record['demo_id']}"
        y = panel_height + 5
        draw.text((6, y), title[:48], fill="black")
        y += 18
        for color, text in key_lines:
            draw.rectangle((7, y + 3, 15, y + 11), fill=color, outline="black")
            draw.text((20, y), text[:78], fill="black")
            y += line_height
        images.append(tile)
    if not images:
        return
    columns = 1
    rows = int(math.ceil(len(images) / columns))
    tile_w = max(img.width for img in images)
    tile_h = max(img.height for img in images)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        sheet.paste(image, (col * tile_w, row * tile_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path.resolve())


def markdown_link(path: str | Path, label: str | None = None, image: bool = False) -> str:
    path = Path(path).resolve()
    label = label or path.name
    prefix = "!" if image else ""
    return f"{prefix}[{label}]({path})"


def build_markdown(output_dir: Path, records: list[dict[str, Any]], bundle_path: Path, sheet_path: Path) -> None:
    lines = [
        "# Oracle Overhead Contact/Goal Point Check",
        "",
        "This bundle tests an oracle-style top-down point selector on seen initial frames.",
        "It uses simulator masks, overhead depth, and camera metadata to select role-specific mask centroids, then projects each point into `world_xyz` and X-ICM `voxel_xyz`.",
        "",
        f"- Contact sheet: {markdown_link(sheet_path, 'oracle_overhead_contact_sheet.png', image=True)}",
        f"- JSON bundle: {markdown_link(bundle_path, 'oracle_overhead_points_bundle.json')}",
        "",
        "Color convention: red = manipulated object/contact, green = goal region, blue = secondary object to move.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['task']} / {record['demo_id']}",
                "",
                f"- Instruction: `{record['language_description']}`",
                f"- Episode: `{record['episode_path']}`",
                f"- Per-demo JSON: {markdown_link(record['json_path'])}",
                f"- Front image: {markdown_link(record['front_image'])}",
                f"- Overhead source: {markdown_link(record['overhead_image'], image=True)}",
                f"- Overhead oracle overlay: {markdown_link(record['overlay_image'], image=True)}",
                "",
                "Roles:",
            ]
        )
        for idx, role in enumerate(record["oracle_contact_hints"]["interaction_roles"], start=1):
            lines.extend(
                [
                    f"- `{idx}` `{role['role']}`: mask `{role['mask_name']}`, pixel `{role['pixel_xy']}`, "
                    f"world `{role['world_xyz']}`, voxel `{role['voxel_xyz']}`",
                ]
            )
        lines.append("")
    (output_dir / "oracle_overhead_points.md").write_text("\n".join(lines).rstrip() + "\n")


def parse_demo_spec(value: str) -> tuple[str, str, int, str]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("demo must be id:task:episode:language")
    return parts[0], parts[1], int(parts[2].replace("episode", "")), parts[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/yf23/datasets/ICRA27-ROBOT")
    parser.add_argument("--output-dir", default="test_files/geometry_affordance_probe/review/figures/oracle_overhead_points")
    parser.add_argument("--camera", default="overhead")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--demo", action="append", type=parse_demo_spec, help="id:task:episode:language. May be repeated.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    items_dir = output_dir / "items"
    records = []

    for demo_id, task, episode, language in args.demo or DEFAULT_DEMOS:
        episode_path = data_root / "seen_tasks" / task / "all_variations" / "episodes" / f"episode{episode}"
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)
        mask, point_cloud, mask_id_to_name = load_camera_arrays(episode_path, args.camera, args.frame_index)
        roles = build_role_points(
            task=task,
            instruction=language,
            mask=mask,
            point_cloud=point_cloud,
            mask_id_to_name=mask_id_to_name,
            camera=args.camera,
            window_radius=args.window_radius,
        )
        overhead_src = episode_path / f"{args.camera}_rgb" / f"{args.frame_index}.png"
        front_src = episode_path / "front_rgb" / f"{args.frame_index}.png"
        overhead_image = copy_image(overhead_src, images_dir / f"{demo_id}_{args.camera}.png")
        front_image = copy_image(front_src, images_dir / f"{demo_id}_front.png")
        overlay_path = images_dir / f"{demo_id}_{args.camera}_oracle_points.png"
        draw_overlay(overhead_src, overlay_path, roles)
        record = {
            "demo_id": demo_id,
            "task": task,
            "language_description": language,
            "episode_path": str(episode_path),
            "frame_index": args.frame_index,
            "front_image": front_image,
            "overhead_image": overhead_image,
            "overlay_image": str(overlay_path.resolve()),
            "oracle_contact_hints": {
                "source_model": "mask_oracle_topdown_centroid",
                "source_view": f"{args.camera}_rgb_initial",
                "source_note": "Upper-bound oracle check using simulator mask names, not a learned VLM.",
                "interaction_roles": roles,
                "use_as": "query_execution_hint_not_retrieval",
            },
        }
        item_path = items_dir / f"{demo_id}.json"
        record["json_path"] = write_json(item_path, record)
        records.append(record)

    bundle_path = output_dir / "oracle_overhead_points_bundle.json"
    write_json(bundle_path, records)
    with (output_dir / "oracle_overhead_points_bundle.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    sheet_path = output_dir / "oracle_overhead_contact_sheet.png"
    make_contact_sheet(records, sheet_path)
    build_markdown(output_dir, records, bundle_path, sheet_path)
    print(f"Wrote {output_dir.resolve()}")
    print(f"Wrote {bundle_path.resolve()}")
    print(f"Wrote {(output_dir / 'oracle_overhead_points.md').resolve()}")


if __name__ == "__main__":
    main()
