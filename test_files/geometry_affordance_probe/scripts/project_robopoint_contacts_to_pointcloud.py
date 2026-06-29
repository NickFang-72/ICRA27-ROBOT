#!/usr/bin/env python3
"""Project RoboPoint 2D contact hints into X-ICM object/voxel coordinates.

This script is intentionally not a retrieval scorer. It converts view-specific
RoboPoint points into execution evidence for the current query:

normalized 2D point -> pixel -> segmentation mask/object -> 3D point cloud
point -> X-ICM voxel coordinate.

It can run on a saved AGNOSTOS/RLBench episode for validation, and the pure
array functions can also be reused at evaluation time with live observation
masks and point clouds.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_xicm_imports(xicm_root: Path) -> None:
    for path in [xicm_root, xicm_root / "PyRep", xicm_root / "RLBench"]:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


class _LightweightDemo:
    def __len__(self) -> int:
        return len(self._observations)

    def __getitem__(self, index: int) -> Any:
        return self._observations[index]


class _LightweightObservation:
    pass


class _LowDimObsUnpickler(pickle.Unpickler):
    """Load RLBench low_dim_obs.pkl without importing PyRep/CoppeliaSim."""

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


def normalized_to_pixel(point: list[float] | tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = min(max(float(point[0]), 0.0), 1.0)
    y = min(max(float(point[1]), 0.0), 1.0)
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def _valid_point(point: np.ndarray) -> bool:
    return point.shape == (3,) and bool(np.all(np.isfinite(point)))


def _lookup_name(mask_id: int, mask_id_to_name: dict[Any, str]) -> str | None:
    return mask_id_to_name.get(mask_id) or mask_id_to_name.get(str(mask_id))


def find_nearest_named_mask_pixel(
    mask: np.ndarray,
    x: int,
    y: int,
    mask_id_to_name: dict[Any, str],
    search_radius: int,
) -> dict[str, Any]:
    height, width = mask.shape[:2]
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)

    exact_mask_id = int(mask[y, x])
    exact_name = _lookup_name(exact_mask_id, mask_id_to_name)
    if exact_name:
        return {
            "pixel": [x, y],
            "mask_id": exact_mask_id,
            "mask_name": exact_name,
            "pixel_distance": 0.0,
            "used_nearest_named_pixel": False,
            "exact_mask_id": exact_mask_id,
            "exact_mask_name": exact_name,
        }

    best: dict[str, Any] | None = None
    for radius in range(1, search_radius + 1):
        x0, x1 = max(0, x - radius), min(width - 1, x + radius)
        y0, y1 = max(0, y - radius), min(height - 1, y + radius)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                mask_id = int(mask[yy, xx])
                name = _lookup_name(mask_id, mask_id_to_name)
                if not name:
                    continue
                dist = math.hypot(xx - x, yy - y)
                if best is None or dist < best["pixel_distance"]:
                    best = {
                        "pixel": [xx, yy],
                        "mask_id": mask_id,
                        "mask_name": name,
                        "pixel_distance": dist,
                        "used_nearest_named_pixel": True,
                        "exact_mask_id": exact_mask_id,
                        "exact_mask_name": exact_name,
                    }
        if best is not None:
            return best

    return {
        "pixel": [x, y],
        "mask_id": exact_mask_id,
        "mask_name": exact_name,
        "pixel_distance": None,
        "used_nearest_named_pixel": False,
        "exact_mask_id": exact_mask_id,
        "exact_mask_name": exact_name,
    }


def local_mask_point_cloud_median(
    mask: np.ndarray,
    point_cloud: np.ndarray,
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
    same_object = local_mask == mask_id
    local_points = local_cloud[same_object]
    local_points = local_points[np.all(np.isfinite(local_points), axis=1)]

    if len(local_points):
        return {
            "world_xyz": np.median(local_points, axis=0).astype(float).tolist(),
            "point_source": "median_same_mask_local_window",
            "num_points_used": int(len(local_points)),
        }

    point = point_cloud[y, x]
    if _valid_point(point):
        return {
            "world_xyz": point.astype(float).tolist(),
            "point_source": "exact_pixel_point_cloud",
            "num_points_used": 1,
        }

    return {
        "world_xyz": None,
        "point_source": "unavailable",
        "num_points_used": 0,
    }


def _create_uniform_pixel_coords_image(resolution: tuple[int, int]) -> np.ndarray:
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


def _transform(coords: np.ndarray, trans: np.ndarray) -> np.ndarray:
    height, width = coords.shape[:2]
    coords = np.reshape(coords, (height * width, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (height, width, -1))


def _pixel_to_world_coords(pixel_coords: np.ndarray, cam_proj_mat_inv: np.ndarray) -> np.ndarray:
    height, width = pixel_coords.shape[:2]
    pixel_coords = np.concatenate([pixel_coords, np.ones((height, width, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    return np.concatenate([world_coords, np.ones((height, width, 1))], axis=-1)


def pointcloud_from_depth_and_camera_params(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Match PyRep VisionSensor.pointcloud_from_depth_and_camera_params."""
    uniform_pixel_coords = _create_uniform_pixel_coords_image(depth.shape)
    pixel_coords = uniform_pixel_coords * np.expand_dims(depth, -1)
    camera_position = np.expand_dims(extrinsics[:3, 3], 0).T
    rotation = extrinsics[:3, :3]
    rotation_inv = rotation.T
    rotation_inv_position = np.matmul(rotation_inv, camera_position)
    world_to_camera = np.concatenate((rotation_inv, -rotation_inv_position), -1)
    cam_proj_mat = np.matmul(intrinsics, world_to_camera)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(pixel_coords, cam_proj_mat_inv), 0)
    return world_coords_homo[..., :-1][0]


def project_points_from_arrays(
    *,
    points_2d_normalized: list[list[float]],
    mask: np.ndarray,
    point_cloud: np.ndarray,
    mask_id_to_name: dict[Any, str],
    camera: str = "front",
    contact_mode: str = "unknown",
    target_object: str | None = None,
    target_part: str | None = None,
    search_radius: int = 8,
    window_radius: int = 2,
    xicm_root: Path | None = None,
) -> dict[str, Any]:
    xicm_root = xicm_root or _repo_root() / "X-ICM"
    _ensure_xicm_imports(xicm_root)
    from utils import point_to_voxel_index  # type: ignore

    height, width = mask.shape[:2]
    projected = []
    for point in points_2d_normalized:
        raw_x, raw_y = normalized_to_pixel(point, width, height)
        snap = find_nearest_named_mask_pixel(mask, raw_x, raw_y, mask_id_to_name, search_radius)
        snap_x, snap_y = snap["pixel"]
        contact_point = local_mask_point_cloud_median(
            mask,
            point_cloud,
            snap_x,
            snap_y,
            int(snap["mask_id"]),
            window_radius,
        )
        world_xyz = contact_point["world_xyz"]
        voxel = None
        if world_xyz is not None:
            voxel = point_to_voxel_index(np.array(world_xyz, dtype=np.float32)).astype(int).tolist()
        projected.append(
            {
                "point_2d_normalized": [float(point[0]), float(point[1])],
                "pixel_xy": [raw_x, raw_y],
                "snapped_pixel_xy": [snap_x, snap_y],
                "mask_id": snap["mask_id"],
                "mask_name": snap["mask_name"],
                "exact_mask_id": snap["exact_mask_id"],
                "exact_mask_name": snap["exact_mask_name"],
                "used_nearest_named_pixel": snap["used_nearest_named_pixel"],
                "pixel_distance_to_named_mask": snap["pixel_distance"],
                "world_xyz": world_xyz,
                "voxel_xyz": voxel,
                "point_source": contact_point["point_source"],
                "num_points_used": contact_point["num_points_used"],
            }
        )

    return {
        "contact_mode": contact_mode,
        "source_view": f"{camera}_rgb_initial",
        "target_object": target_object or "unknown",
        "target_part": target_part or "unknown",
        "points_2d_normalized": points_2d_normalized,
        "candidate_contacts": projected,
        "use_as": "query_contact_hint_for_final_llm_not_retrieval_score",
    }


def load_episode_camera_arrays(
    episode_path: Path,
    *,
    camera: str,
    frame_index: int,
    xicm_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[Any, str]]:
    _ensure_xicm_imports(xicm_root)
    from utils import _image_to_float_array  # type: ignore

    depth_scale = 2**24 - 1
    demo = load_low_dim_obs(episode_path / "low_dim_obs.pkl")
    obs = demo[frame_index]
    mask = decode_rgb_mask(Image.open(episode_path / f"{camera}_mask" / f"{frame_index}.png"))
    cam_depth = _image_to_float_array(Image.open(episode_path / f"{camera}_depth" / f"{frame_index}.png"), depth_scale)
    near = obs.misc[f"{camera}_camera_near"]
    far = obs.misc[f"{camera}_camera_far"]
    cam_depth = (far - near) * cam_depth + near
    point_cloud = pointcloud_from_depth_and_camera_params(
        cam_depth,
        obs.misc[f"{camera}_camera_extrinsics"],
        obs.misc[f"{camera}_camera_intrinsics"],
    )
    mask_id_to_name = obs.misc[f"{camera}_mask_id_to_name"]
    return mask, point_cloud, mask_id_to_name


def parse_points(text: str) -> list[list[float]]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("points_2d_normalized") or data.get("points") or []
    points = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append([float(item[0]), float(item[1])])
    return points


def draw_overlay(
    image_path: Path,
    output_path: Path,
    result: dict[str, Any],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def label_position(x: int, y: int) -> tuple[int, int]:
        label_x = min(max(x + 7, 1), max(width - 46, 1))
        label_y = min(max(y - 13, 1), max(height - 12, 1))
        return label_x, label_y

    def outlined_text(pos: tuple[int, int], text: str, fill: str = "white") -> None:
        x, y = pos
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), text, fill="black")
        draw.text((x, y), text, fill=fill)

    for idx, contact in enumerate(result.get("candidate_contacts", []), start=1):
        x, y = contact["pixel_xy"]
        sx, sy = contact["snapped_pixel_xy"]
        draw.line((x, y, sx, sy), fill="yellow", width=2)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red", outline="white", width=2)
        draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill="lime", outline="black", width=2)
        draw.line((sx - 8, sy, sx + 8, sy), fill="black", width=1)
        draw.line((sx, sy - 8, sx, sy + 8), fill="black", width=1)
        outlined_text(label_position(sx, sy), str(idx), fill="white")
        mask = contact.get("mask_name") or "unknown"
        outlined_text(label_position(sx, sy + 12), str(mask)[:18], fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-path", required=True, help="Saved RLBench episode directory.")
    parser.add_argument("--points", required=True, help="JSON list or object with points_2d_normalized.")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--contact-mode", default="unknown")
    parser.add_argument("--target-object", default="unknown")
    parser.add_argument("--target-part", default="unknown")
    parser.add_argument("--search-radius", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay-output")
    parser.add_argument("--xicm-root", default=str(_repo_root() / "X-ICM"))
    args = parser.parse_args()

    episode_path = Path(args.episode_path)
    xicm_root = Path(args.xicm_root)
    points = parse_points(args.points)
    mask, point_cloud, mask_id_to_name = load_episode_camera_arrays(
        episode_path,
        camera=args.camera,
        frame_index=args.frame_index,
        xicm_root=xicm_root,
    )
    result = project_points_from_arrays(
        points_2d_normalized=points,
        mask=mask,
        point_cloud=point_cloud,
        mask_id_to_name=mask_id_to_name,
        camera=args.camera,
        contact_mode=args.contact_mode,
        target_object=args.target_object,
        target_part=args.target_part,
        search_radius=args.search_radius,
        window_radius=args.window_radius,
        xicm_root=xicm_root,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")

    if args.overlay_output:
        draw_overlay(
            episode_path / f"{args.camera}_rgb" / f"{args.frame_index}.png",
            Path(args.overlay_output),
            result,
        )
    print(out_path)


if __name__ == "__main__":
    main()
