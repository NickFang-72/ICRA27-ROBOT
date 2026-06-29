#!/usr/bin/env python3
"""Build a front/overhead oracle contact-point consistency review bundle.

This is a review-only oracle check. It selects role-specific simulator masks
from both the front and overhead initial RGB frames, projects both selections
into the shared RLBench world frame, compares the projected 3D/voxel points,
and then chooses an overhead point when the views disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import build_oracle_overhead_contact_check as oracle


CONFIRMED_WORLD_THRESHOLD_M = 0.08
CONFIRMED_VOXEL_THRESHOLD = 8.0


def _normalize_role_fields(roles: list[dict[str, Any]], camera: str) -> list[dict[str, Any]]:
    out = []
    for role in roles:
        item = dict(role)
        item["source_view"] = f"{camera}_rgb_initial"
        item["source_model"] = f"mask_oracle_{camera}_centroid"
        old_key = "point_2d_normalized_overhead"
        new_key = f"point_2d_normalized_{camera}"
        if old_key in item:
            item[new_key] = item[old_key]
            if camera != "overhead":
                del item[old_key]
        if "orientation_radians_topdown" in item:
            item["orientation_radians_image"] = item.pop("orientation_radians_topdown")
        out.append(item)
    return out


def build_camera_roles(
    *,
    episode_path: Path,
    task: str,
    instruction: str,
    camera: str,
    frame_index: int,
    window_radius: int,
) -> list[dict[str, Any]]:
    mask, point_cloud, mask_id_to_name = oracle.load_camera_arrays(episode_path, camera, frame_index)
    roles = oracle.build_role_points(
        task=task,
        instruction=instruction,
        mask=mask,
        point_cloud=point_cloud,
        mask_id_to_name=mask_id_to_name,
        camera=camera,
        window_radius=window_radius,
    )
    return _normalize_role_fields(roles, camera)


def role_group_key(role: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(role.get("role", "")),
        str(role.get("target_object", "")),
        str(role.get("target_part", "")),
    )


def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _match_front_role(
    overhead_role: dict[str, Any],
    front_roles: list[dict[str, Any]],
    used_front: set[int],
) -> tuple[int | None, dict[str, Any] | None, float | None, float | None]:
    key = role_group_key(overhead_role)
    candidates = []
    for idx, front_role in enumerate(front_roles):
        if idx in used_front or role_group_key(front_role) != key:
            continue
        world_distance = _distance(overhead_role.get("world_xyz"), front_role.get("world_xyz"))
        voxel_distance = _distance(overhead_role.get("voxel_xyz"), front_role.get("voxel_xyz"))
        sort_distance = world_distance if world_distance is not None else float("inf")
        candidates.append((sort_distance, idx, front_role, world_distance, voxel_distance))
    if not candidates:
        return None, None, None, None
    candidates.sort(key=lambda item: item[0])
    _, idx, front_role, world_distance, voxel_distance = candidates[0]
    return idx, front_role, world_distance, voxel_distance


def choose_final_roles(
    front_roles: list[dict[str, Any]],
    overhead_roles: list[dict[str, Any]],
    *,
    world_threshold_m: float,
    voxel_threshold: float,
) -> list[dict[str, Any]]:
    used_front: set[int] = set()
    final_roles = []
    for final_index, overhead_role in enumerate(overhead_roles, start=1):
        front_idx, front_role, world_distance, voxel_distance = _match_front_role(
            overhead_role,
            front_roles,
            used_front,
        )
        if front_idx is not None:
            used_front.add(front_idx)
        confirmed = (
            world_distance is not None
            and voxel_distance is not None
            and world_distance <= world_threshold_m
            and voxel_distance <= voxel_threshold
        )
        if front_role is None:
            status = "overhead_only_no_front_match"
        elif confirmed:
            status = "confirmed_by_front_and_overhead"
        else:
            status = "disagreed_use_overhead"
        final_roles.append(
            {
                "final_index": final_index,
                "role": overhead_role.get("role"),
                "target_object": overhead_role.get("target_object"),
                "target_part": overhead_role.get("target_part"),
                "contact_mode": overhead_role.get("contact_mode"),
                "chosen_source_view": "overhead_rgb_initial",
                "selection_status": status,
                "world_distance_m": world_distance,
                "voxel_distance": voxel_distance,
                "chosen_pixel_xy_overhead": overhead_role.get("pixel_xy"),
                "chosen_world_xyz": overhead_role.get("world_xyz"),
                "chosen_voxel_xyz": overhead_role.get("voxel_xyz"),
                "overhead": overhead_role,
                "front_match": front_role,
                "human_review": {
                    "front_point_correct": None,
                    "overhead_point_correct": None,
                    "final_point_correct": None,
                    "useful_for_prompt": None,
                    "notes": "",
                },
            }
        )
    return final_roles


def copy_image(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def draw_overlay(
    image_path: Path,
    output_path: Path,
    roles: list[dict[str, Any]],
    *,
    label_key: str = "pixel_xy",
    final: bool = False,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "manipulated_object_contact": "red",
        "goal_region": "lime",
        "secondary_object_to_move": "deepskyblue",
        "constraint_region": "orange",
    }
    status_outline = {
        "confirmed_by_front_and_overhead": "white",
        "disagreed_use_overhead": "yellow",
        "overhead_only_no_front_match": "magenta",
    }
    width, height = image.size

    def outlined_text(x: int, y: int, text: str, fill: str = "white") -> None:
        x = min(max(x, 2), max(width - 20, 2))
        y = min(max(y, 2), max(height - 14, 2))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), text, fill="black")
        draw.text((x, y), text, fill=fill)

    for idx, role in enumerate(roles, start=1):
        if final:
            point = role.get("chosen_pixel_xy_overhead")
            role_name = role.get("role")
            outline = status_outline.get(str(role.get("selection_status")), "black")
        else:
            point = role.get(label_key)
            role_name = role.get("role")
            outline = "black"
        if not point:
            continue
        x, y = int(point[0]), int(point[1])
        color = colors.get(str(role_name), "yellow")
        radius = 6
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline, width=3)
        draw.line((x - 10, y, x + 10, y), fill="black", width=1)
        draw.line((x, y - 10, x, y + 10), fill="black", width=1)
        theta = role.get("orientation_radians_image")
        if theta is not None and not final:
            dx = math.cos(theta) * 18
            dy = math.sin(theta) * 18
            draw.line((x - dx, y - dy, x + dx, y + dy), fill=color, width=3)
        outlined_text(x + 8, y - 8, str(idx))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _resize(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    return image.resize((width, int(image.height * scale)))


def _record_key_lines(record: dict[str, Any]) -> list[str]:
    lines = []
    for role in record["multiview_contact_hints"]["final_roles"]:
        idx = role["final_index"]
        status = role["selection_status"].replace("_", " ")
        dist = role.get("world_distance_m")
        dist_text = "no front match" if dist is None else f"{dist:.3f}m"
        voxel = role.get("chosen_voxel_xyz")
        target = f"{role.get('role')}:{role.get('target_object')}"
        lines.append(f"{idx}. {target} | {status} | dist {dist_text} | voxel {voxel}")
    return lines


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path, thumb_width: int = 180) -> None:
    tiles = []
    panel_names = [
        ("front_image", "front raw"),
        ("front_overlay_image", "front points"),
        ("overhead_image", "top raw"),
        ("overhead_overlay_image", "top points"),
        ("final_overlay_image", "final chosen"),
    ]
    gutter = 8
    line_height = 14
    for record in records:
        panels = []
        for key, title in panel_names:
            image = _resize(Image.open(record[key]).convert("RGB"), thumb_width)
            panel = Image.new("RGB", (image.width, image.height + 18), "white")
            panel.paste(image, (0, 18))
            draw = ImageDraw.Draw(panel)
            draw.text((4, 2), title, fill="black")
            draw.rectangle((0, 18, image.width - 1, panel.height - 1), outline="black")
            panels.append(panel)
        key_lines = _record_key_lines(record)
        panel_height = max(panel.height for panel in panels)
        key_height = 24 + max(1, len(key_lines)) * line_height
        tile_width = len(panels) * thumb_width + (len(panels) - 1) * gutter
        tile = Image.new("RGB", (tile_width, panel_height + key_height), "white")
        x = 0
        for panel in panels:
            tile.paste(panel, (x, 0))
            x += thumb_width + gutter
        draw = ImageDraw.Draw(tile)
        y = panel_height + 4
        draw.text((5, y), f"{record['task']} / {record['demo_id']}"[:90], fill="black")
        y += 16
        for line in key_lines:
            draw.text((8, y), line[:126], fill="black")
            y += line_height
        tiles.append(tile)
    if not tiles:
        return
    sheet_width = max(tile.width for tile in tiles)
    sheet_height = sum(tile.height for tile in tiles)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def markdown_link(path: str | Path, label: str | None = None, image: bool = False) -> str:
    path = Path(path).resolve()
    label = label or path.name
    prefix = "!" if image else ""
    return f"{prefix}[{label}]({path})"


def write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path.resolve())


def build_markdown(output_dir: Path, records: list[dict[str, Any]], bundle_path: Path, sheet_path: Path) -> None:
    lines = [
        "# Oracle Multi-View Contact/Goal Point Check",
        "",
        "This bundle compares front and overhead oracle mask-centroid points after projecting both into the same RLBench world frame and X-ICM voxel grid.",
        "If the projected points agree, the final point is marked confirmed. If they disagree or the front point is missing, the final point falls back to the overhead point.",
        "",
        f"- Contact sheet: {markdown_link(sheet_path, image=True)}",
        f"- JSON bundle: {markdown_link(bundle_path)}",
        "",
        "Marker colors: red = manipulated object/contact, green = goal region, blue = secondary object to move. In the final panel, white outline = confirmed, yellow outline = disagreed/use overhead, magenta outline = overhead-only.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['task']} / {record['demo_id']}",
                "",
                f"- Instruction: `{record['language_description']}`",
                f"- Per-demo JSON: {markdown_link(record['json_path'])}",
                f"- Front raw: {markdown_link(record['front_image'], image=True)}",
                f"- Front points: {markdown_link(record['front_overlay_image'], image=True)}",
                f"- Overhead raw: {markdown_link(record['overhead_image'], image=True)}",
                f"- Overhead points: {markdown_link(record['overhead_overlay_image'], image=True)}",
                f"- Final chosen: {markdown_link(record['final_overlay_image'], image=True)}",
                "",
                "Final roles:",
            ]
        )
        for role in record["multiview_contact_hints"]["final_roles"]:
            lines.append(
                f"- `{role['final_index']}` `{role['role']}` `{role['target_object']}`: "
                f"`{role['selection_status']}`, world distance `{role['world_distance_m']}`, "
                f"chosen voxel `{role['chosen_voxel_xyz']}`"
            )
        lines.append("")
    (output_dir / "oracle_multiview_points.md").write_text("\n".join(lines).rstrip() + "\n")


def parse_demo_spec(value: str) -> tuple[str, str, int, str]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("demo must be id:task:episode:language")
    return parts[0], parts[1], int(parts[2].replace("episode", "")), parts[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/yf23/datasets/ICRA27-ROBOT")
    parser.add_argument("--output-dir", default="test_files/geometry_affordance_probe/review/figures/oracle_multiview_points")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--world-threshold-m", type=float, default=CONFIRMED_WORLD_THRESHOLD_M)
    parser.add_argument("--voxel-threshold", type=float, default=CONFIRMED_VOXEL_THRESHOLD)
    parser.add_argument("--demo", action="append", type=parse_demo_spec, help="id:task:episode:language. May be repeated.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    items_dir = output_dir / "items"
    records = []

    for demo_id, task, episode, language in args.demo or oracle.DEFAULT_DEMOS:
        episode_path = data_root / "seen_tasks" / task / "all_variations" / "episodes" / f"episode{episode}"
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)
        front_roles = build_camera_roles(
            episode_path=episode_path,
            task=task,
            instruction=language,
            camera="front",
            frame_index=args.frame_index,
            window_radius=args.window_radius,
        )
        overhead_roles = build_camera_roles(
            episode_path=episode_path,
            task=task,
            instruction=language,
            camera="overhead",
            frame_index=args.frame_index,
            window_radius=args.window_radius,
        )
        final_roles = choose_final_roles(
            front_roles,
            overhead_roles,
            world_threshold_m=args.world_threshold_m,
            voxel_threshold=args.voxel_threshold,
        )

        front_src = episode_path / "front_rgb" / f"{args.frame_index}.png"
        overhead_src = episode_path / "overhead_rgb" / f"{args.frame_index}.png"
        front_image = copy_image(front_src, images_dir / f"{demo_id}_front.png")
        overhead_image = copy_image(overhead_src, images_dir / f"{demo_id}_overhead.png")
        front_overlay_path = images_dir / f"{demo_id}_front_oracle_points.png"
        overhead_overlay_path = images_dir / f"{demo_id}_overhead_oracle_points.png"
        final_overlay_path = images_dir / f"{demo_id}_final_overhead_points.png"
        draw_overlay(front_src, front_overlay_path, front_roles)
        draw_overlay(overhead_src, overhead_overlay_path, overhead_roles)
        draw_overlay(overhead_src, final_overlay_path, final_roles, final=True)

        record = {
            "demo_id": demo_id,
            "task": task,
            "language_description": language,
            "episode_path": str(episode_path),
            "frame_index": args.frame_index,
            "front_image": front_image,
            "front_overlay_image": str(front_overlay_path.resolve()),
            "overhead_image": overhead_image,
            "overhead_overlay_image": str(overhead_overlay_path.resolve()),
            "final_overlay_image": str(final_overlay_path.resolve()),
            "multiview_contact_hints": {
                "source_model": "mask_oracle_front_overhead_centroid_consistency",
                "source_note": "Upper-bound oracle check using simulator mask names, not a learned VLM.",
                "world_threshold_m": args.world_threshold_m,
                "voxel_threshold": args.voxel_threshold,
                "front_roles": front_roles,
                "overhead_roles": overhead_roles,
                "final_roles": final_roles,
                "use_as": "query_execution_hint_not_retrieval",
            },
        }
        item_path = items_dir / f"{demo_id}.json"
        record["json_path"] = write_json(item_path, record)
        records.append(record)

    bundle_path = output_dir / "oracle_multiview_points_bundle.json"
    write_json(bundle_path, records)
    with (output_dir / "oracle_multiview_points_bundle.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    sheet_path = output_dir / "oracle_multiview_contact_sheet.png"
    make_contact_sheet(records, sheet_path)
    build_markdown(output_dir, records, bundle_path, sheet_path)
    print(f"Wrote {output_dir.resolve()}")
    print(f"Wrote {bundle_path.resolve()}")
    print(f"Wrote {(output_dir / 'oracle_multiview_points.md').resolve()}")


if __name__ == "__main__":
    main()
