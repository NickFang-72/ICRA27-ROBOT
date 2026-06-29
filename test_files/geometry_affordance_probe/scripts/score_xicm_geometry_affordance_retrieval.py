#!/usr/bin/env python3
"""Rank seen demonstrations with dynamic and primitive-geometry scores.

This script is intentionally outside the vanilla X-ICM baseline path. The
original ``lang_vis.out`` retrieval remains reproducible; this file prepares
rankings for the clean geometry/contact-hint ablation only.

Contact points and target-pose descriptors are deliberately prompt-only. They
can be passed through for inspection or final LLM prompting, but they must not
change the retrieval score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable


SEEN_TASKS = {
    "turn_tap",
    "close_jar",
    "light_bulb_in",
    "slide_block_to_color_target",
    "sweep_to_dustpan_of_size",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_money_in_safe",
    "place_shape_in_shape_sorter",
    "put_item_in_drawer",
    "insert_onto_square_peg",
    "open_drawer",
    "reach_and_drag",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "meat_off_grill",
    "stack_cups",
}

GEOMETRY_FIELDS = [
    "action_primitive",
    "motion_type",
    "motion_axis",
    "contact_type",
    "contact_region",
    "constraint_type",
    "alignment_requirement",
]

GEOMETRY_FIELD_WEIGHTS = {
    "action_primitive": 0.45,
    "motion_type": 0.15,
    "motion_axis": 0.15,
    "contact_type": 0.08,
    "contact_region": 0.07,
    "constraint_type": 0.07,
    "alignment_requirement": 0.03,
}
ACTION_MISMATCH_SCORE_CAP = 0.25


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def iter_rows(path: str | Path) -> list[dict[str, Any]]:
    row_path = Path(path)
    if row_path.is_dir():
        bundle = row_path / "review_bundle.jsonl"
        if bundle.exists():
            row_path = bundle
        else:
            index = row_path / "review_index.json"
            if index.exists():
                row_path = index
    if row_path.suffix == ".jsonl":
        return [json.loads(line) for line in row_path.read_text().splitlines() if line.strip()]
    if row_path.suffix == ".csv":
        with row_path.open(newline="") as f:
            return list(csv.DictReader(f))
    payload = load_json(row_path)
    if isinstance(payload, list):
        return payload
    for key in ("ranked", "candidates", "rows", "selected", "dynamic_scores"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    raise ValueError(f"Cannot find rows in {row_path}")


def episode_id_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"episode(\d+)$", str(path))
    return int(match.group(1)) if match else None


def task_from_path(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if len(parts) >= 4 and parts[-3:] and parts[-3] == "all_variations":
        return parts[-4]
    if len(parts) >= 4 and parts[-2] == "episodes":
        return parts[-4]
    return None


def infer_task(row: dict[str, Any]) -> str | None:
    task = row.get("task") or row.get("task_name")
    if task:
        return str(task)
    episode_path = row.get("episode_path") or row.get("demo_path")
    task = task_from_path(episode_path)
    if task:
        return task
    demo_id = str(row.get("demo_id") or "")
    if "_episode" in demo_id:
        return demo_id.split("_episode", 1)[0]
    return None


def infer_episode_id(row: dict[str, Any]) -> int | None:
    for key in ("episode_id", "episode"):
        value = row.get(key)
        if value not in (None, ""):
            return int(value)
    for key in ("episode_path", "demo_path"):
        episode_id = episode_id_from_path(row.get(key))
        if episode_id is not None:
            return episode_id
    demo_id = str(row.get("demo_id") or "")
    match = re.search(r"_episode(\d+)$", demo_id)
    return int(match.group(1)) if match else None


def canonical_demo_id(row: dict[str, Any]) -> str | None:
    demo_id = row.get("demo_id") or row.get("id")
    if demo_id:
        return str(demo_id)
    task = infer_task(row)
    episode_id = infer_episode_id(row)
    if task is not None and episode_id is not None:
        return f"{task}_episode{episode_id}"
    return None


def row_keys(row: dict[str, Any]) -> set[str]:
    keys = set()
    for key in ("demo_id", "id", "scene_id", "episode_path", "demo_path", "demo_dir"):
        value = row.get(key)
        if value:
            keys.add(str(value))
            if key in {"episode_path", "demo_path", "demo_dir"}:
                keys.add(Path(str(value)).name)
    demo_id = canonical_demo_id(row)
    if demo_id:
        keys.add(demo_id)
    task = infer_task(row)
    episode_id = infer_episode_id(row)
    if task is not None and episode_id is not None:
        keys.add(f"{task}:{episode_id}")
        keys.add(f"{task}:episode{episode_id}")
        keys.add(f"{task}_episode{episode_id}")
    return {key for key in keys if key}


def load_descriptor_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in iter_rows(path):
        for key in row_keys(row):
            cache[key] = row
    return cache


def dynamic_score_from_row(row: dict[str, Any]) -> float:
    for key in ("s_dyn", "dynamic_score", "similarity", "score"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"Dynamic score row is missing s_dyn/dynamic_score/similarity/score: {row}")


def load_dynamic_score_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in iter_rows(path):
        row = dict(row)
        row["s_dyn_raw"] = dynamic_score_from_row(row)
        rows.append(row)
    return rows


def load_dynamic_score_rows_from_xicm(
    xicm_root: str | Path,
    query_image: str | Path,
    query_task_instruction: str,
    dynamics_features_pkl: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compute the original X-ICM dynamic similarity on CAIR.

    This imports the X-ICM helper lazily so local syntax checks do not require
    the X-ICM runtime dependencies.
    """

    xicm_root = Path(xicm_root)
    sys.path.insert(0, str(xicm_root))
    import numpy as np  # pylint: disable=import-outside-toplevel
    from rlbench_inference_dynamics_diffusion import extract_diffusion_features  # pylint: disable=import-error,import-outside-toplevel

    if dynamics_features_pkl is None:
        dynamics_features_pkl = xicm_root / "data/dynamics_diffusion/all_diffusion_features.pkl"
    with Path(dynamics_features_pkl).open("rb") as f:
        all_diffusion_features = pickle.load(f)

    _, query_output_image_feat, query_prompt_feat = extract_diffusion_features(
        str(query_image), query_task_instruction
    )
    query_feat = np.concatenate([query_prompt_feat, query_output_image_feat])
    memory_feat = np.concatenate(
        [
            all_diffusion_features["all_prompt_feats"],
            all_diffusion_features["all_output_image_feats"],
        ],
        axis=1,
    )
    similarities = np.dot(memory_feat, query_feat)
    rows = []
    for episode_path, score in zip(all_diffusion_features["all_demo_paths"], similarities):
        row = {
            "episode_path": episode_path,
            "task": task_from_path(episode_path),
            "episode_id": episode_id_from_path(episode_path),
            "s_dyn_raw": float(score),
        }
        row["demo_id"] = canonical_demo_id(row)
        rows.append(row)
    return rows


def normalize_token(text: str) -> list[str]:
    compact = str(text).strip().lower()
    compact = compact.replace(" ", "_")
    parts = [part for part in re.split(r"[^a-z0-9]+", compact) if part]
    if compact and compact != "unknown":
        parts.append(compact)
    return parts


def flatten_descriptor_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_descriptor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (int, float)):
                continue
            yield from flatten_descriptor_values(item)
    elif isinstance(value, str):
        for token in normalize_token(value):
            yield token
    elif not isinstance(value, (int, float, bool)):
        for token in normalize_token(str(value)):
            yield token


def descriptor_tokens(descriptor: dict[str, Any], fields: Iterable[str]) -> set[str]:
    tokens = set()
    for field in fields:
        tokens.update(flatten_descriptor_values(descriptor.get(field)))
    return {token for token in tokens if token not in {"none", "unknown", "null"}}


def first_label(value: Any, default: str = "unknown") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace(" ", "_")
    if isinstance(value, (list, tuple)):
        for item in value:
            label = first_label(item, "")
            if label:
                return label
    return default


def as_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower().replace(" ", "_")] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        labels: list[str] = []
        for item in value:
            labels.extend(as_labels(item))
        return labels
    return []


def action_from_legacy_affordance(affordance: dict[str, Any]) -> str:
    action = first_label(
        affordance.get("motion_affordance")
        or affordance.get("action_primitive")
        or affordance.get("contact_affordance"),
        "unknown",
    )
    mapping = {
        "rotate": "twist",
        "rotate_part": "twist",
        "lift_and_place": "place",
        "lift_then_place": "place",
        "lift_then_insert": "insert",
        "insert_then_twist": "insert",
        "push_surface": "push",
        "pull_handle": "pull",
    }
    return mapping.get(action, action)


def action_from_task(task: str | None, action: str) -> str:
    task_text = first_label(task, "")
    if "push_buttons" in task_text:
        return "press"
    if "sweep_to_dustpan" in task_text:
        return "sweep"
    return action


def normalize_target_part(value: Any) -> str:
    part = first_label(value, "unknown")
    if part in {"handle", "lid", "rim", "knob", "button_top", "slot", "hole", "opening", "body", "edge", "spout", "socket", "surface", "unknown"}:
        return part
    if "button" in part:
        return "button_top"
    if "handle" in part:
        return "handle"
    if "knob" in part:
        return "knob"
    if "rim" in part:
        return "rim"
    if "edge" in part:
        return "edge"
    if "slot" in part:
        return "slot"
    if "hole" in part or "socket" in part:
        return "hole"
    if "opening" in part:
        return "opening"
    if "side" in part or "surface" in part:
        return "surface"
    if "body" in part or "base" in part:
        return "body"
    return "unknown"


def normalize_primary_shape(value: Any, text: str) -> str:
    shape = first_label(value, "")
    text = " ".join([shape, text.lower()])
    if "push_buttons" in text:
        return "button"
    if "insert_onto_square_peg" in text:
        return "peg"
    if "close_jar" in text or "light_bulb_in" in text:
        return "round_object"
    if any(token in text for token in ["open_drawer", "put_item_in_drawer", "put_money_in_safe", "put_groceries_in_cupboard", "place_shape_in_shape_sorter", "slide_block_to_color_target"]):
        return "box_like"
    if "sweep_to_dustpan" in text:
        return "thin_flat_object"
    if "turn_tap" in text:
        return "elongated_tool"
    if "button" in text:
        return "button"
    if "peg" in text:
        return "peg"
    if any(token in text for token in ["drawer", "safe", "cupboard", "block", "box", "rectangular", "cube"]):
        return "box_like"
    if any(token in text for token in ["money", "thin", "flat", "plate"]):
        return "thin_flat_object"
    if any(token in text for token in ["tap", "handle", "spatula", "tool", "sweeping"]):
        return "elongated_tool"
    if any(token in text for token in ["jar", "bulb", "round", "circular", "cylinder", "cylindrical", "rim"]):
        return "round_object"
    if any(token in text for token in ["shape_sorter", "shape_profile", "irregular"]):
        return "irregular"
    if shape in {"round_object", "box_like", "thin_flat_object", "elongated_tool", "button", "peg", "irregular", "unknown"}:
        return shape
    return "unknown"


def normalize_constraint_type(value: Any, action: str, tags: Any) -> str:
    constraint = first_label(value, "unknown")
    tag_set = set(as_labels(tags))
    if action in {"twist", "rotate"} or tag_set.intersection({"hinge", "sliding_axis", "rotational_axis"}):
        return "joint"
    if "slot" in constraint:
        return "slot"
    if "hole" in constraint or "socket" in constraint:
        return "hole"
    if "opening" in constraint or constraint in {"receptacle", "open_container", "closed_container"}:
        return "container"
    if constraint in {"support_cradle", "support"}:
        return "support_surface"
    if constraint in {"slot", "hole", "container", "joint", "support_surface", "surface_target", "free_space", "none", "unknown"}:
        return constraint
    return "none" if constraint in {"", "unknown"} else constraint


def normalize_state(value: Any) -> str:
    state = first_label(value, "unknown")
    if state in {"open", "closed", "attached", "detached", "inside", "on_surface", "free", "unknown"}:
        return state
    if "open" in state:
        return "open"
    if "closed" in state:
        return "closed"
    if "attached" in state or "grasp" in state or "held" in state:
        return "attached"
    if "inside" in state:
        return "inside"
    if "surface" in state:
        return "on_surface"
    if "free" in state:
        return "free"
    return "unknown"


def motion_type_from_axis(axis: Any, action: str) -> str:
    axis = first_label(axis, "unknown")
    if action in {"twist", "rotate"} or "rotat" in axis or "tilt" in axis:
        return "rotational"
    if action == "insert":
        return "insertion"
    if action in {"place", "stack", "lift"}:
        return "vertical"
    if action in {"slide", "sweep", "drag"}:
        return "planar"
    if action in {"push", "pull", "press"}:
        return "linear"
    if axis in {"horizontal", "vertical", "linear", "front_normal"}:
        return "linear"
    return "unknown"


def motion_axis_from_axis(axis: Any, action: str) -> str:
    axis = first_label(axis, "unknown")
    if action in {"twist", "rotate"} or "rotat" in axis or "tilt" in axis:
        return "rotational"
    if action == "insert" or axis in {"slot", "hole", "front_opening", "top_opening", "support_cradle"}:
        return "into_opening"
    if action in {"slide", "sweep", "drag"} or axis == "free_motion":
        return "across_surface"
    if action in {"place", "stack", "lift"} or axis == "vertical":
        return "vertical"
    if action in {"push", "pull"} or axis in {"horizontal", "linear"}:
        return "horizontal"
    if action == "press":
        return "surface_normal"
    return "unknown"


def constraint_from_opening(opening: Any, affordance: dict[str, Any]) -> str:
    opening = first_label(opening, "unknown")
    containment = first_label(affordance.get("containment_affordance"), "")
    articulation = first_label(affordance.get("articulation_affordance"), "")
    if any(token in opening for token in ["slot", "hole", "opening"]):
        return opening
    if containment in {"slot", "hole", "receptacle", "open_container", "closed_container"}:
        return containment
    if "hinge" in articulation or "drawer" in articulation:
        return "joint"
    if opening == "support_cradle":
        return "support_surface"
    if opening in {"none", "unknown"}:
        return "none"
    return opening


def contact_type_from_action(action: str) -> str:
    if action in {"pull", "lift", "place", "insert", "stack", "twist"}:
        return "grasp"
    if action == "press":
        return "press"
    if action in {"push", "slide", "sweep", "drag"}:
        return "surface_contact"
    return "unknown"


def alignment_from_geometry(geometry: dict[str, Any], affordance: dict[str, Any]) -> str:
    text = json.dumps({"geometry": geometry, "affordance": affordance}, sort_keys=True).lower()
    if any(token in text for token in ["alignment_sensitive", "misalignment", "slot", "hole", "peg", "socket", "shape_profile"]):
        return "high"
    if any(token in text for token in ["target", "rack", "shelf", "hoop", "rim"]):
        return "medium"
    return "low"


def object_category_from_text(text: str) -> str:
    text = text.lower()
    if any(token in text for token in ["drawer", "door", "lid", "hinge", "knob", "tap", "switch", "button"]):
        return "articulated_or_control"
    if any(token in text for token in ["slot", "hole", "socket", "peg", "stand", "shape_sorter"]):
        return "alignment_target"
    if any(token in text for token in ["cupboard", "bin", "shelf", "rack", "hoop", "container", "safe"]):
        return "receptacle_or_support"
    if any(token in text for token in ["rope", "spatula", "knife", "umbrella", "charger", "usb"]):
        return "elongated_or_tool"
    return "rigid_object"


def canonical_geometry(task: str | None, geometry: dict[str, Any], affordance: dict[str, Any] | None = None) -> dict[str, Any]:
    affordance = affordance or {}
    geometry = geometry or {}
    action = first_label(geometry.get("action_primitive"), "")
    if not action:
        action = action_from_legacy_affordance(affordance)
    action = action_from_task(task, action)
    old_tags = as_labels(geometry.get("geometry_tags")) or as_labels(geometry.get("task_relevant_geometric_cues")) or as_labels(geometry.get("key_features"))
    old_tags = [
        tag for tag in old_tags
        if tag not in {"left", "right", "left_orientation", "right_orientation", "downward_direction", "upward_direction", "color", "red", "green", "blue", "maroon", "robot_arm", "elbow", "wrist", "static", "standing"}
    ]
    target_part = normalize_target_part(
        geometry.get("target_part")
        or affordance.get("required_contact_region")
        or geometry.get("part_geometry")
    )
    manipulated_object = first_label(geometry.get("manipulated_object"), task or "unknown")
    text = " ".join([task or "", manipulated_object, target_part, " ".join(old_tags)])
    constraint_type = normalize_constraint_type(
        first_label(geometry.get("constraint_type"), constraint_from_opening(geometry.get("opening_geometry"), affordance)),
        action,
        old_tags,
    )
    out = {
        "manipulated_object": manipulated_object,
        "object_category": first_label(geometry.get("object_category"), object_category_from_text(text)),
        "primary_shape": normalize_primary_shape(geometry.get("primary_shape"), text),
        "target_part": target_part,
        "secondary_parts": as_labels(geometry.get("secondary_parts")) or as_labels(geometry.get("part_geometry"))[:3],
        "action_primitive": action,
        "motion_type": first_label(geometry.get("motion_type"), motion_type_from_axis(geometry.get("axis_geometry"), action)),
        "motion_axis": first_label(geometry.get("motion_axis"), motion_axis_from_axis(geometry.get("axis_geometry") or geometry.get("opening_geometry"), action)),
        "contact_type": first_label(geometry.get("contact_type"), contact_type_from_action(action)),
        "contact_region": first_label(geometry.get("contact_region") or affordance.get("required_contact_region"), target_part),
        "constraint_type": constraint_type,
        "alignment_requirement": first_label(geometry.get("alignment_requirement"), alignment_from_geometry(geometry, affordance)),
        "state": normalize_state(geometry.get("state") or geometry.get("pose_relation")),
        "geometry_tags": sorted(set(old_tags + as_labels(geometry.get("geometry_tags")))),
    }
    if geometry.get("execution_clearance_hint") or geometry.get("clearance_geometry"):
        out["execution_clearance_hint"] = first_label(
            geometry.get("execution_clearance_hint") or geometry.get("clearance_geometry"),
            "unknown",
        )
    return out


def field_similarity(seen: dict[str, Any], query: dict[str, Any], field: str) -> float:
    if field == "geometry_tags":
        return jaccard(set(as_labels(seen.get(field))), set(as_labels(query.get(field))))
    return jaccard(set(flatten_descriptor_values(seen.get(field))), set(flatten_descriptor_values(query.get(field))))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def points(value: Any) -> list[tuple[float, float]]:
    out = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                x = float(item[0])
                y = float(item[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                out.append((x, y))
    return out


def point_similarity(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a or not b:
        return 0.0
    distances = []
    for ax, ay in a:
        distances.append(min(math.dist((ax, ay), (bx, by)) for bx, by in b))
    mean_distance = sum(distances) / len(distances)
    return max(0.0, 1.0 - mean_distance / math.sqrt(2.0))


def geometry_similarity(seen: dict[str, Any], query: dict[str, Any]) -> float:
    seen = canonical_geometry(None, seen)
    query = canonical_geometry(None, query)
    score = 0.0
    total = 0.0
    for field, weight in GEOMETRY_FIELD_WEIGHTS.items():
        score += weight * field_similarity(seen, query, field)
        total += weight
    normalized = score / total if total else 0.0
    seen_action = first_label(seen.get("action_primitive"), "")
    query_action = first_label(query.get("action_primitive"), "")
    if seen_action and query_action and seen_action != query_action and "unknown" not in {seen_action, query_action}:
        return min(normalized, ACTION_MISMATCH_SCORE_CAP)
    return normalized


def affordance_similarity(seen: dict[str, Any], query: dict[str, Any]) -> float:
    return 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def descriptor_for_candidate(
    candidate: dict[str, Any],
    descriptor_cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for key in row_keys(candidate):
        if key in descriptor_cache:
            return descriptor_cache[key]
    return None


def assert_seen_only(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    filtered = []
    bad_tasks = set()
    for row in rows:
        task = infer_task(row)
        if task in SEEN_TASKS:
            filtered.append(row)
        else:
            bad_tasks.add(str(task))
    if bad_tasks and mode == "error":
        raise ValueError(f"Found non-seen or unknown task rows: {sorted(bad_tasks)}")
    return filtered


def rank_candidates(
    dynamic_rows: list[dict[str, Any]],
    descriptor_cache: dict[str, dict[str, Any]],
    query_geometry: dict[str, Any],
    query_affordance: dict[str, Any],
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
    seen_only_action: str = "error",
    exclude_demo_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude_demo_ids = exclude_demo_ids or set()
    rows = assert_seen_only(dynamic_rows, seen_only_action)
    scored = []
    for row in rows:
        demo_id = canonical_demo_id(row)
        if demo_id and demo_id in exclude_demo_ids:
            continue
        descriptor = descriptor_for_candidate(row, descriptor_cache)
        if descriptor is None:
            continue
        contact_hints = descriptor.get("contact_hints_i") or descriptor.get("affordance_a_i") or descriptor.get("affordance") or {}
        geometry = canonical_geometry(
            infer_task(row),
            descriptor.get("geometry_g_i") or descriptor.get("geometry") or {},
            contact_hints,
        )
        s_dyn_raw = row.get("s_dyn_raw")
        if s_dyn_raw in (None, ""):
            s_dyn_raw = dynamic_score_from_row(row)
        enriched = {
            **row,
            "demo_id": demo_id,
            "task": infer_task(row),
            "episode_id": infer_episode_id(row),
            "episode_path": row.get("episode_path") or descriptor.get("episode_path"),
            "language_description": descriptor.get("language_description") or row.get("language_description"),
            "s_dyn_raw": float(s_dyn_raw),
            "s_geo": geometry_similarity(geometry, query_geometry),
            "s_aff": 0.0,
            "geometry_g_i": geometry,
            "contact_hints_i": contact_hints,
        }
        scored.append(enriched)

    s_dyn_values = minmax([row["s_dyn_raw"] for row in scored])
    for row, s_dyn in zip(scored, s_dyn_values):
        row["s_dyn"] = s_dyn
        row["score"] = alpha * row["s_dyn"] + beta * row["s_geo"]

    ranked = sorted(scored, key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked[:top_k]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-cache", required=True, help="review_bundle.jsonl, review_index.json, or cache root")
    parser.add_argument("--dynamic-scores", help="JSON/JSONL/CSV rows containing one S_dyn score per seen demo")
    parser.add_argument("--xicm-root", help="Optional X-ICM root for computing S_dyn directly on CAIR")
    parser.add_argument("--query-image", help="Current query front image for direct X-ICM S_dyn extraction")
    parser.add_argument("--query-task-instruction", help="Query language instruction for direct X-ICM S_dyn extraction")
    parser.add_argument("--dynamics-features-pkl", help="Optional all_diffusion_features.pkl override")
    parser.add_argument("--query-geometry", required=True, help="JSON file containing g_j or geometry_g_j")
    parser.add_argument("--query-affordance", help="Legacy/optional JSON file containing contact hints; ignored for retrieval scoring")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0, help="Ignored in clean v1; kept for CLI compatibility")
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--seen-only-action", choices=["error", "filter"], default="error")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode-list-out", help="Optional newline-delimited top-k episode paths")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    descriptor_cache = load_descriptor_cache(args.descriptor_cache)
    if args.dynamic_scores:
        dynamic_rows = load_dynamic_score_rows(args.dynamic_scores)
    else:
        required = [args.xicm_root, args.query_image, args.query_task_instruction]
        if not all(required):
            raise SystemExit("--dynamic-scores or --xicm-root/--query-image/--query-task-instruction is required")
        dynamic_rows = load_dynamic_score_rows_from_xicm(
            args.xicm_root,
            args.query_image,
            args.query_task_instruction,
            args.dynamics_features_pkl,
        )

    query_geometry_doc = load_json(args.query_geometry)
    query_affordance_doc = load_json(args.query_affordance) if args.query_affordance else {}
    query_geometry = query_geometry_doc.get("geometry_g_j") or query_geometry_doc.get("geometry_g_i") or query_geometry_doc
    query_affordance = query_affordance_doc.get("affordance_a_j") or query_affordance_doc.get("affordance_a_i") or query_affordance_doc
    query_geometry = canonical_geometry(None, query_geometry, query_affordance)

    ranked = rank_candidates(
        dynamic_rows,
        descriptor_cache,
        query_geometry,
        query_affordance,
        args.alpha,
        args.beta,
        args.gamma,
        args.top_k,
        args.seen_only_action,
    )
    payload = {
        "formula": "score(D_i,Q_j)=alpha*S_dyn(D_i,Q_j)+beta*S_geo(g_i,g_j)",
        "weights": {"alpha": args.alpha, "beta": args.beta, "gamma_ignored": args.gamma},
        "top_k": args.top_k,
        "component_notes": {
            "S_dyn": "Original X-ICM dynamic similarity, min-max normalized per query across seen candidates.",
            "S_geo": "Weighted similarity over normalized primitive geometry/action fields.",
            "S_aff": "Removed in clean v1. RoboPoint/contact hints are final-prompt evidence only, not retrieval scoring.",
        },
        "ranked": ranked,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    if args.episode_list_out:
        paths = [str(row["episode_path"]) for row in ranked if row.get("episode_path")]
        Path(args.episode_list_out).write_text("\n".join(paths) + ("\n" if paths else ""))
    print(json.dumps({"out": args.out, "ranked": len(ranked), "top_demo": ranked[0]["demo_id"] if ranked else None}, indent=2))


if __name__ == "__main__":
    main()
