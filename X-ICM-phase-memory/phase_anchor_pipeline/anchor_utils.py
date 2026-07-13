"""Utilities for scene anchors used by the phase-anchor compiler."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


VOXEL_MIN = 0
VOXEL_MAX = 99


def as_int_triplet(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        return None
    try:
        if len(value) != 3:
            return None
        return [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None


def clip_voxel(voxel_xyz: Iterable[int]) -> List[int]:
    return [max(VOXEL_MIN, min(VOXEL_MAX, int(value))) for value in voxel_xyz]


def add_voxel_offset(voxel_xyz: Iterable[int], offset_xyz: Iterable[int]) -> List[int]:
    return clip_voxel([int(a) + int(b) for a, b in zip(voxel_xyz, offset_xyz)])


def voxel_distance(a: Iterable[int], b: Iterable[int]) -> float:
    av = list(a)
    bv = list(b)
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(av, bv)))


def normalized_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def text_matches(candidate: Any, target: Any) -> bool:
    candidate_text = normalized_text(candidate)
    target_text = normalized_text(target)
    if not candidate_text or not target_text:
        return False
    return candidate_text in target_text or target_text in candidate_text


def _anchor_from_contact_point(point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    voxel = as_int_triplet(point.get("voxel_xyz"))
    if voxel is None:
        return None
    return {
        "source": "contact_hints_c_j.role_labeled_points",
        "index": point.get("index"),
        "role": point.get("role") or "unknown",
        "role_label": point.get("role_label"),
        "object": point.get("target_object"),
        "part": point.get("target_part"),
        "contact_mode": point.get("contact_mode"),
        "selection_status": point.get("selection_status"),
        "voxel_xyz": voxel,
        "world_xyz": point.get("world_xyz"),
        "source_view": point.get("source_view"),
        "quality": anchor_quality(point),
    }


def _anchor_from_role_binding(binding: Dict[str, Any], next_index: int) -> Optional[Dict[str, Any]]:
    voxel = as_int_triplet(binding.get("voxel_xyz"))
    if voxel is None:
        return None
    return {
        "source": "phase_interpreter.role_bindings",
        "index": binding.get("index") or binding.get("point_id") or next_index,
        "role": binding.get("role") or "unknown",
        "role_label": binding.get("role_label"),
        "object": binding.get("object") or binding.get("target_object"),
        "part": binding.get("part") or binding.get("target_part"),
        "contact_mode": binding.get("contact_mode"),
        "selection_status": "vlm_role_binding",
        "voxel_xyz": voxel,
        "world_xyz": binding.get("world_xyz"),
        "source_view": binding.get("source_view"),
        "quality": 0.45,
    }


def anchor_quality(point: Dict[str, Any]) -> float:
    status = normalized_text(point.get("selection_status"))
    quality = 0.45
    if "confirmed" in status:
        quality = 1.0
    elif "disagreed" in status:
        quality = 0.68
    elif "front_only" in status or "overhead_only" in status:
        quality = 0.55
    if point.get("world_distance_m") is not None:
        try:
            quality -= min(0.2, max(0.0, float(point["world_distance_m"]) - 0.03))
        except (TypeError, ValueError):
            pass
    return max(0.0, min(1.0, quality))


def extract_scene_anchors(
    contact_hints: Dict[str, Any],
    phase_output: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract role-labeled anchors from query contact hints and VLM bindings."""

    anchors: List[Dict[str, Any]] = []
    seen_keys = set()
    for point in contact_hints.get("role_labeled_points") or []:
        anchor = _anchor_from_contact_point(point)
        if anchor is None:
            continue
        key = (anchor["role"], tuple(anchor["voxel_xyz"]), normalized_text(anchor.get("object")))
        if key not in seen_keys:
            seen_keys.add(key)
            anchors.append(anchor)

    next_index = 1000
    if phase_output:
        for binding in phase_output.get("role_bindings") or []:
            anchor = _anchor_from_role_binding(binding, next_index)
            next_index += 1
            if anchor is None:
                continue
            key = (anchor["role"], tuple(anchor["voxel_xyz"]), normalized_text(anchor.get("object")))
            if key not in seen_keys:
                seen_keys.add(key)
                anchors.append(anchor)

    return sorted(
        anchors,
        key=lambda item: (
            normalized_text(item.get("role")) != "manipulated_object_contact",
            -float(item.get("quality") or 0.0),
            int(item.get("index") or 9999),
        ),
    )


def choose_anchor(
    anchors: List[Dict[str, Any]],
    role: str,
    *,
    target_object: Any = None,
    target_part: Any = None,
    preferred_index: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Choose the best anchor for a requested role/object/part."""

    normalized_role = normalized_text(role)
    if normalized_role in {"", "none", "null"}:
        return None, {"reason": "phase does not request an anchor"}

    best_anchor = None
    best_score = -1.0
    candidates = []
    for anchor in anchors:
        score = float(anchor.get("quality") or 0.0)
        if normalized_text(anchor.get("role")) == normalized_role:
            score += 3.0
        elif normalized_role == "goal_region" and normalized_text(anchor.get("role")) in {"constraint_reference"}:
            score += 0.3
        else:
            score -= 1.0
        if preferred_index is not None and str(anchor.get("index")) == str(preferred_index):
            score += 2.0
        if text_matches(anchor.get("object"), target_object):
            score += 0.8
        if text_matches(anchor.get("part"), target_part):
            score += 0.4
        candidates.append(
            {
                "index": anchor.get("index"),
                "role": anchor.get("role"),
                "object": anchor.get("object"),
                "part": anchor.get("part"),
                "score": round(score, 4),
            }
        )
        if score > best_score:
            best_score = score
            best_anchor = anchor

    if best_anchor is None or best_score < 0.0:
        return None, {
            "reason": f"no usable anchor for role={role}",
            "candidates": candidates,
        }
    return best_anchor, {
        "reason": "best_score",
        "score": round(best_score, 4),
        "candidates": sorted(candidates, reverse=True, key=lambda item: item["score"])[:5],
    }
