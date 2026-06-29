#!/usr/bin/env python3
"""Prepare per-key-action X-ICM prompt trajectories from seen demos.

Run this on the CAIR machine inside the X-ICM environment after AGNOSTOS data is
linked. It does not modify the vanilla X-ICM prompt code; it only writes a JSON
payload that can be consumed by render_xicm_geometry_affordance_prompt.py.
Clean v1 treats primitive geometry as retrieval evidence and RoboPoint output
as optional contact hints, not symbolic affordance descriptors.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def _load_retrieval_rows(path: str | None, top_k: int | None = None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("ranked") or payload.get("top_k_demos") or payload.get("selected") or []
    if not isinstance(rows, list):
        raise ValueError(f"retrieval ranking {path} does not contain ranked rows")
    rows = [row if isinstance(row, dict) else {"episode_path": row} for row in rows]
    if top_k is not None and top_k > 0:
        rows = rows[:top_k]
    for row in rows:
        if not row.get("episode_path"):
            raise ValueError(f"retrieval row is missing episode_path: {row}")
    return rows


def _load_descriptor_cache(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    cache_path = Path(path)
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in cache_path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(cache_path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
    for row in rows:
        demo_id = row.get("demo_id")
        demo_dir = row.get("demo_dir")
        task = row.get("task")
        episode_id = row.get("episode_id")
        if demo_id and episode_id is None:
            match = re.search(r"_(\d+)$", str(demo_id))
            if match:
                episode_id = int(match.group(1))
        keys = [
            demo_id,
            demo_dir,
            Path(demo_dir).name if demo_dir else None,
            row.get("episode_path"),
            f"{task}:{episode_id}" if task is not None and episode_id is not None else None,
            f"{task}_episode{episode_id}" if task is not None and episode_id is not None else None,
        ]
        for key in keys:
            if key:
                cache[str(key)] = row
    return cache


def _task_and_episode_from_path(episode_path: Path) -> tuple[str, int]:
    # Expected AGNOSTOS layout: <root>/<task>/all_variations/episodes/episodeN
    task = episode_path.parts[-4]
    episode_name = episode_path.name
    if not episode_name.startswith("episode"):
        raise ValueError(f"cannot infer episode id from {episode_path}")
    return task, int(episode_name.replace("episode", "", 1))


def _lookup_descriptors(cache: dict[str, dict[str, Any]], task: str, episode_id: int, episode_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = [str(episode_path), f"{task}:{episode_id}", f"{task}_episode{episode_id}"]
    row = next((cache[key] for key in keys if key in cache), {})
    return row.get("geometry_g_i") or {}, row.get("contact_hints_i") or row.get("affordance_a_i") or {}


def _import_xicm_helpers(xicm_root: Path):
    sys.path.insert(0, str(xicm_root))
    import form_icl_demonstrations_crosstask_ranking as xicm_form  # pylint: disable=import-error,import-outside-toplevel

    return xicm_form


def _mask_mapping_for_index(xicm_form: Any, episode_path: Path, idx: int, sim_name_to_real_name: dict[str, str]) -> dict[int, str]:
    mask_id_to_sim_name_dict = xicm_form._get_mask_id_to_name_dict(str(episode_path), idx)
    mask_id_to_sim_name = {}
    for camera in xicm_form.CAMERAS:
        mask_id_to_sim_name.update(mask_id_to_sim_name_dict[camera])
    return {
        mask_id: sim_name_to_real_name[name]
        for mask_id, name in mask_id_to_sim_name.items()
        if name in sim_name_to_real_name
    }


def _render_observation_at_keypoint(
    xicm_form: Any,
    episode_path: Path,
    keypoint_idx: int,
    sim_name_to_real_name: dict[str, str],
    task_instruction: str,
) -> str:
    mask_dict = xicm_form._get_mask_dict(str(episode_path), keypoint_idx)
    point_cloud_dict = xicm_form._get_point_cloud_dict(str(episode_path), keypoint_idx)
    mask_id_to_real_name = _mask_mapping_for_index(xicm_form, episode_path, keypoint_idx, sim_name_to_real_name)
    return xicm_form.form_obs(
        mask_dict,
        mask_id_to_real_name,
        point_cloud_dict,
        taskname=task_instruction,
        cross_task_eval=1,
    )


def _demo_to_steps(xicm_form: Any, episode_path: Path, task: str) -> tuple[str, list[dict[str, Any]]]:
    sim_name_to_real_name = xicm_form.seen_sim_name_to_real_name[task]
    with (episode_path / "low_dim_obs.pkl").open("rb") as f:
        demo = pickle.load(f)
    with (episode_path / "variation_descriptions.pkl").open("rb") as f:
        task_instruction = pickle.load(f)[0]

    keypoints = xicm_form._keypoint_discovery(demo)
    steps = []
    for keypoint_idx in keypoints:
        obs_at_key_action = _render_observation_at_keypoint(
            xicm_form,
            episode_path,
            keypoint_idx,
            sim_name_to_real_name,
            task_instruction,
        )
        action = xicm_form._get_action(demo[keypoint_idx], demo[keypoint_idx])
        steps.append(
            {
                "keypoint_index": keypoint_idx,
                "observation": obs_at_key_action,
                "action": action,
            }
        )
    return task_instruction, steps


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    xicm_root = Path(args.xicm_root)
    xicm_form = _import_xicm_helpers(xicm_root)
    descriptor_cache = _load_descriptor_cache(args.descriptor_cache)

    retrieval_rows = _load_retrieval_rows(args.retrieval_ranking, args.top_k)
    selected_episodes = [{"episode_path": episode} for episode in (args.seen_episode or [])]
    selected_episodes.extend(retrieval_rows)
    if not selected_episodes:
        raise ValueError("provide --seen-episode or --retrieval-ranking")

    demos = []
    for rank, item in enumerate(selected_episodes, start=1):
        episode_path = Path(item["episode_path"])
        task, episode_id = _task_and_episode_from_path(episode_path)
        task_instruction, steps = _demo_to_steps(xicm_form, episode_path, task)
        geometry_g_i, contact_hints_i = _lookup_descriptors(descriptor_cache, task, episode_id, episode_path)
        demos.append(
            {
                "rank": rank,
                "task": task,
                "episode_id": episode_id,
                "episode_path": str(episode_path),
                "retrieval_score": item.get("score"),
                "retrieval_components": {
                    "s_dyn": item.get("s_dyn"),
                    "s_geo": item.get("s_geo"),
                },
                "task_instruction": task_instruction,
                "geometry_g_i": geometry_g_i,
                "contact_hints_i": contact_hints_i,
                "steps": steps,
            }
        )

    query = {
        "task_instruction": args.query_task_instruction,
        "observation": args.query_observation,
        "geometry_g_j": _read_json(args.query_geometry),
        "contact_hints_j": _read_json(args.query_affordance),
    }
    return {"retrieved_demos": demos, "query": query}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xicm-root", default=os.environ.get("XICM_ROOT", "/data/yf23/projects/ICRA27-ROBOT/X-ICM"))
    parser.add_argument("--seen-episode", action="append", help="Path to seen task episodeN directory; repeat for top-k demos")
    parser.add_argument("--retrieval-ranking", help="JSON ranking from score_xicm_geometry_affordance_retrieval.py")
    parser.add_argument("--top-k", type=int, help="Limit ranked retrieved demos before prompt preparation")
    parser.add_argument("--descriptor-cache", help="Optional JSON/JSONL cache with geometry_g_i and contact_hints_i")
    parser.add_argument("--query-task-instruction", required=True)
    parser.add_argument("--query-observation", required=True, help="Current unseen X-ICM observation text only")
    parser.add_argument("--query-geometry", help="Optional JSON file for geometry_g_j")
    parser.add_argument("--query-affordance", help="Optional JSON file for contact_hints_j (legacy flag name)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = build_payload(args)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": args.out, "retrieved_demos": len(payload["retrieved_demos"])}, indent=2))


if __name__ == "__main__":
    main()
