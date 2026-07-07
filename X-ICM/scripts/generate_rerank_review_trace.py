#!/usr/bin/env python3
"""Write a step-by-step retrieval review bundle for unseen X-ICM queries."""

import argparse
import csv
import json
import os
import pickle
import random
import shutil
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import form_icl_demonstrations_crosstask_ranking as xicm


ROLE_COLORS = {
    "manipulated_object_contact": (235, 48, 48),
    "goal_region": (40, 190, 90),
    "secondary_object_to_move": (35, 125, 235),
    "constraint_reference": (245, 190, 35),
}


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_text(path, text):
    with open(path, "w") as handle:
        handle.write(text.rstrip() + "\n")


def write_json(path, value):
    with open(path, "w") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def role_color(role):
    return ROLE_COLORS.get(str(role), (180, 60, 220))


def add_overlay_point(points_by_view, view, pixel_xy, role, index, label, chosen):
    if not pixel_xy:
        return
    try:
        x_coord = int(round(float(pixel_xy[0])))
        y_coord = int(round(float(pixel_xy[1])))
    except (TypeError, ValueError, IndexError):
        return
    points_by_view.setdefault(view, []).append(
        {
            "x": x_coord,
            "y": y_coord,
            "role": role,
            "index": index,
            "label": label,
            "chosen": chosen,
        }
    )


def collect_overlay_points(contact_hints):
    points_by_view = {"front": [], "overhead": []}
    for role in contact_hints.get("candidate_contact_coordinates") or []:
        index = role.get("final_index")
        role_name = role.get("role", "contact")
        label = f"{index}:{role_name}"
        chosen_view = role.get("chosen_source_view") or role.get("source_view") or ""
        chosen_front = "front" in chosen_view
        chosen_overhead = "overhead" in chosen_view

        front_role = role.get("front_match") or {}
        overhead_role = role.get("overhead") or {}
        add_overlay_point(
            points_by_view,
            "front",
            role.get("chosen_pixel_xy_front") or front_role.get("pixel_xy"),
            role_name,
            index,
            label,
            chosen_front,
        )
        add_overlay_point(
            points_by_view,
            "overhead",
            role.get("chosen_pixel_xy_overhead") or overhead_role.get("pixel_xy"),
            role_name,
            index,
            label,
            chosen_overhead,
        )
    return points_by_view


def draw_marker(draw, point, scale):
    x_coord = point["x"] * scale
    y_coord = point["y"] * scale
    radius = 6 * scale
    color = role_color(point["role"])
    bounds = [x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius]
    if point["chosen"]:
        draw.ellipse(bounds, fill=color, outline=(0, 0, 0), width=max(2, scale))
    else:
        draw.ellipse(bounds, outline=color, width=max(3, scale))
    draw.line([x_coord - radius - 3, y_coord, x_coord + radius + 3, y_coord], fill=(0, 0, 0), width=max(1, scale))
    draw.line([x_coord, y_coord - radius - 3, x_coord, y_coord + radius + 3], fill=(0, 0, 0), width=max(1, scale))

    text = str(point["index"])
    text_x = min(max(2, x_coord + radius + 2), 512 - 34)
    text_y = min(max(2, y_coord - radius - 2), 512 - 16)
    draw.rectangle([text_x - 1, text_y - 1, text_x + 26, text_y + 14], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((text_x + 2, text_y), text, fill=(0, 0, 0))


def draw_contact_overlays(output_dir, contact_hints):
    points_by_view = collect_overlay_points(contact_hints)
    overlay_paths = {}
    scale = 4
    for view, image_name in [("front", "query_front_rgb_0.png"), ("overhead", "query_overhead_rgb_0.png")]:
        src_path = os.path.join(output_dir, image_name)
        if not os.path.exists(src_path):
            continue
        image = Image.open(src_path).convert("RGB")
        image = image.resize((image.width * scale, image.height * scale))
        draw = ImageDraw.Draw(image)
        for point in points_by_view.get(view, []):
            draw_marker(draw, point, scale)
        out_path = os.path.join(output_dir, f"contact_points_overlay_{view}.png")
        image.save(out_path)
        overlay_paths[view] = out_path

    if "front" in overlay_paths and "overhead" in overlay_paths:
        front = Image.open(overlay_paths["front"]).convert("RGB")
        overhead = Image.open(overlay_paths["overhead"]).convert("RGB")
        combined = Image.new("RGB", (front.width + overhead.width, max(front.height, overhead.height)), (255, 255, 255))
        combined.paste(front, (0, 0))
        combined.paste(overhead, (front.width, 0))
        combined_path = os.path.join(output_dir, "contact_points_overlay_front_overhead.png")
        combined.save(combined_path)
        overlay_paths["front_overhead"] = combined_path
    return overlay_paths


def read_instruction(episode_path):
    with open(os.path.join(episode_path, "variation_descriptions.pkl"), "rb") as handle:
        descriptions = pickle.load(handle)
    return descriptions[0]


def merge_mask_names(mask_id_to_name_by_camera):
    merged = {}
    for camera in xicm.CAMERAS:
        merged.update(mask_id_to_name_by_camera.get(camera, {}))
    return merged


def query_goal_state(task_name, geometry, contact_hints):
    profile = xicm._interaction_profile(task_name, geometry, contact_hints)
    goal_state = xicm._goal_state_descriptor(
        task_name,
        geometry,
        contact_hints,
        profile,
        raw_goal=(
            geometry.get("goal_state_h_j")
            or geometry.get("target_pose_h_j")
            or geometry.get("target_pose_j")
            or geometry.get("target_pose")
            or geometry.get("goal_state_descriptor_j")
        ),
        use_as="query_goal_state",
    )
    return profile, goal_state


def score_broad_pool(similarity, candidates, query_geometry, query_contact, ranking_metric):
    candidate_scores = [candidate["dynamic_score_raw"] for candidate in candidates]
    sim_min = min(candidate_scores) if candidate_scores else 0.0
    sim_max = max(candidate_scores) if candidate_scores else 0.0
    sim_span = sim_max - sim_min

    alpha, beta, _ = xicm._augmented_weights(ranking_metric)
    delta, penalty_weight = xicm._profile_weights(ranking_metric)
    use_v2 = xicm._is_v2_ranking(ranking_metric)
    use_v3 = xicm._is_v3_ranking(ranking_metric)
    use_v4 = xicm._is_v4_ranking(ranking_metric)
    use_plan = xicm._is_plan_ranking(ranking_metric)
    plan_weight = float(os.environ.get("XICM_GA_PLAN_WEIGHT", "0.45")) if use_plan else 0.0

    query_task = query_geometry.get("task_key") or query_geometry.get("manipulated_object") or ""
    query_profile = xicm._interaction_profile(query_task, query_geometry, query_contact)
    ranked = []

    for candidate in candidates:
        task = candidate["task"]
        row = candidate["row"]
        seen_contact = row.get("contact_hints_i") or row.get("affordance_a_i") or {}
        seen_geometry = xicm._canonical_geometry(task, row.get("geometry_g_i") or {}, seen_contact)
        seen_profile = xicm._interaction_profile(task, seen_geometry, seen_contact)
        s_dyn = 0.0 if sim_span == 0 else (candidate["dynamic_score_raw"] - sim_min) / sim_span
        s_geo = xicm._geometry_similarity(seen_geometry, query_geometry)

        if use_v3 or use_v4 or use_plan:
            s_profile = xicm._mechanical_similarity(
                seen_profile,
                query_profile,
                seen_geometry,
                query_geometry,
                seen_contact,
                query_contact,
            )
            penalty = xicm._v3_conflict_penalty(seen_profile, query_profile)
        else:
            s_profile = xicm._profile_similarity(seen_profile, query_profile)
            penalty = xicm._profile_conflict_penalty(seen_profile, query_profile) if use_v2 else 0.0

        plan_compat = (
            xicm._plan_compatibility(seen_profile, query_profile)
            if use_plan
            else {
                "tier": "not_plan_scored",
                "score": 0.0,
                "reason": "",
                "seen_family": xicm._family_name(seen_profile),
                "query_family": xicm._family_name(query_profile),
            }
        )

        score = alpha * s_dyn + beta * s_geo
        if use_v2 or use_v3 or use_v4 or use_plan:
            score += delta * s_profile - penalty_weight * penalty
        if use_plan:
            score += plan_weight * plan_compat["score"]
        raw_score = score
        score_cap = xicm._plan_score_cap(plan_compat["tier"]) if use_plan else None
        if score_cap is not None:
            score = min(score, score_cap)

        ranked.append(
            {
                "score": float(score),
                "raw_score": float(raw_score),
                "score_cap": score_cap,
                "index": candidate["index"],
                "task": task,
                "episode_id": candidate["episode_id"],
                "dynamic_rank": candidate["dynamic_rank"],
                "dynamic_score_raw": candidate["dynamic_score_raw"],
                "s_dyn": float(s_dyn),
                "s_geo": float(s_geo),
                "s_profile": float(s_profile),
                "s_plan": float(plan_compat["score"]),
                "penalty": float(penalty),
                "compatibility_tier": plan_compat["tier"],
                "compatibility_reason": plan_compat["reason"],
                "query_family": plan_compat["query_family"],
                "seen_family": plan_compat["seen_family"],
                "seen_profile": seen_profile,
                "seen_geometry": seen_geometry,
            }
        )

    ranked.sort(reverse=True, key=lambda item: item["score"])
    return xicm._attention_bias_for_ranked_items(ranked)


def row_for_item(item, fine_rank=None):
    return {
        "fine_rank": fine_rank,
        "dynamic_rank": item.get("dynamic_rank"),
        "task": item.get("task"),
        "episode_id": item.get("episode_id"),
        "score": item.get("score"),
        "raw_score": item.get("raw_score"),
        "dynamic_score_raw": item.get("dynamic_score_raw"),
        "s_dyn": item.get("s_dyn"),
        "s_geo": item.get("s_geo"),
        "s_profile": item.get("s_profile"),
        "s_plan": item.get("s_plan"),
        "penalty": item.get("penalty"),
        "attention_bias": item.get("attention_bias"),
        "seen_family": item.get("seen_family"),
        "query_family": item.get("query_family"),
        "compatibility_tier": item.get("compatibility_tier"),
        "compatibility_reason": item.get("compatibility_reason"),
    }


def sample_tasks(task_root, count, seed):
    tasks = [
        name
        for name in sorted(os.listdir(task_root))
        if os.path.isdir(os.path.join(task_root, name, "all_variations", "episodes", "episode0"))
        and name in xicm.unseen_task_name_to_handler
    ]
    return random.Random(seed).sample(tasks, min(count, len(tasks)))


def build_review_for_task(task_name, episode_id, output_dir, ranking_metric, top_k):
    episode_path = os.path.join(
        xicm.unseen_path,
        task_name,
        "all_variations",
        "episodes",
        f"episode{episode_id}",
    )
    instruction = read_instruction(episode_path)
    handler = xicm.create_task_handler(task_name)

    mask_dict = xicm._get_mask_dict(episode_path, 0)
    mask_names_by_camera = xicm._get_mask_id_to_name_dict(episode_path, 0)
    mask_id_to_sim_name = merge_mask_names(mask_names_by_camera)
    point_cloud_dict = xicm._get_point_cloud_dict(episode_path, 0)
    mask_id_to_real_name = {
        mask_id: handler.sim_name_to_real_name[name]
        for mask_id, name in mask_id_to_sim_name.items()
        if name in handler.sim_name_to_real_name
    }
    query_observation = xicm.form_obs(
        mask_dict,
        mask_id_to_real_name,
        point_cloud_dict,
        taskname=instruction,
        cross_task_eval=1,
    )
    query_geometry, query_contact = xicm._query_descriptors(
        task_name,
        instruction,
        mask_dict=mask_dict,
        mask_id_to_sim_name=mask_id_to_sim_name,
        point_cloud_dict=point_cloud_dict,
    )
    query_geometry = dict(query_geometry)
    query_geometry["task_key"] = task_name
    query_profile, goal_state = query_goal_state(task_name, query_geometry, query_contact)

    front_rgb = os.path.join(episode_path, "front_rgb", "0.png")
    overhead_rgb = os.path.join(episode_path, "overhead_rgb", "0.png")
    shutil.copy2(front_rgb, os.path.join(output_dir, "query_front_rgb_0.png"))
    shutil.copy2(overhead_rgb, os.path.join(output_dir, "query_overhead_rgb_0.png"))
    overlay_paths = draw_contact_overlays(output_dir, query_contact)

    all_demo_paths = xicm.all_diffusion_features["all_demo_paths"]
    all_output_image_feats = xicm.all_diffusion_features["all_output_image_feats"]
    all_prompt_feats = xicm.all_diffusion_features["all_prompt_feats"]
    _, query_output_image_feat, query_prompt_feat = xicm.extract_diffusion_features(front_rgb, instruction)
    query_feat = np.concatenate([query_prompt_feat, query_output_image_feat])
    memory_feat = np.concatenate([all_prompt_feats, all_output_image_feats], axis=1)
    similarity = np.dot(memory_feat, query_feat)

    review_cache = xicm._load_augmented_review_cache()
    requested = xicm._rerank_candidate_count(len(all_demo_paths), top_k)
    broad_candidates = xicm._dynamic_shortlist_candidates(
        similarity,
        all_demo_paths,
        review_cache,
        requested,
    )
    broad_rows = [
        {
            "dynamic_rank": item["dynamic_rank"],
            "task": item["task"],
            "episode_id": item["episode_id"],
            "dynamic_score_raw": item["dynamic_score_raw"],
            "demo_path": all_demo_paths[item["index"]],
        }
        for item in broad_candidates
    ]

    fine_pool = score_broad_pool(
        similarity,
        broad_candidates,
        query_geometry,
        query_contact,
        ranking_metric,
    )
    final_ranked = xicm._rank_augmented_indices(
        similarity,
        all_demo_paths,
        query_geometry,
        query_contact,
        ranking_metric,
        top_k,
    )
    final_prompt = xicm._format_augmented_user_prompt(
        final_ranked,
        all_demo_paths,
        query_observation,
        task_name,
        instruction,
        query_geometry,
        query_contact,
        include_geometry=xicm._include_geometry(ranking_metric),
        include_affordance=xicm._include_affordance(ranking_metric),
        use_v2=(
            xicm._is_v2_ranking(ranking_metric)
            or xicm._is_v3_ranking(ranking_metric)
            or xicm._is_v4_ranking(ranking_metric)
            or xicm._is_plan_ranking(ranking_metric)
        ),
        use_v3=xicm._is_v3_ranking(ranking_metric),
        use_v4=xicm._is_v4_ranking(ranking_metric),
        use_plan=xicm._is_plan_ranking(ranking_metric),
    )

    write_json(
        os.path.join(output_dir, "00_query_metadata.json"),
        {
            "task": task_name,
            "episode_id": episode_id,
            "instruction": instruction,
            "ranking_metric": ranking_metric,
            "top_k": top_k,
            "rerank_pool_requested": requested,
            "front_rgb_source": front_rgb,
            "overhead_rgb_source": overhead_rgb,
            "contact_overlay_paths": overlay_paths,
        },
    )
    write_json(
        os.path.join(output_dir, "01_extracted_query.json"),
        {
            "current_observation": query_observation,
            "geometry_g_j": query_geometry,
            "goal_state_h_j": goal_state,
            "contact_hints_c_j": query_contact,
            "interaction_profile_p_j": query_profile,
            "mask_id_to_sim_name": mask_id_to_sim_name,
            "mask_id_to_real_name": mask_id_to_real_name,
        },
    )
    write_text(
        os.path.join(output_dir, "01_extracted_query.md"),
        "\n\n".join(
            [
                f"# Extraction: {task_name} episode{episode_id}",
                "## Current observation",
                query_observation,
                "## Query descriptors",
                xicm._format_compact_query_descriptor(query_geometry, goal_state),
                "## Query contact hints",
                xicm._format_compact_query_contact_hints(query_contact),
                "## Interaction profile",
                xicm._format_profile_block("Precise interaction signature p_j", query_profile),
            ]
        ),
    )
    write_csv(
        os.path.join(output_dir, "02_broad_dynamic_top50.csv"),
        broad_rows,
        ["dynamic_rank", "task", "episode_id", "dynamic_score_raw", "demo_path"],
    )
    write_text(
        os.path.join(output_dir, "02_broad_dynamic_top50.md"),
        "\n".join(
            [
                "# Broad retrieval: dynamic diffusion only",
                f"Requested pool size: {requested}",
                "",
                *[
                    f"{row['dynamic_rank']}. {row['task']} episode{row['episode_id']} "
                    f"score={row['dynamic_score_raw']:.6f}"
                    for row in broad_rows
                ],
            ]
        ),
    )
    fine_rows = [row_for_item(item, rank) for rank, item in enumerate(fine_pool, start=1)]
    write_csv(
        os.path.join(output_dir, "03_fine_rerank_scored_pool_top50.csv"),
        fine_rows,
        [
            "fine_rank",
            "dynamic_rank",
            "task",
            "episode_id",
            "score",
            "raw_score",
            "dynamic_score_raw",
            "s_dyn",
            "s_geo",
            "s_profile",
            "s_plan",
            "penalty",
            "attention_bias",
            "seen_family",
            "query_family",
            "compatibility_tier",
            "compatibility_reason",
        ],
    )
    write_json(
        os.path.join(output_dir, "03_fine_rerank_scored_pool_top50.json"),
        fine_pool,
    )
    final_rows = [row_for_item(item, rank) for rank, item in enumerate(final_ranked, start=1)]
    write_csv(
        os.path.join(output_dir, "04_final_selected_top5.csv"),
        final_rows,
        [
            "fine_rank",
            "dynamic_rank",
            "task",
            "episode_id",
            "score",
            "raw_score",
            "dynamic_score_raw",
            "s_dyn",
            "s_geo",
            "s_profile",
            "s_plan",
            "penalty",
            "attention_bias",
            "seen_family",
            "query_family",
            "compatibility_tier",
            "compatibility_reason",
        ],
    )
    write_json(
        os.path.join(output_dir, "04_final_selected_top5.json"),
        final_ranked,
    )
    write_text(os.path.join(output_dir, "05_final_prompt.txt"), final_prompt)
    write_json(
        os.path.join(output_dir, "05_final_messages_qwenvl_preview.json"),
        {
            "system_prompt_note": "The actual QwenVL agent prepends its augmented system prompt and attaches front/overhead images before this user prompt.",
            "query_images": {
                "front_rgb": os.path.join(output_dir, "query_front_rgb_0.png"),
                "overhead_rgb": os.path.join(output_dir, "query_overhead_rgb_0.png"),
            },
            "user_prompt": final_prompt,
        },
    )
    write_text(
        os.path.join(output_dir, "README.md"),
        "\n".join(
            [
                f"# {task_name} episode{episode_id}",
                "",
                f"Instruction: {instruction}",
                "",
                "Files:",
                "- `query_front_rgb_0.png` and `query_overhead_rgb_0.png`: current unseen images.",
                "- `contact_points_overlay_front.png`, `contact_points_overlay_overhead.png`, and `contact_points_overlay_front_overhead.png`: c_j points drawn on RGB views.",
                "- `01_extracted_query.md/json`: current observation, g_j, h_j, c_j, and p_j.",
                "- `02_broad_dynamic_top50.csv/md`: broad stage, dynamic diffusion only.",
                "- `03_fine_rerank_scored_pool_top50.csv/json`: fine stage scores for the 50 broad candidates.",
                "- `04_final_selected_top5.csv/json`: final k=5 retrieved demos used in the prompt.",
                "- `05_final_prompt.txt`: final user prompt text.",
                "",
                "Final selected demos:",
                *[
                    f"{row['fine_rank']}. {row['task']} episode{row['episode_id']} "
                    f"(dynamic_rank={row['dynamic_rank']}, score={float(row['score']):.4f}, "
                    f"S_dyn={float(row['s_dyn']):.3f}, S_geo={float(row['s_geo']):.3f}, "
                    f"S_profile={float(row['s_profile']):.3f}, penalty={float(row['penalty']):.3f})"
                    for row in final_rows
                ],
            ]
        ),
    )
    return {
        "task": task_name,
        "episode_id": episode_id,
        "instruction": instruction,
        "output_dir": output_dir,
        "final_selected": final_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/data/yf23/projects/ICRA27-ROBOT/review")
    parser.add_argument("--name", default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-candidates", default="50")
    parser.add_argument("--ranking-metric", default="lang_vis.out.geo.aff_v3.rerank_top50")
    args = parser.parse_args()

    os.environ["XICM_GA_RERANK_CANDIDATES"] = str(args.rerank_candidates)
    run_name = args.name or datetime.now().strftime("rerank_k5_pipeline_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tasks = sample_tasks(xicm.unseen_path, args.count, args.seed)
    summaries = []
    for index, task_name in enumerate(tasks, start=1):
        task_dir = os.path.join(run_dir, f"{index:02d}_{task_name}_episode{args.episode_id}")
        os.makedirs(task_dir, exist_ok=True)
        print(f"[{index}/{len(tasks)}] reviewing {task_name} episode{args.episode_id}", flush=True)
        summaries.append(
            build_review_for_task(
                task_name,
                args.episode_id,
                task_dir,
                args.ranking_metric,
                args.top_k,
            )
        )

    write_json(
        os.path.join(run_dir, "selected_examples.json"),
        {
            "ranking_metric": args.ranking_metric,
            "top_k": args.top_k,
            "rerank_candidates": args.rerank_candidates,
            "seed": args.seed,
            "examples": summaries,
        },
    )
    write_csv(
        os.path.join(run_dir, "final_selected_summary.csv"),
        [
            {
                "query_task": summary["task"],
                "query_episode_id": summary["episode_id"],
                "selected_rank": row["fine_rank"],
                "selected_task": row["task"],
                "selected_episode_id": row["episode_id"],
                "dynamic_rank": row["dynamic_rank"],
                "score": row["score"],
                "s_dyn": row["s_dyn"],
                "s_geo": row["s_geo"],
                "s_profile": row["s_profile"],
                "penalty": row["penalty"],
                "seen_family": row["seen_family"],
                "query_family": row["query_family"],
            }
            for summary in summaries
            for row in summary["final_selected"]
        ],
        [
            "query_task",
            "query_episode_id",
            "selected_rank",
            "selected_task",
            "selected_episode_id",
            "dynamic_rank",
            "score",
            "s_dyn",
            "s_geo",
            "s_profile",
            "penalty",
            "seen_family",
            "query_family",
        ],
    )
    write_text(
        os.path.join(run_dir, "README.md"),
        "\n".join(
            [
                "# Rerank k=5 pipeline review",
                "",
                f"Ranking metric: `{args.ranking_metric}`",
                f"Top-k final demos: `{args.top_k}`",
                f"Broad dynamic shortlist size: `{args.rerank_candidates}`",
                f"Random seed: `{args.seed}`",
                "",
                "Per-example folders:",
                *[f"- `{os.path.basename(summary['output_dir'])}`: {summary['instruction']}" for summary in summaries],
                "",
                "Read each folder in order: extraction, broad dynamic retrieval, fine rerank, final selected top-5, final prompt.",
                "The contact overlay PNGs in each folder show the query c_j role points on front/overhead RGB views.",
            ]
        ),
    )
    print(run_dir)


if __name__ == "__main__":
    main()
