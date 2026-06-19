import copy
import glob
import itertools
import math
import os
import pickle
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
from PIL import Image
from pyrep.objects import VisionSensor
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils import _image_to_float_array, normalize_quaternion, point_to_voxel_index, quaternion_to_discrete_euler, CAMERAS
from utils import encode_img
import json
from main import PROJECT_ROOT

from rlbench_inference_dynamics_diffusion import *

seen_sim_name_to_real_name={}

demo_num_per_icl=1
demo_nums = 200


seen_path=os.path.join(PROJECT_ROOT, "data/seen_tasks/train")
unseen_path = os.path.join(PROJECT_ROOT, "data/unseen_tasks/test") 
f=open(os.path.join(PROJECT_ROOT, "data/dynamics_diffusion/all_diffusion_features.pkl"), 'rb')
all_diffusion_features=pickle.load(f)
f.close()

AUGMENTED_REVIEW_CACHE = None
AUGMENTED_DEMO_STEPS_CACHE = {}

GEOMETRY_FIELDS = [
    "manipulated_object",
    "key_features",
    "primary_shape",
    "part_geometry",
    "opening_geometry",
    "axis_geometry",
    "clearance_geometry",
    "task_relevant_geometric_cues",
]

AFFORDANCE_FIELDS = [
    "grasp_affordance",
    "contact_affordance",
    "motion_affordance",
    "containment_affordance",
    "articulation_affordance",
    "required_contact_region",
    "preferred_contact_points",
    "precision_requirement",
    "failure_sensitive_property",
]

UNSEEN_DESCRIPTOR_RULES = {
    "put_toilet_roll_on_stand": {
        "geometry": {"manipulated_object": "toilet_roll_and_stand", "key_features": ["toilet_roll", "cylindrical", "hole", "stand", "vertical_peg", "alignment_sensitive"], "part_geometry": ["roll", "central_hole", "stand"], "opening_geometry": "hole", "axis_geometry": "vertical", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["central_hole", "stand_peg"]},
        "affordance": {"grasp_affordance": "body_grasp", "contact_affordance": "insert_object", "motion_affordance": "insert", "containment_affordance": "hole", "articulation_affordance": "none", "required_contact_region": "roll_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "misalignment"},
    },
    "put_knife_on_chopping_board": {
        "geometry": {"manipulated_object": "knife_and_chopping_board", "key_features": ["knife", "elongated", "flat", "handle", "chopping_board", "flat_target_region"], "part_geometry": ["knife_handle", "blade", "board"], "opening_geometry": "none", "axis_geometry": "free_motion", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["flat_board", "elongated_knife"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "place", "containment_affordance": "none", "articulation_affordance": "none", "required_contact_region": "knife_handle", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_target_region"},
    },
    "close_fridge": {
        "geometry": {"manipulated_object": "fridge_door", "key_features": ["fridge", "rectangular", "front_door", "handle", "hinge", "vertical_axis"], "part_geometry": ["door", "handle", "hinge"], "opening_geometry": "front_opening", "axis_geometry": "vertical", "clearance_geometry": "swing_path", "task_relevant_geometric_cues": ["door_panel", "hinge_axis"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "push_surface", "motion_affordance": "push", "containment_affordance": "closed_container", "articulation_affordance": "hinge_open_close", "required_contact_region": "door_or_handle", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_axis"},
    },
    "close_microwave": {
        "geometry": {"manipulated_object": "microwave_door", "key_features": ["microwave", "rectangular", "front_door", "handle", "hinge", "vertical_axis"], "part_geometry": ["door", "handle", "hinge"], "opening_geometry": "front_opening", "axis_geometry": "vertical", "clearance_geometry": "swing_path", "task_relevant_geometric_cues": ["door_panel", "hinge_axis"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "push_surface", "motion_affordance": "push", "containment_affordance": "closed_container", "articulation_affordance": "hinge_open_close", "required_contact_region": "door_or_handle", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_axis"},
    },
    "close_laptop_lid": {
        "geometry": {"manipulated_object": "laptop_lid", "key_features": ["laptop", "flat_panel", "screen", "base", "hinge", "horizontal_axis"], "part_geometry": ["screen_panel", "base", "hinge"], "opening_geometry": "none", "axis_geometry": "horizontal", "clearance_geometry": "swing_path", "task_relevant_geometric_cues": ["open_angle", "hinge_axis"]},
        "affordance": {"grasp_affordance": "edge_grasp", "contact_affordance": "push_surface", "motion_affordance": "rotate", "containment_affordance": "none", "articulation_affordance": "hinge_open_close", "required_contact_region": "lid_edge_or_panel", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_axis"},
    },
    "phone_on_base": {
        "geometry": {"manipulated_object": "phone_and_base", "key_features": ["phone", "rectangular", "flat", "base", "dock", "alignment_sensitive"], "part_geometry": ["phone_body", "base"], "opening_geometry": "slot", "axis_geometry": "free_motion", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["base_slot", "phone_alignment"]},
        "affordance": {"grasp_affordance": "body_grasp", "contact_affordance": "insert_object", "motion_affordance": "place", "containment_affordance": "slot", "articulation_affordance": "none", "required_contact_region": "phone_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "misalignment"},
    },
    "toilet_seat_down": {
        "geometry": {"manipulated_object": "toilet_seat", "key_features": ["seat", "ring", "flat_panel", "hinge", "horizontal_axis", "rotation"], "part_geometry": ["seat", "hinge"], "opening_geometry": "hole", "axis_geometry": "horizontal", "clearance_geometry": "swing_path", "task_relevant_geometric_cues": ["hinge_axis", "seat_ring"]},
        "affordance": {"grasp_affordance": "rim_grasp", "contact_affordance": "push_surface", "motion_affordance": "rotate", "containment_affordance": "none", "articulation_affordance": "hinge_open_close", "required_contact_region": "seat_rim", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_axis"},
    },
    "lamp_off": {
        "geometry": {"manipulated_object": "lamp_switch", "key_features": ["lamp", "switch", "button", "small", "raised_surface"], "part_geometry": ["switch", "lamp_body"], "opening_geometry": "none", "axis_geometry": "none", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["switch_location"]},
        "affordance": {"grasp_affordance": "none", "contact_affordance": "push_surface", "motion_affordance": "push", "containment_affordance": "none", "articulation_affordance": "none", "required_contact_region": "switch_or_button", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "wrong_button"},
    },
    "lamp_on": {
        "geometry": {"manipulated_object": "lamp_switch", "key_features": ["lamp", "switch", "button", "small", "raised_surface"], "part_geometry": ["switch", "lamp_body"], "opening_geometry": "none", "axis_geometry": "none", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["switch_location"]},
        "affordance": {"grasp_affordance": "none", "contact_affordance": "push_surface", "motion_affordance": "push", "containment_affordance": "none", "articulation_affordance": "none", "required_contact_region": "switch_or_button", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "wrong_button"},
    },
    "put_books_on_bookshelf": {
        "geometry": {"manipulated_object": "books_and_bookshelf", "key_features": ["books", "rectangular", "bookshelf", "shelf", "open_container", "slot"], "part_geometry": ["books", "shelf"], "opening_geometry": "front_opening", "axis_geometry": "free_motion", "clearance_geometry": "narrow_path", "task_relevant_geometric_cues": ["shelf_opening", "book_orientation"]},
        "affordance": {"grasp_affordance": "body_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "insert", "containment_affordance": "receptacle", "articulation_affordance": "none", "required_contact_region": "book_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "collision_with_shelf"},
    },
    "put_umbrella_in_umbrella_stand": {
        "geometry": {"manipulated_object": "umbrella_and_stand", "key_features": ["umbrella", "elongated", "stand", "open_container", "vertical_insertion"], "part_geometry": ["umbrella_handle", "stand_opening"], "opening_geometry": "top_opening", "axis_geometry": "vertical", "clearance_geometry": "narrow_path", "task_relevant_geometric_cues": ["stand_opening", "umbrella_axis"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "insert_object", "motion_affordance": "insert", "containment_affordance": "receptacle", "articulation_affordance": "none", "required_contact_region": "umbrella_handle_or_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "misalignment"},
    },
    "open_grill": {
        "geometry": {"manipulated_object": "grill_lid", "key_features": ["grill", "lid", "handle", "hinge", "rotation_axis"], "part_geometry": ["lid", "handle", "hinge"], "opening_geometry": "top_opening", "axis_geometry": "horizontal", "clearance_geometry": "swing_path", "task_relevant_geometric_cues": ["lid_handle", "hinge_axis"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "pull_handle", "motion_affordance": "pull", "containment_affordance": "open_container", "articulation_affordance": "hinge_open_close", "required_contact_region": "handle", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_grasp"},
    },
    "put_rubbish_in_bin": {
        "geometry": {"manipulated_object": "rubbish_and_bin", "key_features": ["rubbish", "bin", "open_container", "top_opening", "receptacle"], "part_geometry": ["rubbish", "bin"], "opening_geometry": "top_opening", "axis_geometry": "free_motion", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["bin_opening", "object_body"]},
        "affordance": {"grasp_affordance": "body_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "place", "containment_affordance": "receptacle", "articulation_affordance": "none", "required_contact_region": "object_body", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "missed_receptacle"},
    },
    "take_usb_out_of_computer": {
        "geometry": {"manipulated_object": "usb_stick", "key_features": ["usb", "rectangular", "port", "slot", "inserted", "alignment_sensitive"], "part_geometry": ["usb_body", "computer_port"], "opening_geometry": "slot", "axis_geometry": "linear", "clearance_geometry": "narrow_path", "task_relevant_geometric_cues": ["port_axis", "usb_body"]},
        "affordance": {"grasp_affordance": "pinch", "contact_affordance": "pull_handle", "motion_affordance": "pull", "containment_affordance": "slot", "articulation_affordance": "none", "required_contact_region": "usb_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "misalignment"},
    },
    "take_lid_off_saucepan": {
        "geometry": {"manipulated_object": "saucepan_lid", "key_features": ["lid", "round", "knob", "saucepan", "top_opening"], "part_geometry": ["lid", "knob", "pan"], "opening_geometry": "top_opening", "axis_geometry": "vertical", "clearance_geometry": "requires_lift", "task_relevant_geometric_cues": ["lid_knob", "pan_rim"]},
        "affordance": {"grasp_affordance": "knob_grasp", "contact_affordance": "lift_top", "motion_affordance": "lift", "containment_affordance": "closed_container", "articulation_affordance": "lid_remove", "required_contact_region": "lid_knob", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_grasp"},
    },
    "take_plate_off_colored_dish_rack": {
        "geometry": {"manipulated_object": "plate_and_dish_rack", "key_features": ["plate", "round", "thin", "rim", "rack", "slot"], "part_geometry": ["plate", "rack_slot"], "opening_geometry": "slot", "axis_geometry": "free_motion", "clearance_geometry": "narrow_path", "task_relevant_geometric_cues": ["plate_rim", "rack_slot"]},
        "affordance": {"grasp_affordance": "rim_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "lift", "containment_affordance": "slot", "articulation_affordance": "none", "required_contact_region": "plate_rim", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "collision_with_rack"},
    },
    "basketball_in_hoop": {
        "geometry": {"manipulated_object": "basketball_and_hoop", "key_features": ["ball", "round", "hoop", "ring", "target_region", "top_opening"], "part_geometry": ["ball", "hoop"], "opening_geometry": "top_opening", "axis_geometry": "vertical", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["hoop_ring", "ball_center"]},
        "affordance": {"grasp_affordance": "body_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "place", "containment_affordance": "receptacle", "articulation_affordance": "none", "required_contact_region": "ball_body", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "missed_receptacle"},
    },
    "scoop_with_spatula": {
        "geometry": {"manipulated_object": "spatula", "key_features": ["spatula", "elongated_tool", "flat_blade", "thin_edge", "scoop"], "part_geometry": ["handle", "flat_blade"], "opening_geometry": "none", "axis_geometry": "free_motion", "clearance_geometry": "requires_slide_under", "task_relevant_geometric_cues": ["blade_edge", "object_under_blade"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "push_surface", "motion_affordance": "scoop", "containment_affordance": "none", "articulation_affordance": "none", "required_contact_region": "spatula_handle", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "lost_contact"},
    },
    "straighten_rope": {
        "geometry": {"manipulated_object": "rope", "key_features": ["rope", "flexible", "elongated", "deformable", "endpoints"], "part_geometry": ["rope_body", "endpoints"], "opening_geometry": "none", "axis_geometry": "free_motion", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["rope_curve", "endpoints"]},
        "affordance": {"grasp_affordance": "pinch", "contact_affordance": "drag_object", "motion_affordance": "drag", "containment_affordance": "none", "articulation_affordance": "flexible_deform", "required_contact_region": "rope_endpoint_or_body", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "lost_contact"},
    },
    "turn_oven_on": {
        "geometry": {"manipulated_object": "oven_knob", "key_features": ["oven", "knob", "round", "rotational_axis", "front_face"], "part_geometry": ["knob"], "opening_geometry": "none", "axis_geometry": "front_normal", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["knob", "rotation_axis"]},
        "affordance": {"grasp_affordance": "knob_grasp", "contact_affordance": "rotate_part", "motion_affordance": "rotate", "containment_affordance": "none", "articulation_affordance": "screw_twist", "required_contact_region": "oven_knob", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_axis"},
    },
    "beat_the_buzz": {
        "geometry": {"manipulated_object": "buzzer", "key_features": ["buzzer", "button", "round", "small", "raised_surface"], "part_geometry": ["button"], "opening_geometry": "none", "axis_geometry": "none", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["button_top"]},
        "affordance": {"grasp_affordance": "none", "contact_affordance": "push_surface", "motion_affordance": "push", "containment_affordance": "none", "articulation_affordance": "none", "required_contact_region": "button_top", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "wrong_button"},
    },
    "water_plants": {
        "geometry": {"manipulated_object": "watering_can_and_plant", "key_features": ["watering_can", "handle", "spout", "plant", "pour", "target_region"], "part_geometry": ["handle", "spout", "plant"], "opening_geometry": "spout", "axis_geometry": "tilt_axis", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["spout_direction", "plant_target"]},
        "affordance": {"grasp_affordance": "handle_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "pour", "containment_affordance": "receptacle", "articulation_affordance": "none", "required_contact_region": "watering_can_handle", "preferred_contact_points": [], "precision_requirement": "medium", "failure_sensitive_property": "wrong_target_region"},
    },
    "unplug_charger": {
        "geometry": {"manipulated_object": "charger_plug", "key_features": ["charger", "plug", "socket", "slot", "inserted", "alignment_sensitive"], "part_geometry": ["plug_body", "socket"], "opening_geometry": "slot", "axis_geometry": "linear", "clearance_geometry": "narrow_path", "task_relevant_geometric_cues": ["plug_axis", "socket"]},
        "affordance": {"grasp_affordance": "pinch", "contact_affordance": "pull_handle", "motion_affordance": "pull", "containment_affordance": "slot", "articulation_affordance": "none", "required_contact_region": "plug_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "misalignment"},
    },
}


def _is_augmented_ranking(ranking_metric):
    return "geo_aff" in ranking_metric or ".geo" in ranking_metric or ".aff" in ranking_metric


def _include_geometry(ranking_metric):
    return "geo_aff" in ranking_metric or ".geo" in ranking_metric


def _include_affordance(ranking_metric):
    return "geo_aff" in ranking_metric or ".aff" in ranking_metric


def _augmented_weights(ranking_metric):
    if all(os.environ.get(name) is not None for name in ["XICM_GA_ALPHA", "XICM_GA_BETA", "XICM_GA_GAMMA"]):
        return float(os.environ["XICM_GA_ALPHA"]), float(os.environ["XICM_GA_BETA"]), float(os.environ["XICM_GA_GAMMA"])
    if "geo_aff" in ranking_metric:
        return 0.65, 0.30, 0.05
    if ".geo" in ranking_metric:
        return 0.65, 0.35, 0.0
    if ".aff" in ranking_metric:
        return 0.65, 0.0, 0.35
    return 1.0, 0.0, 0.0


def _task_episode_from_path(path):
    task = path.split("/")[-4]
    episode = int(path.split("/")[-1].replace("episode", "", 1))
    return task, episode


def _load_augmented_review_cache():
    global AUGMENTED_REVIEW_CACHE
    if AUGMENTED_REVIEW_CACHE is not None:
        return AUGMENTED_REVIEW_CACHE
    cache_path = os.environ.get(
        "XICM_GA_REVIEW_BUNDLE",
        "/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_full_cache/review_bundle.jsonl",
    )
    cache = {}
    with open(cache_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[(row["task"], int(row["episode_id"]))] = row
    AUGMENTED_REVIEW_CACHE = cache
    return cache


def _normalize_token(text):
    compact = str(text).strip().lower().replace(" ", "_")
    parts = [part for part in re.split(r"[^a-z0-9]+", compact) if part]
    if compact and compact != "unknown":
        parts.append(compact)
    return parts


def _flatten_descriptor_values(value):
    if value is None:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten_descriptor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (int, float)):
                continue
            yield from _flatten_descriptor_values(item)
    elif isinstance(value, str):
        yield from _normalize_token(value)
    elif not isinstance(value, (int, float, bool)):
        yield from _normalize_token(str(value))


def _descriptor_tokens(descriptor, fields):
    tokens = set()
    for field in fields:
        tokens.update(_flatten_descriptor_values(descriptor.get(field)))
    return {token for token in tokens if token not in {"none", "unknown", "null"}}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _points(value):
    if not isinstance(value, list):
        return []
    out = []
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


def _point_similarity(a, b):
    if not a or not b:
        return 0.0
    distances = []
    for ax, ay in a:
        distances.append(min(math.dist((ax, ay), (bx, by)) for bx, by in b))
    return max(0.0, 1.0 - (sum(distances) / len(distances)) / math.sqrt(2.0))


def _geometry_similarity(seen_geometry, query_geometry):
    return _jaccard(
        _descriptor_tokens(seen_geometry, GEOMETRY_FIELDS),
        _descriptor_tokens(query_geometry, GEOMETRY_FIELDS),
    )


def _affordance_similarity(seen_affordance, query_affordance):
    label_score = _jaccard(
        _descriptor_tokens(seen_affordance, AFFORDANCE_FIELDS),
        _descriptor_tokens(query_affordance, AFFORDANCE_FIELDS),
    )
    seen_points = _points(seen_affordance.get("preferred_contact_points"))
    query_points = _points(query_affordance.get("preferred_contact_points"))
    if seen_points and query_points:
        return 0.8 * label_score + 0.2 * _point_similarity(seen_points, query_points)
    return label_score


def _query_descriptors(task_key, language_goal):
    if task_key in UNSEEN_DESCRIPTOR_RULES:
        item = UNSEEN_DESCRIPTOR_RULES[task_key]
        return item["geometry"], item["affordance"]
    text = language_goal.lower().replace(" ", "_")
    geometry = {
        "manipulated_object": task_key,
        "key_features": _normalize_token(text),
        "part_geometry": [],
        "opening_geometry": "unknown",
        "axis_geometry": "unknown",
        "clearance_geometry": "unknown",
        "task_relevant_geometric_cues": _normalize_token(text),
    }
    affordance = {
        "grasp_affordance": "unknown",
        "contact_affordance": "unknown",
        "motion_affordance": "unknown",
        "containment_affordance": "none",
        "articulation_affordance": "none",
        "required_contact_region": "unknown",
        "preferred_contact_points": [],
        "precision_requirement": "medium",
        "failure_sensitive_property": "wrong_contact_point",
    }
    return geometry, affordance


def _format_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _format_feature_block(title, values, fields):
    lines = [f"{title}:"]
    for field in fields:
        default = [] if field in {"key_features", "part_geometry", "task_relevant_geometric_cues", "preferred_contact_points"} else "unknown"
        lines.append(f"- {field}: {_format_value(values.get(field, default))}")
    return "\n".join(lines)


def _rank_augmented_indices(similarity, all_demo_paths, query_geometry, query_affordance, ranking_metric, top_k):
    review_cache = _load_augmented_review_cache()
    sim_min = float(np.min(similarity))
    sim_max = float(np.max(similarity))
    sim_span = sim_max - sim_min
    alpha, beta, gamma = _augmented_weights(ranking_metric)
    ranked = []
    for idx, demo_path in enumerate(all_demo_paths):
        task, episode_id = _task_episode_from_path(demo_path)
        row = review_cache.get((task, episode_id))
        if row is None:
            continue
        s_dyn = 0.0 if sim_span == 0 else (float(similarity[idx]) - sim_min) / sim_span
        s_geo = _geometry_similarity(row.get("geometry_g_i") or {}, query_geometry)
        s_aff = _affordance_similarity(row.get("affordance_a_i") or {}, query_affordance)
        score = alpha * s_dyn + beta * s_geo + gamma * s_aff
        ranked.append((score, idx, s_dyn, s_geo, s_aff))
    ranked.sort(reverse=True, key=lambda item: item[0])
    return ranked[:top_k]

class base_task_handler:
    def __init__(self, sim_name_to_real_name):
            self.sim_name_to_real_name = sim_name_to_real_name
            self.save_root = os.path.join(unseen_path, type(self).__name__)
            self.num_demos = demo_num_per_icl
            print(f"Task handler {type(self).__name__} using demonstrations from {self.save_root}")
            random.seed(42)

    def get_user_prompt_ranking(self, mask_dict, mask_id_to_sim_name, point_cloud_dict, custom_num_demos=-1, taskname='None', image_path=None, seed=0, ranking_metric="lang_vis.out"):
        assert os.path.exists(self.save_root), f"Cannot find save root {self.save_root}"
        if custom_num_demos==-1:
            pass
        else:
            self.num_demos=custom_num_demos
        
        mask_id_to_real_name = {mask_id: self.sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                            if name in self.sim_name_to_real_name}
        obs = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict, taskname=taskname, cross_task_eval=1)

        all_demo_paths = all_diffusion_features['all_demo_paths']
        
        if "lang_vis.out" in ranking_metric:
            
            all_input_image_feats = all_diffusion_features['all_input_image_feats']
            all_output_image_feats = all_diffusion_features['all_output_image_feats']
            all_prompt_feats = all_diffusion_features['all_prompt_feats']

            query_input_image_feat, query_output_image_feat, \
                query_prompt_feat = extract_diffusion_features(image_path, taskname)

            query_feat = np.concatenate([query_prompt_feat, query_output_image_feat])  # (2048,)
            memory_feat = np.concatenate([all_prompt_feats, all_output_image_feats], axis=1)  # (M, 2048)

            similarity = np.dot(memory_feat, query_feat)

            if _is_augmented_ranking(ranking_metric):
                query_geometry, query_affordance = _query_descriptors(type(self).__name__, taskname)
                ranked = _rank_augmented_indices(
                    similarity,
                    all_demo_paths,
                    query_geometry,
                    query_affordance,
                    ranking_metric,
                    self.num_demos,
                )
                return _format_augmented_user_prompt(
                    ranked,
                    all_demo_paths,
                    obs,
                    type(self).__name__,
                    taskname,
                    query_geometry,
                    query_affordance,
                    include_geometry=_include_geometry(ranking_metric),
                    include_affordance=_include_affordance(ranking_metric),
                )

            top_indices = np.argsort(similarity)[::-1]

            selected_indices = top_indices[:self.num_demos]
        
        elif "random" in ranking_metric:
            selected_indices = random.sample(range(len(all_demo_paths)), self.num_demos)
        else:
            raise ValueError(f"Invalid ranking metric: {ranking_metric}")


        output = ""
        for i, selected_idx in enumerate(selected_indices):
            icl_episode_path=all_demo_paths[selected_idx]
            icl_task_name=icl_episode_path.split('/')[-4]
            icl_episode_id=int(icl_episode_path.split('/')[-1][7:])
            print("the %d-th icl: "%(i+1), ranking_metric, icl_episode_path)
            train_demos = get_stored_demos_crosstask(seen_path, icl_task_name, icl_episode_id, 1, seen_sim_name_to_real_name[icl_task_name])

            for epi in train_demos:
                output += f"{epi[0]}>{epi[1]}, "
        
        return output + obs + ">"



    def save_in_context_demonstrations(self, custom_num_demos=-1):
        if custom_num_demos==-1:
            pass
        else:
            self.num_demos=custom_num_demos

        train_demos = get_stored_demos(unseen_path, type(self).__name__, demo_nums, self.sim_name_to_real_name)

        # iterate over demo_nums demonstrations, each time take self.num_demos demonstrations
        for i, start_idx in enumerate(range(0, len(train_demos) - self.num_demos + 1, self.num_demos)):
            if start_idx + self.num_demos <= len(train_demos):
                output = ""
                for epi in train_demos[start_idx:start_idx+self.num_demos]:
                    output += f"{epi[0]}>{epi[1]}, "

                d = os.path.join(unseen_path, type(self).__name__, f"demonstrations_{self.num_demos}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'{i}.txt'), "w") as f:
                    f.write(output)
    
    def save_in_context_demonstrations_crosstask(self, custom_num_demos=-1):
        if custom_num_demos==-1:
            pass
        else:
            self.num_demos=custom_num_demos
        
        train_tasknames=os.listdir(seen_path)
        
        # iterate over demo_nums demonstrations, each time take self.num_demos demonstrations
        for i, start_idx in enumerate(range(0, demo_nums - self.num_demos + 1, self.num_demos)):
            if start_idx + self.num_demos <= demo_nums:
                output = ""

                for taskname in train_tasknames:
                    train_demos = get_stored_demos_crosstask(seen_path, taskname, start_idx, self.num_demos, seen_sim_name_to_real_name[taskname])

                    for epi in train_demos:
                        output += f"{epi[0]}>{epi[1]}, "

                d = os.path.join(unseen_path, type(self).__name__, f"demonstrations.crosstask_{self.num_demos}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f'{i}.txt'), "w") as f:
                    f.write(output)

    


class close_jar(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "jar_lid0": "lid",
            "jar0": "jar",
        }
        super().__init__(sim_name_to_real_name)


class open_drawer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "drawer_bottom": "drawer",
        }
        super().__init__(sim_name_to_real_name)

class slide_block_to_color_target(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "target1": "target",
            "block": "block"
        }

        super().__init__(sim_name_to_real_name)

class sweep_to_dustpan_of_size(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "dustpan_tall": "dustpan",
            "broom_holder": "broom holder"
        }
        super().__init__(sim_name_to_real_name)

class meat_off_grill(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "chicken_visual": "chicken",
            "grill_visual": "grill"
        }
        super().__init__(sim_name_to_real_name)

class turn_tap(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "tap_left_visual": "left tap",
            "tap_right_visual": "right tap"
        }
        super().__init__(sim_name_to_real_name)

class put_item_in_drawer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "item": "item",
            "drawer_frame": "drawer"
        }
        super().__init__(sim_name_to_real_name)

class stack_blocks(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "stack_blocks_target0": "first block",
            "stack_blocks_target1": "second block",
            "stack_blocks_target2": "third block",
            "stack_blocks_target3": "fourth block",
            "stack_blocks_target_plane": "plane",
        }
        super().__init__(sim_name_to_real_name)

class light_bulb_in(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "bulb0": "blub",
            "lamp_screw": "lamp screw",
        }
        super().__init__(sim_name_to_real_name)

class put_money_in_safe(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "dollar_stack": "money",
            "safe_body": "shelf",
        }
        super().__init__(sim_name_to_real_name)

class place_wine_at_rack_location(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "wine_bottle_visual": "wine",
            "rack_top_visual": "rack",
        }
        # super().__init__(system_prompt, sim_name_to_real_name, num_demos, num_keypoints)
        super().__init__(sim_name_to_real_name)


class put_groceries_in_cupboard(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cupboard": "cupboard",
            "crackers_visual": "cracker",
        }
        super().__init__(sim_name_to_real_name)


class place_shape_in_shape_sorter(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cube": "cube",
            # "shape_sorter": "shape sorter",
            "shape_sorter_visual": "shape sorter",
        }
        super().__init__(sim_name_to_real_name)

class push_buttons(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "target_button_wrap0": "button",
        }
        super().__init__(sim_name_to_real_name)

class insert_onto_square_peg(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "square_ring": "ring",
            "pillar0": "first spok",
            "pillar1": "second spok",
            "pillar2": "third spok"
            }
        super().__init__(sim_name_to_real_name)

class stack_cups(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cup1_visual": "first cup",
            "cup2_visual": "second cup",
            "cup3_visual": "third cup",
        }
        super().__init__(sim_name_to_real_name)

class place_cups(base_task_handler):
    def __init__(self):

        sim_name_to_real_name = {
            "mug_visual1": "first cup",
            "mug_visual0": "second cup",
            "mug_visual2": "third cup",
            "mug_visual3": "forth cup",
            "place_cups_holder_spoke0": "first holder",
            "place_cups_holder_spoke1": "second holder",
            "place_cups_holder_spoke2": "third holder"
        }
        super().__init__(sim_name_to_real_name)



class reach_and_drag(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "stick": "stick",
            "cube": "cube"
        }
        super().__init__(sim_name_to_real_name)



########## zero shot tasks ################
class basketball_in_hoop(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "basket_ball_hoop_visual": "hoop",
            "ball": "ball"
        }
        super().__init__(sim_name_to_real_name)

class scoop_with_spatula(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "Cuboid": "cube",
            "spatula_visual": "spatula"
        }
        super().__init__(sim_name_to_real_name)

class straighten_rope(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "head": "rope head",
            "head_target": "rope head target",
            "head_tail": "rope head tail",
            "tail": "rope tail",
        }
        super().__init__(sim_name_to_real_name)

class turn_oven_on(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "oven_door": "oven door",
            "oven_knob_8": "first oven knob",
            "oven_knob_9": "second oven knob",
        }
        super().__init__(sim_name_to_real_name)

class beat_the_buzz(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "Cuboid": "cube",
            "wand_visual": "pole",
            "wand_visual_sub": "pole head"
        }
        super().__init__(sim_name_to_real_name)

class water_plants(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "waterer_visual": "waterer",
            "plant_visual": "plant",
            "base_visual": "waterer base"
        }
        super().__init__(sim_name_to_real_name)

class unplug_charger(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "charger_visual": "charger",
            "task_wall": "wall",
        }
        super().__init__(sim_name_to_real_name)

class phone_on_base(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "phone_visual": "phone",
            "phone_case_visual": "phone case",
        }
        super().__init__(sim_name_to_real_name)

class toilet_seat_down(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "toilet_seat_up_toilet": "toilet seat up_toilet",
            "toilet_seat_up_seat": "toilet seat up_seat",
            "toilet": "toilet",
        }
        super().__init__(sim_name_to_real_name)

class lamp_off(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "push_button_target": "button",
            "target_button_topPlate": "button topPlate",
            "target_button_wrap": "button wrap",
        }
        super().__init__(sim_name_to_real_name)

class lamp_on(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "push_button_target": "button",
            "target_button_topPlate": "button topPlate",
            "target_button_wrap": "button wrap",
        }
        super().__init__(sim_name_to_real_name)

class put_books_on_bookshelf(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "book0_visual": "first book",
            "book1_visual": "second book",
            "book2_visual": "third book",
            "bookshelf_visual": "bookshelf",
        }
        super().__init__(sim_name_to_real_name)

class put_umbrella_in_umbrella_stand(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "umbrella_visual": "umbrella",
            "stand_visual": "umbrella stand",
        }
        super().__init__(sim_name_to_real_name)

class open_grill(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "grill_visual": "grill",
            "lid_visual": "lid",
            "handle_visual": "handle",
        }
        super().__init__(sim_name_to_real_name)

class put_rubbish_in_bin(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "rubbish_visual": "rubbish",
            "bin_visual": "bin",
        }
        super().__init__(sim_name_to_real_name)

class take_usb_out_of_computer(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "computer_visual": "computer",
            "usb_visual": "usb",
        }
        super().__init__(sim_name_to_real_name)

class take_lid_off_saucepan(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "saucepan_visual": "saucepan",
            "saucepan_lid_visual": "saucepan lid",
        }
        super().__init__(sim_name_to_real_name)

class take_plate_off_colored_dish_rack(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "plate_visual": "plate",
            "dish_rack_pillar0": "first dish rack",
            "dish_rack_pillar1": "second dish rack",
        }
        super().__init__(sim_name_to_real_name)

class close_fridge(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "fridge_base_visual": "fridge base",
            "door_top_visual": "fridge top door",
            "door_bottom_visual": "fridge bottom door",
        }
        super().__init__(sim_name_to_real_name)

class close_microwave(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "microwave_door": "microwave door",
            "microwave_frame_vis": "microwave frame",
        }
        super().__init__(sim_name_to_real_name)

class close_laptop_lid(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "lid_visual": "lid",
            "laptop_holder": "laptop holder",
            "base_visual":"laptop base"
        }
        super().__init__(sim_name_to_real_name)

class put_toilet_roll_on_stand(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "toilet_roll_visual": "toilet roll",
            "holder_visual": "holder",
            "stand_base": "stand_base",
        }
        super().__init__(sim_name_to_real_name)

class put_knife_on_chopping_board(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "knife_visual": "knife",
            "chopping_board_visual": "chopping board",
        }
        super().__init__(sim_name_to_real_name)


seen_task_name_to_handler = {
    "close_jar": close_jar,
    "open_drawer": open_drawer,
    "slide_block_to_color_target": slide_block_to_color_target,
    "sweep_to_dustpan_of_size": sweep_to_dustpan_of_size,
    "meat_off_grill": meat_off_grill,
    "turn_tap": turn_tap,
    "put_item_in_drawer": put_item_in_drawer,
    "stack_blocks": stack_blocks,
    "light_bulb_in": light_bulb_in,
    "put_money_in_safe": put_money_in_safe,
    "place_wine_at_rack_location": place_wine_at_rack_location, 
    "put_groceries_in_cupboard": put_groceries_in_cupboard,
    "place_shape_in_shape_sorter": place_shape_in_shape_sorter,
    "push_buttons": push_buttons,
    "stack_cups": stack_cups,
    "place_cups": place_cups,
    "insert_onto_square_peg":insert_onto_square_peg,
    "reach_and_drag": reach_and_drag,
    }

unseen_task_name_to_handler = {
    "put_toilet_roll_on_stand": put_toilet_roll_on_stand,
    "put_knife_on_chopping_board": put_knife_on_chopping_board,
    "close_fridge": close_fridge,
    "close_microwave": close_microwave,
    "close_laptop_lid":close_laptop_lid,
    "phone_on_base": phone_on_base,
    "toilet_seat_down": toilet_seat_down,
    "lamp_off": lamp_off,
    "lamp_on": lamp_on,
    "put_books_on_bookshelf": put_books_on_bookshelf,
    "put_umbrella_in_umbrella_stand": put_umbrella_in_umbrella_stand,
    "open_grill": open_grill,
    "put_rubbish_in_bin": put_rubbish_in_bin,
    "take_usb_out_of_computer": take_usb_out_of_computer,
    "take_lid_off_saucepan": take_lid_off_saucepan,
    "take_plate_off_colored_dish_rack": take_plate_off_colored_dish_rack,
    "basketball_in_hoop":basketball_in_hoop,
    "scoop_with_spatula":scoop_with_spatula,
    "straighten_rope":straighten_rope,
    "turn_oven_on":turn_oven_on,
    "beat_the_buzz": beat_the_buzz,
    "water_plants": water_plants,
    "unplug_charger": unplug_charger
    }


task_name_to_handler = unseen_task_name_to_handler
def create_task_handler(task_name):
    return task_name_to_handler[task_name]()


train_tasknames=os.listdir(seen_path)
for taskname in train_tasknames:
    handler = seen_task_name_to_handler[taskname]()
    seen_sim_name_to_real_name[taskname]=handler.sim_name_to_real_name
    del handler


# discretize translation, rotation, gripper open
def _get_action(
        obs_tp1,
        obs_tm1):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat)
    trans_indicies = []
    ignore_collisions = int(obs_tm1.ignore_collisions)

    index = point_to_voxel_index(
        obs_tp1.gripper_pose[:3])
    trans_indicies.extend(index.tolist())

    rot_and_grip_indicies = disc_rot.tolist()
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies + rot_and_grip_indicies

def _get_point_cloud_dict(epis_path, idx):
    # This function gets the point cloud using the same operations as PerAct Colab Tutorial
    DEPTH_SCALE = 2**24 - 1
    point_cloud_dict = {}
    for camera_type in CAMERAS:
        with open(os.path.join(epis_path, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        cam_extrinsics = demo[idx].misc[f"{camera_type}_camera_extrinsics"]
        cam_intrinsics = demo[idx].misc[f"{camera_type}_camera_intrinsics"]
        cam_depth = _image_to_float_array(Image.open(os.path.join(epis_path, f"{camera_type}_depth", f"{idx}.png")), DEPTH_SCALE)
        near = demo[idx].misc[f"{camera_type}_camera_near"]
        far = demo[idx].misc[f"{camera_type}_camera_far"]
        cam_depth = (far - near) * cam_depth + near
        point_cloud_dict[camera_type] = VisionSensor.pointcloud_from_depth_and_camera_params(cam_depth, cam_extrinsics, cam_intrinsics) # reconstructed 3D point cloud in world coordinate frame

    return point_cloud_dict

def _get_mask_dict(epis_path, idx):
    mask_dict = {}
    for camera in CAMERAS:
        img = Image.open(os.path.join(epis_path, f"{camera}_mask", f"{idx}.png"))
        rgb_mask = np.array(img, dtype=int)
        mask_dict[camera] = rgb_mask[:, :, 0] + rgb_mask[:, :, 1]*256 + rgb_mask[:, :, 2]*256*256
    return mask_dict

def _get_mask_id_to_name_dict(epis_path, idx):
    with open(os.path.join(epis_path, "low_dim_obs.pkl"), "rb") as f:
        low_dim_obs = pickle.load(f)
    mask_id_to_name_dict = {}
    for camera in CAMERAS:
        mask_id_to_name_dict[camera] = low_dim_obs[idx].misc[f"{camera}_mask_id_to_name"]
    return mask_id_to_name_dict

# add individual data points to replay
def _add_keypoints_to_replay(
        buffer,
        i,
        demo,
        episode_keypoints,
        epis_path_depth,
        epis_path_char,
        sim_name_to_real_name,
        cross_task_eval
    ):
    prev_action = None
    cur_index = i

    mask_dict = _get_mask_dict(epis_path_char, cur_index)

    mask_id_to_sim_name_dict = _get_mask_id_to_name_dict(epis_path_char, cur_index)
    point_cloud_dict = _get_point_cloud_dict(epis_path_depth, cur_index)
    
    mask_id_to_sim_name = {}
    for camera in CAMERAS:
        mask_id_to_sim_name.update(mask_id_to_sim_name_dict[camera])

    mask_id_to_real_name = {mask_id: sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                        if name in sim_name_to_real_name}

    avg_coord = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict, demo.language_descriptions[0], cross_task_eval)

    buffer.append(avg_coord)
    actions = []
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        action = _get_action(
            obs_tp1, obs_tp1)

        actions.append(action)
    
    buffer.append(actions)

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def _keypoint_discovery(demo, delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    #print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints

def get_stored_demos(dataset_root, task_name, amount, sim_name_to_real_name):
    total_num_keypoints = 0
    buffer = []
    task_root = os.path.join(dataset_root, task_name, 'all_variations', 'episodes')

    for epi_id in tqdm(range(amount)):
        epis_path_depth = os.path.join(task_root, f'episode{epi_id}')
        epis_path_char = os.path.join(task_root, f'episode{epi_id}')

        with open(os.path.join(epis_path_depth, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(epis_path_depth, 'variation_number.pkl'), 'rb') as f:
            demo.variation_number = pickle.load(f)

        # language description
        with open(os.path.join(epis_path_depth, 'variation_descriptions.pkl'), 'rb') as f:
            demo.language_descriptions = pickle.load(f)

        episode_keypoints = _keypoint_discovery(demo)

        tmp = []
        _add_keypoints_to_replay(
            tmp, 0, demo, episode_keypoints, epis_path_depth, epis_path_char, sim_name_to_real_name, cross_task_eval=0)
        tmp.append(f"{epis_path_depth}/front_rgb/0.png")
        buffer.append(tmp)

    print("Average number of steps: ", sum([len(each[1]) for each in buffer])/len(buffer))
    return buffer


def get_stored_demos_crosstask(dataset_root, task_name, start_idx, num_demos, sim_name_to_real_name, cross_task_eval=1):
    total_num_keypoints = 0
    buffer = []
    task_root = os.path.join(dataset_root, task_name, 'all_variations', 'episodes')

    for epi_id in tqdm(range(start_idx, start_idx+num_demos)):
        epis_path_depth = os.path.join(task_root, f'episode{epi_id}')
        epis_path_char = os.path.join(task_root, f'episode{epi_id}')

        with open(os.path.join(epis_path_depth, 'low_dim_obs.pkl'), 'rb') as f:
            demo = pickle.load(f)
        with open(os.path.join(epis_path_depth, 'variation_number.pkl'), 'rb') as f:
            demo.variation_number = pickle.load(f)

        # language description
        with open(os.path.join(epis_path_depth, 'variation_descriptions.pkl'), 'rb') as f:
            demo.language_descriptions = pickle.load(f)

        episode_keypoints = _keypoint_discovery(demo)

        tmp = []
        _add_keypoints_to_replay(
            tmp, 0, demo, episode_keypoints, epis_path_depth, epis_path_char, sim_name_to_real_name, cross_task_eval)
        tmp.append(f"{epis_path_depth}/front_rgb/0.png")
        buffer.append(tmp)
    # print("Average number of steps: ", sum([len(each[1]) for each in buffer])/len(buffer))
    return buffer



def form_obs(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict,
    taskname='None',
    cross_task_eval=0):
    
    # convert object id to char and average and discretize point cloud per object
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}

    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name:
            continue
        avg_point_list = []
        for camera in CAMERAS:
            mask = mask_dict[camera]
            point_cloud = point_cloud_dict[camera]
            if not np.any(mask == mask_id):
                continue
            avg_point_list.append(np.mean(point_cloud[mask == mask_id].reshape(-1, 3), axis = 0))

        avg_point = sum(avg_point_list) / len(avg_point_list)
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(point_to_voxel_index(avg_point))
    
    if cross_task_eval:
        return "['instruction': "+taskname+", "+str(real_name_to_avg_coord)+"]"
    return str(real_name_to_avg_coord)


def get_stored_demo_key_action_steps(dataset_root, task_name, episode_id, sim_name_to_real_name, cross_task_eval=1):
    cache_key = (dataset_root, task_name, episode_id, cross_task_eval)
    if cache_key in AUGMENTED_DEMO_STEPS_CACHE:
        return AUGMENTED_DEMO_STEPS_CACHE[cache_key]

    episode_path = os.path.join(dataset_root, task_name, "all_variations", "episodes", f"episode{episode_id}")
    with open(os.path.join(episode_path, "low_dim_obs.pkl"), "rb") as f:
        demo = pickle.load(f)
    with open(os.path.join(episode_path, "variation_descriptions.pkl"), "rb") as f:
        task_instruction = pickle.load(f)[0]

    steps = []
    for step_index, keypoint in enumerate(_keypoint_discovery(demo), start=1):
        mask_dict = _get_mask_dict(episode_path, keypoint)
        mask_id_to_sim_name_dict = _get_mask_id_to_name_dict(episode_path, keypoint)
        point_cloud_dict = _get_point_cloud_dict(episode_path, keypoint)

        mask_id_to_sim_name = {}
        for camera in CAMERAS:
            mask_id_to_sim_name.update(mask_id_to_sim_name_dict[camera])
        mask_id_to_real_name = {
            mask_id: sim_name_to_real_name[name]
            for mask_id, name in mask_id_to_sim_name.items()
            if name in sim_name_to_real_name
        }

        observation = form_obs(
            mask_dict,
            mask_id_to_real_name,
            point_cloud_dict,
            taskname=task_instruction,
            cross_task_eval=cross_task_eval,
        )
        action = _get_action(demo[keypoint], demo[keypoint])
        steps.append({"step": step_index, "keypoint_index": keypoint, "observation": observation, "action": action})

    result = {"task_instruction": task_instruction, "steps": steps}
    AUGMENTED_DEMO_STEPS_CACHE[cache_key] = result
    return result


def _format_augmented_demo(rank, task_name, episode_id, retrieval_scores, include_geometry, include_affordance):
    review = _load_augmented_review_cache().get((task_name, episode_id), {})
    demo = get_stored_demo_key_action_steps(
        seen_path,
        task_name,
        episode_id,
        seen_sim_name_to_real_name[task_name],
        cross_task_eval=1,
    )
    lines = [
        f"Seen demonstration {rank}:",
        "Task instruction:",
        demo["task_instruction"],
        f"Retrieval scores: score={retrieval_scores[0]:.4f}, S_dyn={retrieval_scores[1]:.4f}, S_geo={retrieval_scores[2]:.4f}, S_aff={retrieval_scores[3]:.4f}",
        "",
    ]
    if include_geometry:
        lines.extend([_format_feature_block("Geometry description g_i", review.get("geometry_g_i") or {}, GEOMETRY_FIELDS), ""])
    if include_affordance:
        lines.extend([_format_feature_block("Affordance description a_i", review.get("affordance_a_i") or {}, AFFORDANCE_FIELDS), ""])

    lines.append("Key observation-action trajectory:")
    for step in demo["steps"]:
        lines.extend(
            [
                f"Step {step['step']} observation:",
                step["observation"],
                f"Step {step['step']} 7D action:",
                json.dumps(step["action"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _format_augmented_user_prompt(
    ranked,
    all_demo_paths,
    query_observation,
    query_task_key,
    query_task_instruction,
    query_geometry,
    query_affordance,
    include_geometry=True,
    include_affordance=True,
):
    lines = [
        f"You will receive {len(ranked)} top-k retrieved seen demonstrations from the AGNOSTOS seen-task training set. Use all of them as in-context examples for the current unseen query.",
        "",
        "Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to the retrieved seen demonstrations using action trends, geometry, and affordances.",
        "",
        "Important rules:",
        "- Each seen demonstration includes per-key-action observations paired with the corresponding 7D action.",
        "- The unseen query includes only the current/initial observation, task instruction, and the descriptor types used by this ablation.",
        "- Do not use unseen demonstrations, unseen future frames, unseen ground-truth actions, or after-states.",
        "- Preserve the X-ICM output format: only a list of 7D action lists, such as [[x, y, z, roll, pitch, yaw, gripper], ...].",
        "",
    ]

    for rank, item in enumerate(ranked, start=1):
        score, selected_idx, s_dyn, s_geo, s_aff = item
        task_name, episode_id = _task_episode_from_path(all_demo_paths[selected_idx])
        lines.extend([
            _format_augmented_demo(
                rank,
                task_name,
                episode_id,
                (score, s_dyn, s_geo, s_aff),
                include_geometry,
                include_affordance,
            ),
            "",
        ])

    lines.extend([
        "Unseen query:",
        "Task instruction:",
        query_task_instruction,
        "Task key:",
        query_task_key,
        "",
        "Current observation:",
        query_observation,
        "",
    ])
    if include_geometry:
        lines.extend([_format_feature_block("Geometry description g_j", query_geometry, GEOMETRY_FIELDS), ""])
    if include_affordance:
        lines.extend([_format_feature_block("Affordance description a_j", query_affordance, AFFORDANCE_FIELDS), ""])
    lines.append("Predict the key 7D action sequence for the unseen task. Return only a Python-style list of 7D action lists:")
    return "\n".join(lines)




if __name__ == "__main__":
    pass
