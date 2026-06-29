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
    "action_primitive",
    "motion_type",
    "motion_axis",
    "contact_type",
    "contact_region",
    "constraint_type",
    "alignment_requirement",
]

CONTACT_HINT_FIELDS = [
    "contact_mode",
    "source_view",
    "target_object",
    "target_part",
    "points_2d_normalized",
    "contact_region_text",
    "candidate_contact_coordinates",
    "use_as",
]

GOAL_STATE_FIELDS = [
    "goal_state_type",
    "manipulated_object",
    "target_object_or_region",
    "required_final_relation",
    "contact_or_release_target",
    "required_motion_constraint",
    "required_orientation_or_alignment",
    "release_or_stop_condition",
    "success_check",
    "goal_tags",
    "transfer_note",
]

# Legacy name kept so older helper calls do not break. In the clean v1 path
# these are RoboPoint contact hints, not symbolic affordance descriptors and not
# a retrieval score.
AFFORDANCE_FIELDS = CONTACT_HINT_FIELDS

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

PROFILE_FIELDS = [
    "interaction_family",
    "motion_sequence",
    "contact_strategy",
    "target_relation",
    "axis_constraint",
    "articulation_model",
    "precision_driver",
    "transfer_caution",
]

PROFILE_FIELD_WEIGHTS = {
    "interaction_family": 0.22,
    "motion_sequence": 0.22,
    "contact_strategy": 0.16,
    "target_relation": 0.16,
    "axis_constraint": 0.10,
    "articulation_model": 0.08,
    "precision_driver": 0.04,
    "transfer_caution": 0.02,
}

V3_CONTACT_COMPATIBILITY = {
    "hole_over_vertical_stand": {
        "ring_to_peg_insertion": 1.00,
        "threaded_socket_insertion": 0.45,
        "shape_profile_insertion": 0.30,
        "nested_object_stacking": 0.20,
    },
    "object_into_shelf": {
        "object_into_open_receptacle": 0.85,
        "object_into_drawer": 0.75,
        "object_to_rack_placement": 0.65,
        "thin_object_slot_insertion": 0.45,
        "object_to_holder_placement": 0.35,
        "nested_object_stacking": 0.25,
    },
    "round_object_into_open_goal": {
        "object_into_open_receptacle": 0.85,
        "sweep_into_receptacle": 0.50,
        "object_to_holder_placement": 0.35,
        "nested_object_stacking": 0.30,
        "rigid_object_stacking": 0.20,
    },
    "tool_scoop_under_object": {
        "sweep_into_receptacle": 0.90,
        "tool_drag_to_target": 0.85,
        "surface_slide_to_target": 0.65,
        "remove_from_support_surface": 0.35,
    },
}

V3_FAMILY_GROUPS = [
    {
        "hole_over_vertical_stand",
        "ring_to_peg_insertion",
        "threaded_socket_insertion",
    },
    {
        "object_into_shelf",
        "object_into_open_receptacle",
        "object_into_drawer",
        "object_to_rack_placement",
    },
    {
        "round_object_into_open_goal",
        "object_into_open_receptacle",
        "sweep_into_receptacle",
    },
    {
        "tool_scoop_under_object",
        "tool_drag_to_target",
        "sweep_into_receptacle",
        "surface_slide_to_target",
    },
    {
        "linear_pull_from_slot",
        "linear_handle_pull",
        "remove_flat_object_from_rack",
    },
    {
        "hinged_door_close",
        "hinged_panel_close",
        "hinged_lid_open",
    },
    {
        "button_or_switch_press",
        "button_press",
    },
    {
        "knob_or_handle_rotation",
        "screw_closure",
    },
]

PLAN_FAMILY_COMPATIBILITY = {
    "flat_object_docking_place": {
        "near": {"object_to_holder_placement", "object_to_rack_placement"},
        "weak": {"object_on_flat_surface", "object_into_open_receptacle"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "thin_object_slot_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
            "knob_or_handle_rotation",
        },
    },
    "object_on_flat_surface": {
        "near": {"object_to_holder_placement", "object_to_rack_placement", "remove_from_support_surface"},
        "weak": {"flat_object_docking_place", "surface_slide_to_target"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "thin_object_slot_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
            "hinged_door_close",
            "hinged_panel_close",
        },
    },
    "hinged_door_close": {
        "near": {"linear_handle_pull", "hinged_panel_close", "hinged_lid_open"},
        "weak": {"surface_slide_to_target", "button_press"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "thin_object_slot_insertion",
            "screw_closure",
            "object_to_holder_placement",
            "object_to_rack_placement",
        },
    },
    "hinged_panel_close": {
        "near": {"linear_handle_pull", "hinged_door_close", "hinged_lid_open"},
        "weak": {"surface_slide_to_target", "button_press"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "thin_object_slot_insertion",
            "screw_closure",
            "object_to_holder_placement",
            "object_to_rack_placement",
        },
    },
    "hinged_lid_open": {
        "near": {"linear_handle_pull", "hinged_door_close", "hinged_panel_close"},
        "weak": {"remove_from_support_surface"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "thin_object_slot_insertion",
            "screw_closure",
        },
    },
    "button_or_switch_press": {
        "near": {"button_press"},
        "weak": {"knob_or_handle_rotation"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "object_to_holder_placement",
            "object_to_rack_placement",
            "object_into_open_receptacle",
            "object_into_drawer",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "linear_handle_pull",
        },
    },
    "object_into_open_receptacle": {
        "near": {"object_into_drawer", "object_into_shelf", "sweep_into_receptacle", "round_object_into_open_goal"},
        "weak": {"object_to_holder_placement", "object_to_rack_placement", "thin_object_slot_insertion"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "object_into_shelf": {
        "near": {"object_into_open_receptacle", "object_into_drawer", "object_to_rack_placement"},
        "weak": {"thin_object_slot_insertion", "object_to_holder_placement"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "elongated_object_into_stand": {
        "near": {"object_into_open_receptacle", "object_to_holder_placement", "object_to_rack_placement"},
        "weak": {"ring_to_peg_insertion", "thin_object_slot_insertion"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "hole_over_vertical_stand": {
        "near": {"ring_to_peg_insertion", "threaded_socket_insertion", "object_to_holder_placement"},
        "weak": {"nested_object_stacking", "shape_profile_insertion"},
        "blocked": {
            "rigid_object_stacking",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
            "knob_or_handle_rotation",
            "linear_handle_pull",
        },
    },
    "linear_pull_from_slot": {
        "near": {"linear_handle_pull", "remove_flat_object_from_rack"},
        "weak": {"thin_object_slot_insertion", "object_to_rack_placement"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "lift_lid_from_container": {
        "near": {"remove_from_support_surface", "linear_handle_pull"},
        "weak": {"screw_closure", "object_into_open_receptacle"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "button_press",
            "button_or_switch_press",
        },
    },
    "remove_flat_object_from_rack": {
        "near": {"linear_handle_pull", "remove_from_support_surface", "object_to_rack_placement"},
        "weak": {"thin_object_slot_insertion", "object_to_holder_placement"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "round_object_into_open_goal": {
        "near": {"object_into_open_receptacle", "sweep_into_receptacle"},
        "weak": {"object_to_holder_placement", "object_to_rack_placement"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "tool_scoop_under_object": {
        "near": {"tool_drag_to_target", "surface_slide_to_target", "sweep_into_receptacle"},
        "weak": {"remove_from_support_surface"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "deformable_drag_straighten": {
        "near": {"tool_drag_to_target", "surface_slide_to_target", "sweep_into_receptacle"},
        "weak": {"remove_from_support_surface"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
    "knob_or_handle_rotation": {
        "near": {"screw_closure"},
        "weak": {"button_press", "button_or_switch_press", "linear_handle_pull"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "object_to_holder_placement",
            "object_to_rack_placement",
            "object_into_open_receptacle",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
        },
    },
    "pour_to_target": {
        "near": {"object_to_holder_placement", "object_to_rack_placement"},
        "weak": {"linear_handle_pull", "object_into_open_receptacle"},
        "blocked": {
            "nested_object_stacking",
            "rigid_object_stacking",
            "shape_profile_insertion",
            "ring_to_peg_insertion",
            "threaded_socket_insertion",
            "screw_closure",
            "button_press",
            "button_or_switch_press",
        },
    },
}

PLAN_TIER_SCORES = {
    "exact": 1.0,
    "near": 0.72,
    "weak": 0.35,
    "blocked": 0.0,
    "unknown": 0.20,
}

TASK_PROFILE_OVERRIDES = {
    "close_jar": {
        "interaction_family": "screw_closure",
        "motion_sequence": "grasp_lid_align_with_rim_twist_about_vertical_axis_release",
        "contact_strategy": "rim_or_lid_grasp",
        "target_relation": "lid_seats_on_cylindrical_container_rim",
        "axis_constraint": "vertical_rotation_axis",
        "articulation_model": "screw_twist",
        "precision_driver": "keep_lid_centered_on_jar_rim",
        "transfer_caution": "not_a_generic_push_or_insert_demo",
    },
    "insert_onto_square_peg": {
        "interaction_family": "ring_to_peg_insertion",
        "motion_sequence": "grasp_ring_lift_align_hole_to_vertical_peg_lower_release",
        "contact_strategy": "edge_grasp_on_ring_body",
        "target_relation": "central_hole_goes_over_spoke_or_peg",
        "axis_constraint": "vertical_insertion_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "hole_center_alignment",
        "transfer_caution": "use_for_hole_over_peg_not_for_docking_flat_objects",
    },
    "light_bulb_in": {
        "interaction_family": "threaded_socket_insertion",
        "motion_sequence": "grasp_bulb_align_base_insert_then_twist",
        "contact_strategy": "body_or_base_grasp",
        "target_relation": "threaded_base_enters_socket",
        "axis_constraint": "socket_rotation_axis",
        "articulation_model": "insert_then_screw_twist",
        "precision_driver": "thread_alignment_before_twist",
        "transfer_caution": "not_a_plain_slot_insertion",
    },
    "meat_off_grill": {
        "interaction_family": "remove_from_support_surface",
        "motion_sequence": "grasp_object_lift_clear_support_surface",
        "contact_strategy": "body_grasp",
        "target_relation": "object_leaves_flat_support_surface",
        "axis_constraint": "vertical_clearance",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "avoid_dragging_on_grill",
        "transfer_caution": "not_a_target_placement_demo",
    },
    "open_drawer": {
        "interaction_family": "linear_handle_pull",
        "motion_sequence": "grasp_handle_pull_along_drawer_axis",
        "contact_strategy": "handle_grasp",
        "target_relation": "sliding_container_opens_outward",
        "axis_constraint": "linear_slide_axis",
        "articulation_model": "drawer_slide",
        "precision_driver": "hold_handle_through_pull",
        "transfer_caution": "good_for_pull_from_slot_or_handle_tasks",
    },
    "place_cups": {
        "interaction_family": "object_to_holder_placement",
        "motion_sequence": "grasp_cup_lift_align_to_holder_lower_release",
        "contact_strategy": "cup_body_or_rim_grasp",
        "target_relation": "cup_sits_or_hangs_on_holder",
        "axis_constraint": "vertical_placement_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "holder_contact_alignment",
        "transfer_caution": "closer_to_docking_than_shape_sorter_insertion",
    },
    "place_shape_in_shape_sorter": {
        "interaction_family": "shape_profile_insertion",
        "motion_sequence": "grasp_shape_lift_match_profile_to_hole_lower_release",
        "contact_strategy": "edge_grasp_on_shape_body",
        "target_relation": "shape_profile_passes_through_matching_hole",
        "axis_constraint": "vertical_insertion_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "shape_orientation_and_hole_match",
        "transfer_caution": "bad_analogy_for_flat_docking_pull_or_hinge_tasks",
    },
    "place_wine_at_rack_location": {
        "interaction_family": "object_to_rack_placement",
        "motion_sequence": "grasp_bottle_lift_align_to_rack_location_place_release",
        "contact_strategy": "bottle_body_or_neck_grasp",
        "target_relation": "object_resting_in_rack_slot_or_target_location",
        "axis_constraint": "free_motion_with_rack_alignment",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "target_location_alignment",
        "transfer_caution": "placement_not_full_insertion",
    },
    "push_buttons": {
        "interaction_family": "button_press",
        "motion_sequence": "move_to_button_top_press_release",
        "contact_strategy": "direct_surface_press",
        "target_relation": "end_effector_contacts_small_button_top",
        "axis_constraint": "surface_normal_press_axis",
        "articulation_model": "button_travel",
        "precision_driver": "button_identity_and_contact_point",
        "transfer_caution": "good_for_lamp_or_buzzer_press_tasks_only",
    },
    "put_groceries_in_cupboard": {
        "interaction_family": "object_into_open_receptacle",
        "motion_sequence": "grasp_object_lift_move_to_cupboard_opening_place_release",
        "contact_strategy": "body_grasp",
        "target_relation": "object_inside_open_cupboard_or_shelf",
        "axis_constraint": "free_motion_with_opening_clearance",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "avoid_shelf_collision",
        "transfer_caution": "placement_into_receptacle_not_slot_insertion",
    },
    "put_item_in_drawer": {
        "interaction_family": "object_into_drawer",
        "motion_sequence": "grasp_object_lift_move_into_open_drawer_release",
        "contact_strategy": "body_grasp",
        "target_relation": "object_inside_open_drawer",
        "axis_constraint": "free_motion_with_drawer_clearance",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "drawer_opening_clearance",
        "transfer_caution": "placement_into_container_not_handle_pull",
    },
    "put_money_in_safe": {
        "interaction_family": "thin_object_slot_insertion",
        "motion_sequence": "pinch_thin_object_align_edge_to_slot_insert_release",
        "contact_strategy": "pinch_thin_object_edge",
        "target_relation": "thin_object_enters_safe_slot",
        "axis_constraint": "slot_insertion_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "thin_edge_slot_alignment",
        "transfer_caution": "only_good_for_thin_slot_insertion",
    },
    "reach_and_drag": {
        "interaction_family": "tool_drag_to_target",
        "motion_sequence": "grasp_tool_contact_object_drag_toward_target",
        "contact_strategy": "tool_handle_grasp_and_object_edge_contact",
        "target_relation": "object_translates_on_table_to_target_region",
        "axis_constraint": "planar_drag_axis",
        "articulation_model": "surface_sliding_contact",
        "precision_driver": "maintain_tool_object_contact",
        "transfer_caution": "not_a_pick_and_place_demo",
    },
    "slide_block_to_color_target": {
        "interaction_family": "surface_slide_to_target",
        "motion_sequence": "contact_block_side_push_or_slide_to_target",
        "contact_strategy": "surface_push_contact",
        "target_relation": "object_slides_on_table_to_colored_target",
        "axis_constraint": "planar_translation_axis",
        "articulation_model": "surface_sliding_contact",
        "precision_driver": "target_region_direction",
        "transfer_caution": "not_for_lift_or_insert_tasks",
    },
    "stack_blocks": {
        "interaction_family": "rigid_object_stacking",
        "motion_sequence": "grasp_block_lift_align_above_base_block_lower_release",
        "contact_strategy": "body_grasp",
        "target_relation": "object_supports_on_top_of_another_object",
        "axis_constraint": "vertical_stack_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "stable_top_surface_alignment",
        "transfer_caution": "stacking_not_insertion",
    },
    "stack_cups": {
        "interaction_family": "nested_object_stacking",
        "motion_sequence": "grasp_cup_lift_align_opening_to_base_cup_lower_release",
        "contact_strategy": "cup_body_or_rim_grasp",
        "target_relation": "cup_nests_or_stacks_on_cup",
        "axis_constraint": "vertical_nesting_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "rim_and_opening_alignment",
        "transfer_caution": "cup_nesting_not_shape_profile_insertion",
    },
    "sweep_to_dustpan_of_size": {
        "interaction_family": "sweep_into_receptacle",
        "motion_sequence": "contact_dirt_or_tool_sweep_along_floor_into_dustpan",
        "contact_strategy": "tool_or_surface_push_contact",
        "target_relation": "loose_material_enters_dustpan_opening",
        "axis_constraint": "planar_sweep_axis",
        "articulation_model": "surface_sliding_contact",
        "precision_driver": "dustpan_lip_alignment",
        "transfer_caution": "not_a_grasped_object_placement_demo",
    },
    "turn_tap": {
        "interaction_family": "knob_or_handle_rotation",
        "motion_sequence": "grasp_knob_or_handle_rotate_about_axis_release",
        "contact_strategy": "knob_or_handle_grasp",
        "target_relation": "rotary_control_changes_state",
        "axis_constraint": "local_rotation_axis",
        "articulation_model": "rotary_joint",
        "precision_driver": "rotation_axis_and_direction",
        "transfer_caution": "good_for_oven_knob_not_for_push_button",
    },
    "put_toilet_roll_on_stand": {
        "interaction_family": "hole_over_vertical_stand",
        "motion_sequence": "grasp_roll_lift_align_center_hole_to_stand_lower_release",
        "contact_strategy": "body_grasp",
        "target_relation": "central_hole_goes_over_vertical_stand",
        "axis_constraint": "vertical_insertion_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "hole_center_to_stand_axis",
        "transfer_caution": "hole_over_peg_not_shape_sorter_profile",
    },
    "put_knife_on_chopping_board": {
        "interaction_family": "object_on_flat_surface",
        "motion_sequence": "grasp_handle_lift_place_on_flat_board_release",
        "contact_strategy": "handle_grasp",
        "target_relation": "elongated_object_resting_on_flat_surface",
        "axis_constraint": "free_motion_to_surface",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "object_orientation_on_board",
        "transfer_caution": "flat_surface_placement_not_container_insertion",
    },
    "close_fridge": {
        "interaction_family": "hinged_door_close",
        "motion_sequence": "contact_door_or_handle_push_along_swing_path_until_closed",
        "contact_strategy": "door_surface_or_handle_contact",
        "target_relation": "hinged_door_rotates_to_closed_frame",
        "axis_constraint": "vertical_hinge_axis",
        "articulation_model": "hinge_open_close",
        "precision_driver": "push_on_correct_side_of_door",
        "transfer_caution": "not_a_shape_insert_or_screw_task",
    },
    "close_microwave": {
        "interaction_family": "hinged_door_close",
        "motion_sequence": "contact_door_or_handle_push_along_swing_path_until_closed",
        "contact_strategy": "door_surface_or_handle_contact",
        "target_relation": "hinged_door_rotates_to_closed_frame",
        "axis_constraint": "vertical_hinge_axis",
        "articulation_model": "hinge_open_close",
        "precision_driver": "push_on_outer_door_panel",
        "transfer_caution": "not_a_shape_insert_or_screw_task",
    },
    "close_laptop_lid": {
        "interaction_family": "hinged_panel_close",
        "motion_sequence": "contact_lid_edge_or_panel_push_down_about_hinge",
        "contact_strategy": "panel_edge_or_surface_contact",
        "target_relation": "flat_panel_rotates_down_to_base",
        "axis_constraint": "horizontal_hinge_axis",
        "articulation_model": "hinge_open_close",
        "precision_driver": "push_on_lid_not_base",
        "transfer_caution": "not_a_grasped_object_placement_demo",
    },
    "phone_on_base": {
        "interaction_family": "flat_object_docking_place",
        "motion_sequence": "grasp_phone_body_lift_align_to_base_cradle_lower_release",
        "contact_strategy": "phone_body_grasp",
        "target_relation": "flat_phone_rests_on_base_cradle_or_charger",
        "axis_constraint": "free_motion_with_pose_alignment",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "phone_orientation_to_base_contacts",
        "transfer_caution": "prefer_holder_or_rack_placement_over_shape_sorter_insertion",
    },
    "toilet_seat_down": {
        "interaction_family": "hinged_panel_close",
        "motion_sequence": "contact_seat_rim_push_down_about_hinge",
        "contact_strategy": "rim_or_panel_contact",
        "target_relation": "seat_rotates_down_to_bowl",
        "axis_constraint": "horizontal_hinge_axis",
        "articulation_model": "hinge_open_close",
        "precision_driver": "contact_moving_seat_not_bowl",
        "transfer_caution": "not_a_pick_place_or_insert_task",
    },
    "lamp_off": {
        "interaction_family": "button_or_switch_press",
        "motion_sequence": "move_to_lamp_switch_press_release",
        "contact_strategy": "direct_surface_press",
        "target_relation": "small_switch_changes_lamp_state",
        "axis_constraint": "surface_normal_press_axis",
        "articulation_model": "button_or_switch_travel",
        "precision_driver": "switch_identity_and_contact_point",
        "transfer_caution": "prefer_push_buttons_examples",
    },
    "lamp_on": {
        "interaction_family": "button_or_switch_press",
        "motion_sequence": "move_to_lamp_switch_press_release",
        "contact_strategy": "direct_surface_press",
        "target_relation": "small_switch_changes_lamp_state",
        "axis_constraint": "surface_normal_press_axis",
        "articulation_model": "button_or_switch_travel",
        "precision_driver": "switch_identity_and_contact_point",
        "transfer_caution": "prefer_push_buttons_examples",
    },
    "put_books_on_bookshelf": {
        "interaction_family": "object_into_shelf",
        "motion_sequence": "grasp_books_lift_align_to_shelf_opening_insert_or_place_release",
        "contact_strategy": "book_body_grasp",
        "target_relation": "books_end_inside_shelf_opening",
        "axis_constraint": "free_motion_with_shelf_clearance",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "avoid_collision_with_shelf_edges",
        "transfer_caution": "shelf_placement_not_shape_profile_matching",
    },
    "put_umbrella_in_umbrella_stand": {
        "interaction_family": "elongated_object_into_stand",
        "motion_sequence": "grasp_umbrella_lift_align_long_axis_to_stand_opening_lower_release",
        "contact_strategy": "handle_or_body_grasp",
        "target_relation": "elongated_object_enters_top_open_stand",
        "axis_constraint": "vertical_insertion_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "long_axis_to_stand_opening",
        "transfer_caution": "more_like_vertical_receptacle_than_shape_sorter",
    },
    "open_grill": {
        "interaction_family": "hinged_lid_open",
        "motion_sequence": "grasp_handle_pull_up_about_hinge",
        "contact_strategy": "handle_grasp",
        "target_relation": "lid_rotates_open_from_grill_body",
        "axis_constraint": "horizontal_hinge_axis",
        "articulation_model": "hinge_open_close",
        "precision_driver": "hold_lid_handle_through_arc",
        "transfer_caution": "hinged_handle_motion_not_linear_drawer_pull",
    },
    "put_rubbish_in_bin": {
        "interaction_family": "object_into_open_receptacle",
        "motion_sequence": "grasp_rubbish_lift_move_over_bin_opening_lower_release",
        "contact_strategy": "object_body_grasp",
        "target_relation": "object_inside_top_open_bin",
        "axis_constraint": "vertical_drop_or_place_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "bin_opening_center",
        "transfer_caution": "receptacle_placement_not_slot_insertion",
    },
    "take_usb_out_of_computer": {
        "interaction_family": "linear_pull_from_slot",
        "motion_sequence": "pinch_usb_body_pull_along_port_axis_until_clear",
        "contact_strategy": "pinch_exposed_connector_body",
        "target_relation": "inserted_object_exits_slot",
        "axis_constraint": "linear_port_axis",
        "articulation_model": "rigid_linear_extraction",
        "precision_driver": "pull_straight_along_port_axis",
        "transfer_caution": "prefer_handle_pull_demos_over_insert_demos",
    },
    "take_lid_off_saucepan": {
        "interaction_family": "lift_lid_from_container",
        "motion_sequence": "grasp_knob_lift_lid_vertically_clear_rim",
        "contact_strategy": "knob_grasp",
        "target_relation": "lid_separates_from_pan_rim",
        "axis_constraint": "vertical_lift_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "grasp_lid_knob",
        "transfer_caution": "not_a_screw_twist_if_lid_is_loose",
    },
    "take_plate_off_colored_dish_rack": {
        "interaction_family": "remove_flat_object_from_rack",
        "motion_sequence": "grasp_plate_rim_lift_out_of_rack_clear_slot",
        "contact_strategy": "rim_grasp",
        "target_relation": "plate_leaves_rack_slot",
        "axis_constraint": "free_motion_with_slot_clearance",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "avoid_rack_collision",
        "transfer_caution": "removal_not_insertion",
    },
    "basketball_in_hoop": {
        "interaction_family": "round_object_into_open_goal",
        "motion_sequence": "grasp_ball_lift_align_over_hoop_lower_or_release",
        "contact_strategy": "ball_body_grasp",
        "target_relation": "ball_center_passes_through_hoop_ring",
        "axis_constraint": "vertical_goal_axis",
        "articulation_model": "rigid_free_motion",
        "precision_driver": "hoop_ring_center",
        "transfer_caution": "receptacle_goal_not_shape_sorter_profile",
    },
    "scoop_with_spatula": {
        "interaction_family": "tool_scoop_under_object",
        "motion_sequence": "grasp_spatula_handle_slide_blade_under_object_lift_or_carry",
        "contact_strategy": "tool_handle_grasp_blade_contact",
        "target_relation": "thin_blade_goes_under_target_object",
        "axis_constraint": "shallow_planar_approach_then_lift",
        "articulation_model": "tool_object_contact",
        "precision_driver": "blade_edge_under_object",
        "transfer_caution": "not_a_direct_body_grasp",
    },
    "straighten_rope": {
        "interaction_family": "deformable_drag_straighten",
        "motion_sequence": "pinch_rope_endpoint_drag_to_reduce_curve",
        "contact_strategy": "pinch_rope_endpoint_or_body",
        "target_relation": "deformable_object_changes_shape",
        "axis_constraint": "planar_drag_axis",
        "articulation_model": "deformable_contact",
        "precision_driver": "rope_endpoint_selection",
        "transfer_caution": "not_a_rigid_pick_place_task",
    },
    "turn_oven_on": {
        "interaction_family": "knob_or_handle_rotation",
        "motion_sequence": "grasp_or_contact_oven_knob_rotate_about_front_axis",
        "contact_strategy": "knob_grasp_or_tangent_contact",
        "target_relation": "rotary_control_changes_oven_state",
        "axis_constraint": "front_normal_rotation_axis",
        "articulation_model": "rotary_joint",
        "precision_driver": "knob_axis_and_rotation_direction",
        "transfer_caution": "prefer_turn_tap_over_push_button",
    },
    "beat_the_buzz": {
        "interaction_family": "button_or_switch_press",
        "motion_sequence": "move_to_buzzer_button_press_release",
        "contact_strategy": "direct_surface_press",
        "target_relation": "button_top_pressed",
        "axis_constraint": "surface_normal_press_axis",
        "articulation_model": "button_travel",
        "precision_driver": "button_top_center",
        "transfer_caution": "prefer_push_buttons_examples",
    },
    "water_plants": {
        "interaction_family": "pour_to_target",
        "motion_sequence": "grasp_watering_can_handle_lift_position_spout_over_plant_tilt_to_pour",
        "contact_strategy": "handle_grasp",
        "target_relation": "spout_aims_at_plant_target_region",
        "axis_constraint": "tilt_axis_with_spout_direction",
        "articulation_model": "rigid_object_tilt",
        "precision_driver": "spout_over_plant",
        "transfer_caution": "weak_seen_analogy_no_seen_pour_task",
    },
    "unplug_charger": {
        "interaction_family": "linear_pull_from_slot",
        "motion_sequence": "pinch_charger_body_pull_straight_out_from_socket",
        "contact_strategy": "pinch_exposed_plug_body",
        "target_relation": "inserted_plug_exits_wall_socket",
        "axis_constraint": "linear_socket_axis",
        "articulation_model": "rigid_linear_extraction",
        "precision_driver": "pull_straight_along_socket_axis",
        "transfer_caution": "prefer_handle_pull_demos_over_insert_demos",
    },
}

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
        "geometry": {"manipulated_object": "phone_and_base", "key_features": ["phone", "rectangular", "flat", "base", "dock", "cradle_contact", "alignment_sensitive"], "part_geometry": ["phone_body", "base_cradle"], "opening_geometry": "support_cradle", "axis_geometry": "free_motion", "clearance_geometry": "open_path", "task_relevant_geometric_cues": ["base_contact_patch", "phone_orientation"]},
        "affordance": {"grasp_affordance": "body_grasp", "contact_affordance": "lift_and_place", "motion_affordance": "place", "containment_affordance": "support_cradle", "articulation_affordance": "none", "required_contact_region": "phone_body", "preferred_contact_points": [], "precision_requirement": "high", "failure_sensitive_property": "pose_misalignment"},
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
    # Clean v1 restarts geometry as the only retrieval descriptor. RoboPoint is
    # contact-hint evidence for the final prompt after validation, not S_aff.
    return ".aff" in ranking_metric and "geo_aff" not in ranking_metric


def _is_v2_ranking(ranking_metric):
    return "v2" in ranking_metric


def _is_v3_ranking(ranking_metric):
    return "v3" in ranking_metric


def _is_v4_ranking(ranking_metric):
    return "v4" in ranking_metric


def _is_plan_ranking(ranking_metric):
    return "geo_plan" in ranking_metric or ".plan" in ranking_metric


def _augmented_weights(ranking_metric):
    if all(os.environ.get(name) is not None for name in ["XICM_GA_ALPHA", "XICM_GA_BETA", "XICM_GA_GAMMA"]):
        # Contact hints/points are prompt-only. Keep the gamma env var accepted
        # for older launch scripts, but never let it affect retrieval.
        return float(os.environ["XICM_GA_ALPHA"]), float(os.environ["XICM_GA_BETA"]), 0.0
    if _is_plan_ranking(ranking_metric):
        return 0.72, 0.03, 0.0
    if _is_v4_ranking(ranking_metric):
        return 0.70, 0.05, 0.0
    if _is_v3_ranking(ranking_metric):
        return 0.45, 0.10, 0.0
    if _is_v2_ranking(ranking_metric):
        return 0.82, 0.04, 0.0
    if "geo_aff" in ranking_metric:
        return 0.65, 0.35, 0.0
    if ".geo" in ranking_metric:
        return 0.65, 0.35, 0.0
    if ".aff" in ranking_metric:
        return 1.0, 0.0, 0.0
    return 1.0, 0.0, 0.0


def _profile_weights(ranking_metric):
    if _is_plan_ranking(ranking_metric):
        return (
            float(os.environ.get("XICM_GA_DELTA", "0.25")),
            float(os.environ.get("XICM_GA_PENALTY", "0.55")),
        )
    if _is_v4_ranking(ranking_metric):
        return (
            float(os.environ.get("XICM_GA_DELTA", "0.40")),
            float(os.environ.get("XICM_GA_PENALTY", "0.45")),
        )
    if _is_v3_ranking(ranking_metric):
        return (
            float(os.environ.get("XICM_GA_DELTA", "0.45")),
            float(os.environ.get("XICM_GA_PENALTY", "0.60")),
        )
    return (
        float(os.environ.get("XICM_GA_DELTA", "0.22")),
        float(os.environ.get("XICM_GA_PENALTY", "0.30")),
    )


def _task_episode_from_path(path):
    task = path.split("/")[-4]
    episode = int(path.split("/")[-1].replace("episode", "", 1))
    return task, episode


def _select_diverse_ranked_items(ranked, top_k):
    max_per_task = int(os.environ.get("XICM_GA_MAX_PER_TASK", "2"))
    max_per_family = int(os.environ.get("XICM_GA_MAX_PER_FAMILY", "3"))
    selected = []
    task_counts = {}
    family_counts = {}

    def add_item(item, enforce_task=True, enforce_family=True):
        task = item.get("task", "")
        family = _family_name(item.get("seen_profile", {}))
        if enforce_task and task_counts.get(task, 0) >= max_per_task:
            return False
        if enforce_family and family_counts.get(family, 0) >= max_per_family:
            return False
        selected.append(item)
        task_counts[task] = task_counts.get(task, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        return True

    for item in ranked:
        if len(selected) >= top_k:
            break
        add_item(item, enforce_task=True, enforce_family=True)

    for item in ranked:
        if len(selected) >= top_k:
            break
        if item in selected:
            continue
        add_item(item, enforce_task=True, enforce_family=False)

    for item in ranked:
        if len(selected) >= top_k:
            break
        if item in selected:
            continue
        add_item(item, enforce_task=False, enforce_family=False)

    return selected[:top_k]


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


def _first_label(value, default="unknown"):
    if isinstance(value, str) and value.strip():
        return value.strip().lower().replace(" ", "_")
    if isinstance(value, (list, tuple)):
        for item in value:
            label = _first_label(item, "")
            if label:
                return label
    return default


def _as_labels(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower().replace(" ", "_")] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        labels = []
        for item in value:
            labels.extend(_as_labels(item))
        return labels
    return []


def _action_from_affordance(affordance):
    action = _first_label(
        affordance.get("motion_affordance")
        or affordance.get("action_primitive")
        or affordance.get("contact_affordance"),
        "unknown",
    )
    if action in {"rotate", "rotate_part"}:
        return "twist"
    if action in {"lift_and_place", "lift_then_place"}:
        return "place"
    if action in {"lift_then_insert", "insert_then_twist"}:
        return "insert"
    if action in {"push_surface"}:
        return "push"
    if action in {"pull_handle"}:
        return "pull"
    if action in {"button_press"}:
        return "press"
    return action


def _action_from_task(task_key, action):
    task_text = _first_label(task_key, "")
    if "push_buttons" in task_text:
        return "press"
    if "sweep_to_dustpan" in task_text:
        return "sweep"
    return action


def _normalize_target_part(value):
    part = _first_label(value, "unknown")
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


def _normalize_primary_shape(value, text):
    shape = _first_label(value, "")
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


def _normalize_constraint_type(value, action, tags):
    constraint = _first_label(value, "unknown")
    tags = set(_as_labels(tags))
    if action in {"twist", "rotate"} or tags.intersection({"hinge", "sliding_axis", "rotational_axis"}):
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


def _normalize_state(value):
    state = _first_label(value, "unknown")
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


def _motion_type_from_axis(axis, action):
    axis = _first_label(axis, "unknown")
    action = _first_label(action, "unknown")
    if action in {"twist", "rotate"} or "rotat" in axis or "tilt" in axis:
        return "rotational"
    if action in {"insert"}:
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


def _motion_axis_from_axis(axis, action):
    axis = _first_label(axis, "unknown")
    action = _first_label(action, "unknown")
    if action in {"twist", "rotate"} or "rotat" in axis or "tilt" in axis:
        return "rotational"
    if action in {"insert"} or axis in {"slot", "hole", "front_opening", "top_opening", "support_cradle"}:
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


def _constraint_from_opening(opening, affordance):
    opening = _first_label(opening, "unknown")
    containment = _first_label(affordance.get("containment_affordance"), "")
    articulation = _first_label(affordance.get("articulation_affordance"), "")
    if any(token in opening for token in ["slot", "hole", "opening"]):
        return opening
    if containment in {"slot", "hole", "receptacle", "open_container", "closed_container"}:
        return containment
    if "hinge" in articulation or "drawer" in articulation:
        return "joint"
    if opening in {"support_cradle"}:
        return "support_surface"
    if opening in {"none", "unknown"}:
        return "none"
    return opening


def _contact_type_from_action(action):
    if action in {"pull", "lift", "place", "insert", "stack", "twist"}:
        return "grasp"
    if action == "press":
        return "press"
    if action in {"push", "slide", "sweep", "drag"}:
        return "surface_contact"
    return "unknown"


def _alignment_from_geometry(geometry, affordance):
    text = json.dumps({"geometry": geometry, "affordance": affordance}, sort_keys=True).lower()
    if any(token in text for token in ["alignment_sensitive", "misalignment", "slot", "hole", "peg", "socket", "shape_profile"]):
        return "high"
    if any(token in text for token in ["target", "rack", "shelf", "hoop", "rim"]):
        return "medium"
    return "low"


def _object_category_from_text(text):
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


def _canonical_geometry(task_key, geometry, affordance=None):
    affordance = affordance or {}
    geometry = geometry or {}
    action = _first_label(geometry.get("action_primitive"), "")
    if not action:
        action = _action_from_affordance(affordance)
    action = _action_from_task(task_key, action)
    old_tags = _as_labels(geometry.get("geometry_tags")) or _as_labels(geometry.get("task_relevant_geometric_cues")) or _as_labels(geometry.get("key_features"))
    old_tags = [
        tag for tag in old_tags
        if tag not in {"left", "right", "left_orientation", "right_orientation", "downward_direction", "upward_direction", "color", "red", "green", "blue", "maroon", "robot_arm", "elbow", "wrist", "static", "standing"}
    ]
    target_part = _normalize_target_part(
        geometry.get("target_part")
        or affordance.get("required_contact_region")
        or geometry.get("part_geometry")
    )
    manipulated_object = _first_label(geometry.get("manipulated_object"), task_key)
    text = " ".join([task_key, manipulated_object, target_part, " ".join(old_tags)])
    constraint_type = _normalize_constraint_type(
        _first_label(geometry.get("constraint_type"), _constraint_from_opening(geometry.get("opening_geometry"), affordance)),
        action,
        old_tags,
    )
    canonical = {
        "manipulated_object": manipulated_object,
        "object_category": _first_label(geometry.get("object_category"), _object_category_from_text(text)),
        "primary_shape": _normalize_primary_shape(geometry.get("primary_shape"), text),
        "target_part": target_part,
        "secondary_parts": _as_labels(geometry.get("secondary_parts")) or _as_labels(geometry.get("part_geometry"))[:3],
        "action_primitive": action,
        "motion_type": _first_label(geometry.get("motion_type"), _motion_type_from_axis(geometry.get("axis_geometry"), action)),
        "motion_axis": _first_label(geometry.get("motion_axis"), _motion_axis_from_axis(geometry.get("axis_geometry") or geometry.get("opening_geometry"), action)),
        "contact_type": _first_label(geometry.get("contact_type"), _contact_type_from_action(action)),
        "contact_region": _first_label(geometry.get("contact_region") or affordance.get("required_contact_region"), target_part),
        "constraint_type": constraint_type,
        "alignment_requirement": _first_label(geometry.get("alignment_requirement"), _alignment_from_geometry(geometry, affordance)),
        "state": _normalize_state(geometry.get("state") or geometry.get("pose_relation")),
        "geometry_tags": sorted(set(old_tags + _as_labels(geometry.get("geometry_tags")))),
    }
    if geometry.get("execution_clearance_hint") or geometry.get("clearance_geometry"):
        canonical["execution_clearance_hint"] = _first_label(
            geometry.get("execution_clearance_hint") or geometry.get("clearance_geometry"),
            "unknown",
        )
    return canonical


def _field_match_score(seen_geometry, query_geometry, field):
    seen_value = seen_geometry.get(field)
    query_value = query_geometry.get(field)
    if field == "geometry_tags":
        return _jaccard(set(_as_labels(seen_value)), set(_as_labels(query_value)))
    return _profile_value_similarity(seen_value, query_value)


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
    seen_geometry = _canonical_geometry("", seen_geometry)
    query_geometry = _canonical_geometry("", query_geometry)
    score = 0.0
    total = 0.0
    for field, weight in GEOMETRY_FIELD_WEIGHTS.items():
        score += weight * _field_match_score(seen_geometry, query_geometry, field)
        total += weight
    normalized = score / total if total else 0.0
    seen_action = _first_label(seen_geometry.get("action_primitive"), "")
    query_action = _first_label(query_geometry.get("action_primitive"), "")
    if seen_action and query_action and seen_action != query_action and "unknown" not in {seen_action, query_action}:
        return min(normalized, ACTION_MISMATCH_SCORE_CAP)
    return normalized


def _affordance_similarity(seen_affordance, query_affordance):
    # Clean v1 does not use RoboPoint/contact hints for retrieval.
    return 0.0


def _profile_tokens(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        tokens = set()
        for item in value:
            tokens.update(_profile_tokens(item))
        return tokens
    return set(_normalize_token(value))


def _interaction_profile(task_key, geometry, affordance):
    geometry = _canonical_geometry(task_key, geometry, affordance)
    profile = {
        "interaction_family": geometry.get("manipulated_object", task_key),
        "motion_sequence": geometry.get("action_primitive", "unknown"),
        "contact_strategy": geometry.get("contact_region") or geometry.get("contact_type") or "unknown",
        "target_relation": geometry.get("constraint_type") or "unknown",
        "axis_constraint": geometry.get("motion_axis", "unknown"),
        "articulation_model": geometry.get("motion_type", "unknown"),
        "precision_driver": geometry.get("alignment_requirement") or "unknown",
        "transfer_caution": "primitive_geometry_profile",
    }
    override = TASK_PROFILE_OVERRIDES.get(task_key)
    if override:
        profile.update(override)
    return profile


def _goal_state_type_from_profile(profile):
    semantic_fields = [
        "interaction_family",
        "motion_sequence",
        "contact_strategy",
        "target_relation",
        "axis_constraint",
        "articulation_model",
        "precision_driver",
    ]
    text = " ".join(str(profile.get(field, "")) for field in semantic_fields).lower()
    if any(token in text for token in ["button", "switch", "knob", "tap", "rotary_control", "oven_state"]):
        return "control_or_articulation_state"
    if any(token in text for token in ["hinge", "door", "lid", "seat", "sliding_container", "opens", "closes"]):
        return "articulated_part_state"
    if any(token in text for token in ["exits", "leaves", "separates", "remove", "removal", "extraction", "out_from_socket"]):
        return "removal_or_extraction_state"
    if any(token in text for token in ["pour", "spout", "plant"]):
        return "pouring_target_state"
    if any(token in text for token in ["deformable", "rope", "straighten"]):
        return "deformable_shape_state"
    if any(token in text for token in ["scoop", "sweep", "drag", "slide_blade", "tool"]):
        return "tool_contact_state"
    if any(token in text for token in ["hole", "peg", "spoke", "socket", "thread", "shape_profile", "insert"]):
        return "aligned_insertion_or_docking"
    if any(token in text for token in ["inside", "shelf", "receptacle", "container", "hoop", "bin", "drawer"]):
        return "placement_inside_or_on_target"
    if any(token in text for token in ["support", "holder", "rack", "cradle", "dock", "resting", "sits", "hangs"]):
        return "supported_or_docked_pose"
    return "task_goal_state"


def _goal_state_descriptor(task_key, geometry, affordance, profile=None, raw_goal=None, use_as="query_goal_state"):
    if isinstance(raw_goal, dict) and raw_goal:
        transfer_note = (
            "unseen query success state; prioritize this over retrieved demo goals"
            if use_as == "query_goal_state"
            else "seen demo success state; use as an analogy only if compatible with the unseen query goal"
        )
        goal = {
            "goal_state_type": raw_goal.get("goal_state_type") or raw_goal.get("target_pose_type") or "unknown",
            "manipulated_object": raw_goal.get("manipulated_object") or task_key,
            "target_object_or_region": raw_goal.get("target_object_or_region") or "unknown",
            "required_final_relation": (
                raw_goal.get("required_final_relation")
                or raw_goal.get("required_spatial_relation")
                or raw_goal.get("containment_requirement")
                or "unknown"
            ),
            "contact_or_release_target": raw_goal.get("contact_or_release_target") or raw_goal.get("target_object_or_region") or "unknown",
            "required_motion_constraint": (
                raw_goal.get("required_motion_constraint")
                or raw_goal.get("placement_mode")
                or raw_goal.get("support_type")
                or "unknown"
            ),
            "required_orientation_or_alignment": (
                raw_goal.get("required_orientation_or_alignment")
                or raw_goal.get("required_object_orientation")
                or raw_goal.get("alignment_requirement")
                or "unknown"
            ),
            "release_or_stop_condition": raw_goal.get("release_or_stop_condition") or raw_goal.get("release_condition") or "unknown",
            "success_check": raw_goal.get("success_check") or raw_goal.get("release_condition") or "unknown",
            "goal_tags": raw_goal.get("goal_tags") or raw_goal.get("target_pose_tags") or [],
            "transfer_note": raw_goal.get("transfer_note") or transfer_note,
        }
        return goal

    geometry = _canonical_geometry(task_key, geometry, affordance)
    profile = profile or _interaction_profile(task_key, geometry, affordance)
    target_relation = _first_label(profile.get("target_relation"), geometry.get("constraint_type", "unknown"))
    precision_driver = _first_label(profile.get("precision_driver"), geometry.get("alignment_requirement", "unknown"))
    goal_type = _goal_state_type_from_profile(profile)
    manipulated_object = geometry.get("manipulated_object") or task_key
    contact_or_release_target = (
        geometry.get("target_part")
        or geometry.get("contact_region")
        or precision_driver
        or "unknown"
    )
    motion_constraint = (
        profile.get("axis_constraint")
        or geometry.get("motion_axis")
        or profile.get("motion_sequence")
        or "unknown"
    )
    orientation_or_alignment = precision_driver
    if geometry.get("alignment_requirement") and geometry.get("alignment_requirement") != "unknown":
        orientation_or_alignment = f"{geometry.get('alignment_requirement')}_alignment:{precision_driver}"

    if goal_type in {"placement_inside_or_on_target", "supported_or_docked_pose", "aligned_insertion_or_docking"}:
        release_or_stop = f"release only after {target_relation}"
    elif goal_type in {"removal_or_extraction_state", "articulated_part_state", "control_or_articulation_state", "deformable_shape_state", "tool_contact_state", "pouring_target_state"}:
        release_or_stop = f"stop when {target_relation}"
    else:
        release_or_stop = f"complete when {target_relation}"

    tags = sorted(
        set(
            _as_labels(goal_type)
            + _as_labels(profile.get("interaction_family"))
            + _as_labels(profile.get("target_relation"))
            + _as_labels(profile.get("axis_constraint"))
            + _as_labels(geometry.get("action_primitive"))
            + _as_labels(geometry.get("constraint_type"))
        )
    )
    transfer_note = (
        "unseen query success state; prioritize this over retrieved demo goals"
        if use_as == "query_goal_state"
        else "seen demo success state; use as an analogy only if compatible with the unseen query goal"
    )
    return {
        "goal_state_type": goal_type,
        "manipulated_object": manipulated_object,
        "target_object_or_region": precision_driver if precision_driver != "unknown" else target_relation,
        "required_final_relation": target_relation,
        "contact_or_release_target": contact_or_release_target,
        "required_motion_constraint": _first_label(motion_constraint, "unknown"),
        "required_orientation_or_alignment": orientation_or_alignment,
        "release_or_stop_condition": release_or_stop,
        "success_check": target_relation,
        "goal_tags": tags,
        "transfer_note": transfer_note,
    }


def _profile_value_similarity(seen_value, query_value):
    seen_tokens = _profile_tokens(seen_value)
    query_tokens = _profile_tokens(query_value)
    if not seen_tokens or not query_tokens:
        return 0.0
    if seen_tokens == query_tokens:
        return 1.0
    overlap = len(seen_tokens & query_tokens) / len(seen_tokens | query_tokens)
    if overlap > 0:
        return overlap
    seen_text = "_".join(sorted(seen_tokens))
    query_text = "_".join(sorted(query_tokens))
    if seen_text in query_text or query_text in seen_text:
        return 0.6
    return 0.0


def _profile_similarity(seen_profile, query_profile):
    score = 0.0
    total = 0.0
    for field, weight in PROFILE_FIELD_WEIGHTS.items():
        score += weight * _profile_value_similarity(seen_profile.get(field), query_profile.get(field))
        total += weight
    if total == 0:
        return 0.0
    return score / total


def _profile_has(profile, *needles):
    text = " ".join(str(profile.get(field, "")) for field in PROFILE_FIELDS).lower()
    return any(needle in text for needle in needles)


def _family_name(profile):
    return str(profile.get("interaction_family", "")).strip().lower()


def _contact_family_similarity(seen_profile, query_profile):
    seen_family = _family_name(seen_profile)
    query_family = _family_name(query_profile)
    if not seen_family or not query_family:
        return 0.0
    if seen_family == query_family:
        return 1.0

    compatibility = V3_CONTACT_COMPATIBILITY.get(query_family, {})
    if seen_family in compatibility:
        return compatibility[seen_family]

    for group in V3_FAMILY_GROUPS:
        if seen_family in group and query_family in group:
            return 0.65

    return 0.35 * _profile_value_similarity(seen_family, query_family)


def _plan_tier_for_family(seen_family, query_family):
    if not seen_family or not query_family:
        return "unknown", "missing interaction family"
    if seen_family == query_family:
        return "exact", "same interaction family"

    spec = PLAN_FAMILY_COMPATIBILITY.get(query_family, {})
    for tier in ("near", "weak", "blocked"):
        if seen_family in spec.get(tier, set()):
            return tier, f"{seen_family} is {tier} for query family {query_family}"

    for group in V3_FAMILY_GROUPS:
        if seen_family in group and query_family in group:
            return "near", "shared v3 mechanical family group"

    contact_score = _contact_family_similarity(
        {"interaction_family": seen_family},
        {"interaction_family": query_family},
    )
    if contact_score >= 0.80:
        return "near", f"high contact-family compatibility {contact_score:.2f}"
    if contact_score >= 0.35:
        return "weak", f"weak contact-family compatibility {contact_score:.2f}"
    if spec:
        return "blocked", f"not listed as compatible with query family {query_family}"
    return "unknown", "no plan compatibility rule"


def _plan_compatibility(seen_profile, query_profile):
    seen_family = _family_name(seen_profile)
    query_family = _family_name(query_profile)
    tier, reason = _plan_tier_for_family(seen_family, query_family)
    return {
        "tier": tier,
        "score": PLAN_TIER_SCORES.get(tier, PLAN_TIER_SCORES["unknown"]),
        "reason": reason,
        "seen_family": seen_family,
        "query_family": query_family,
    }


def _plan_score_cap(tier):
    if tier == "blocked":
        return float(os.environ.get("XICM_GA_PLAN_BLOCK_CAP", "0.15"))
    if tier == "weak":
        return float(os.environ.get("XICM_GA_PLAN_WEAK_CAP", "0.55"))
    if tier == "unknown":
        return float(os.environ.get("XICM_GA_PLAN_UNKNOWN_CAP", "0.45"))
    return None


def _field_similarity(a, b, field):
    return _profile_value_similarity(a.get(field), b.get(field))


def _mechanical_similarity(seen_profile, query_profile, seen_geometry, query_geometry, seen_affordance, query_affordance):
    seen_geometry = _canonical_geometry("", seen_geometry, seen_affordance)
    query_geometry = _canonical_geometry("", query_geometry, query_affordance)
    geometry_score = (
        0.45 * _field_match_score(seen_geometry, query_geometry, "action_primitive")
        + 0.15 * _field_match_score(seen_geometry, query_geometry, "motion_type")
        + 0.15 * _field_match_score(seen_geometry, query_geometry, "motion_axis")
        + 0.12 * _field_match_score(seen_geometry, query_geometry, "contact_region")
        + 0.13 * _field_match_score(seen_geometry, query_geometry, "constraint_type")
    )
    seen_action = _first_label(seen_geometry.get("action_primitive"), "")
    query_action = _first_label(query_geometry.get("action_primitive"), "")
    if seen_action and query_action and seen_action != query_action and "unknown" not in {seen_action, query_action}:
        geometry_score = min(geometry_score, ACTION_MISMATCH_SCORE_CAP)
    profile_score = (
        0.38 * _contact_family_similarity(seen_profile, query_profile)
        + 0.20 * _profile_value_similarity(seen_profile.get("target_relation"), query_profile.get("target_relation"))
        + 0.17 * _profile_value_similarity(seen_profile.get("motion_sequence"), query_profile.get("motion_sequence"))
        + 0.13 * _profile_value_similarity(seen_profile.get("contact_strategy"), query_profile.get("contact_strategy"))
        + 0.12 * _profile_value_similarity(seen_profile.get("axis_constraint"), query_profile.get("axis_constraint"))
    )
    return max(0.0, min(1.0, 0.55 * profile_score + 0.45 * geometry_score))


def _profile_conflict_penalty(seen_profile, query_profile):
    penalty = 0.0

    if _profile_has(query_profile, "linear_pull_from_slot", "extraction", "pull_straight"):
        if _profile_has(seen_profile, "insert", "shape_profile", "slot_insertion", "screw_closure"):
            penalty += 0.40
        if not _profile_has(seen_profile, "pull", "handle", "drawer"):
            penalty += 0.10

    if _profile_has(query_profile, "flat_object_docking_place", "phone"):
        if _profile_has(seen_profile, "shape_profile_insertion", "screw_closure", "thin_object_slot_insertion"):
            penalty += 0.35
        if _profile_has(seen_profile, "holder", "rack", "placement", "place"):
            penalty = max(0.0, penalty - 0.12)

    if _profile_has(query_profile, "hinged_door_close", "hinged_panel_close"):
        if _profile_has(seen_profile, "shape_profile_insertion", "slot_insertion", "screw_closure"):
            penalty += 0.35
        if not _profile_has(seen_profile, "push", "hinge", "surface", "handle"):
            penalty += 0.10

    if _profile_has(query_profile, "button_or_switch_press"):
        if not _profile_has(seen_profile, "button", "press", "switch"):
            penalty += 0.30

    if _profile_has(query_profile, "knob_or_handle_rotation"):
        if not _profile_has(seen_profile, "rotate", "rotation", "rotary", "twist"):
            penalty += 0.30

    if _profile_has(query_profile, "pour_to_target"):
        if not _profile_has(seen_profile, "lift", "place", "handle", "target"):
            penalty += 0.20

    if _profile_has(query_profile, "object_into_open_receptacle"):
        if _profile_has(seen_profile, "shape_profile_insertion", "screw_closure", "button"):
            penalty += 0.25

    return min(1.0, max(0.0, penalty))


def _v3_conflict_penalty(seen_profile, query_profile):
    penalty = _profile_conflict_penalty(seen_profile, query_profile)

    if _profile_has(query_profile, "tool_scoop_under_object", "slide_blade_under", "shallow_planar_approach"):
        if not _profile_has(seen_profile, "tool", "sweep", "drag", "slide", "surface_sliding_contact"):
            penalty += 0.55
        if _profile_has(seen_profile, "screw_closure", "twist", "stack", "shape_profile_insertion", "ring_to_peg"):
            penalty += 0.25

    if _profile_has(query_profile, "object_into_shelf", "shelf_clearance"):
        if _profile_has(seen_profile, "object_to_holder_placement", "nested_object_stacking", "cup"):
            penalty += 0.35
        if not _profile_has(seen_profile, "shelf", "rack", "drawer", "receptacle", "slot"):
            penalty += 0.15

    if _profile_has(query_profile, "round_object_into_open_goal", "hoop_ring_center"):
        if _profile_has(seen_profile, "nested_object_stacking", "rigid_object_stacking"):
            penalty += 0.35
        if not _profile_has(seen_profile, "receptacle", "goal", "open", "container", "target"):
            penalty += 0.15

    if _profile_has(query_profile, "hole_over_vertical_stand", "hole_center_to_stand_axis"):
        if not _profile_has(seen_profile, "hole", "peg", "spoke", "vertical", "socket"):
            penalty += 0.35
        if _profile_has(seen_profile, "shape_profile_insertion"):
            penalty += 0.12

    return min(1.0, max(0.0, penalty))


def _attention_bias_for_ranked_items(ranked):
    if not ranked:
        return ranked
    scores = [item["score"] for item in ranked]
    low = min(scores)
    high = max(scores)
    span = high - low
    for item in ranked:
        if span <= 1e-9:
            item["attention_bias"] = 1.0
        else:
            item["attention_bias"] = max(0.0, min(1.0, 0.10 + 0.90 * (item["score"] - low) / span))
    return ranked


def _write_retrieval_audit(ranking_metric, query_task, query_profile, ranked):
    audit_path = os.environ.get("XICM_GA_AUDIT_JSONL")
    if not audit_path:
        return
    top_limit = int(os.environ.get("XICM_GA_AUDIT_TOPK", "20"))
    tier_counts = {}
    family_counts = {}
    top = []
    for item in ranked[:top_limit]:
        tier = item.get("compatibility_tier", "not_plan_scored")
        family = _family_name(item.get("seen_profile", {})) or "unknown"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        top.append(
            {
                "rank": len(top) + 1,
                "task": item.get("task"),
                "episode_id": item.get("episode_id"),
                "score": item.get("score"),
                "raw_score": item.get("raw_score"),
                "attention_bias": item.get("attention_bias"),
                "s_dyn": item.get("s_dyn"),
                "s_geo": item.get("s_geo"),
                "s_profile": item.get("s_profile"),
                "s_plan": item.get("s_plan"),
                "transfer_penalty": item.get("penalty"),
                "compatibility_tier": tier,
                "compatibility_reason": item.get("compatibility_reason"),
                "seen_family": family,
                "score_cap": item.get("score_cap"),
            }
        )
    record = {
        "ranking_metric": ranking_metric,
        "query_task": query_task,
        "query_family": _family_name(query_profile),
        "tier_counts": tier_counts,
        "seen_family_counts": family_counts,
        "top": top,
    }
    try:
        directory = os.path.dirname(audit_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(audit_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"WARNING: failed to write retrieval audit {audit_path}: {exc}")


def _contact_mode_from_affordance(affordance):
    grasp = _first_label(affordance.get("grasp_affordance"), "")
    contact = _first_label(affordance.get("contact_affordance"), "")
    motion = _first_label(affordance.get("motion_affordance"), "")
    combined = " ".join([grasp, contact, motion])
    if any(token in combined for token in ["press", "button"]):
        return "press_point"
    if any(token in combined for token in ["push", "surface_contact", "sweep", "slide", "drag"]):
        return "single_contact"
    if any(token in combined for token in ["grasp", "pinch", "pull", "twist", "lift", "place", "insert"]):
        return "grasp_pair"
    return "region_hint"


def _contact_hints_from_affordance(task_key, affordance, use_as):
    affordance = affordance or {}
    region = _first_label(affordance.get("required_contact_region"), "unknown")
    points = affordance.get("preferred_contact_points") or []
    return {
        "contact_mode": _contact_mode_from_affordance(affordance),
        "source_view": "front_rgb_initial",
        "target_object": _first_label(task_key, "unknown"),
        "target_part": _normalize_target_part(region),
        "points_2d_normalized": points,
        "contact_region_text": region,
        "candidate_contact_coordinates": [],
        "use_as": use_as,
    }


def _query_descriptors(task_key, language_goal):
    if task_key in UNSEEN_DESCRIPTOR_RULES:
        item = UNSEEN_DESCRIPTOR_RULES[task_key]
        affordance = item.get("affordance") or {}
        return (
            _canonical_geometry(task_key, item["geometry"], affordance),
            _contact_hints_from_affordance(task_key, affordance, "unseen_query_contact_region_hint_not_retrieval"),
        )
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
    return _canonical_geometry(task_key, geometry, {}), _contact_hints_from_affordance(
        task_key,
        {},
        "unseen_query_contact_region_hint_not_retrieval",
    )


def _format_value(value):
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _format_feature_block(title, values, fields):
    lines = [f"{title}:"]
    for field in fields:
        default = [] if field in {"secondary_parts", "geometry_tags", "points_2d_normalized", "candidate_contact_coordinates", "goal_tags"} else "unknown"
        lines.append(f"- {field}: {_format_value(values.get(field, default))}")
    return "\n".join(lines)


def _format_geometry_field_guide():
    return "\n".join(
        [
            "Geometry descriptor field guide:",
            "- action_primitive: the primitive manipulation verb to transfer first, such as pull, push, twist, press, place, insert, slide, sweep, scoop, or lift.",
            "- motion_type: the coarse motion family, such as linear, rotational, vertical, planar, or insertion.",
            "- motion_axis: the primitive movement axis or relation, such as horizontal, vertical, rotational, into_opening, across_surface, or surface_normal.",
            "- contact_type: how the robot should physically interact, such as grasp, press, or surface_contact.",
            "- contact_region: the object part or region that should be touched or grasped.",
            "- constraint_type: the physical constraint that shapes the action, such as joint, slot, hole, container, support_surface, or free_space.",
            "- alignment_requirement: how much geometric alignment matters before motion; high means orientation/centering is important.",
            "- execution_clearance_hint, if present: a movement hint for avoiding collision; use it for the final action path, not for deciding demo similarity.",
        ]
    )


def _format_contact_field_guide():
    return "\n".join(
        [
            "Contact hint field guide:",
            "- contact_mode: whether the hint is a single contact, grasp pair, press point, or region-level cue.",
            "- target_object and target_part: the object and part the contact hint refers to.",
            "- points_2d_normalized: optional image-space points in normalized coordinates; use as visual contact hints for action generation, never as retrieval scores.",
            "- contact_region_text: short natural-language region name for the intended contact.",
            "- candidate_contact_coordinates: optional 3D/voxel contact candidates when available.",
            "- use_as: explains whether the hint is from a seen demo or the unseen query.",
        ]
    )


def _format_goal_state_field_guide():
    return "\n".join(
        [
            "Goal-state/contact-pose descriptor field guide:",
            "- h_i describes what the seen demo is trying to finish with; h_j describes the unseen query's required success state.",
            "- Retrieved demos are action analogies: their object identity and final goal may differ from the unseen query.",
            "- goal_state_type: broad success family, such as supported_or_docked_pose, placement_inside_or_on_target, aligned_insertion_or_docking, articulated_part_state, control_or_articulation_state, removal_or_extraction_state, tool_contact_state, deformable_shape_state, or pouring_target_state.",
            "- required_final_relation: the final physical relation that must be true when the task succeeds.",
            "- target_object_or_region and contact_or_release_target: the target site, object part, or reference region that matters for the final state.",
            "- required_motion_constraint: the movement, axis, or constraint needed to reach that state.",
            "- required_orientation_or_alignment: orientation, centering, insertion, docking, or contact alignment needed before release/stop.",
            "- release_or_stop_condition and success_check: use these to decide when to open the gripper, stop pushing/pulling/rotating, or finish the motion.",
            "- transfer_note: reminds whether the descriptor is a seen-demo analogy or the unseen-query goal.",
        ]
    )


def _format_profile_block(title, values):
    lines = [f"{title}:"]
    for field in PROFILE_FIELDS:
        lines.append(f"- {field}: {_format_value(values.get(field, 'unknown'))}")
    return "\n".join(lines)


def _rank_augmented_indices(similarity, all_demo_paths, query_geometry, query_affordance, ranking_metric, top_k):
    review_cache = _load_augmented_review_cache()
    sim_min = float(np.min(similarity))
    sim_max = float(np.max(similarity))
    sim_span = sim_max - sim_min
    alpha, beta, gamma = _augmented_weights(ranking_metric)
    delta, penalty_weight = _profile_weights(ranking_metric)
    use_v2 = _is_v2_ranking(ranking_metric)
    use_v3 = _is_v3_ranking(ranking_metric)
    use_v4 = _is_v4_ranking(ranking_metric)
    use_plan = _is_plan_ranking(ranking_metric)
    plan_weight = float(os.environ.get("XICM_GA_PLAN_WEIGHT", "0.45")) if use_plan else 0.0
    query_task = query_geometry.get("task_key") or query_geometry.get("manipulated_object") or ""
    query_profile = _interaction_profile(query_task, query_geometry, query_affordance)
    ranked = []
    for idx, demo_path in enumerate(all_demo_paths):
        task, episode_id = _task_episode_from_path(demo_path)
        row = review_cache.get((task, episode_id))
        if row is None:
            continue
        seen_affordance = row.get("contact_hints_i") or row.get("affordance_a_i") or {}
        seen_geometry = _canonical_geometry(task, row.get("geometry_g_i") or {}, seen_affordance)
        s_dyn = 0.0 if sim_span == 0 else (float(similarity[idx]) - sim_min) / sim_span
        s_geo = _geometry_similarity(seen_geometry, query_geometry)
        s_aff = 0.0
        seen_profile = _interaction_profile(task, seen_geometry, seen_affordance)
        if use_v3 or use_v4 or use_plan:
            s_profile = _mechanical_similarity(
                seen_profile,
                query_profile,
                seen_geometry,
                query_geometry,
                seen_affordance,
                query_affordance,
            )
            penalty = _v3_conflict_penalty(seen_profile, query_profile)
        else:
            s_profile = _profile_similarity(seen_profile, query_profile)
            penalty = _profile_conflict_penalty(seen_profile, query_profile) if use_v2 else 0.0
        plan_compat = _plan_compatibility(seen_profile, query_profile) if use_plan else {
            "tier": "not_plan_scored",
            "score": 0.0,
            "reason": "",
            "seen_family": _family_name(seen_profile),
            "query_family": _family_name(query_profile),
        }
        s_plan = plan_compat["score"]
        seen_goal_state = _goal_state_descriptor(
            task,
            seen_geometry,
            seen_affordance,
            seen_profile,
            raw_goal=(
                row.get("goal_state_h_i")
                or row.get("target_pose_h_i")
                or row.get("target_pose_i")
                or row.get("target_pose_descriptor_i")
                or row.get("goal_state_descriptor_i")
            ),
            use_as="seen_demo_goal_state",
        )
        score = alpha * s_dyn + beta * s_geo
        if use_v2 or use_v3 or use_v4 or use_plan:
            score += delta * s_profile - penalty_weight * penalty
        if use_plan:
            score += plan_weight * s_plan
        raw_score = score
        score_cap = _plan_score_cap(plan_compat["tier"]) if use_plan else None
        if score_cap is not None:
            score = min(score, score_cap)
        ranked.append(
            {
                "score": score,
                "raw_score": raw_score,
                "score_cap": score_cap,
                "index": idx,
                "task": task,
                "episode_id": episode_id,
                "s_dyn": s_dyn,
                "s_geo": s_geo,
                "s_aff": s_aff,
                "s_profile": s_profile,
                "s_plan": s_plan,
                "penalty": penalty,
                "compatibility_tier": plan_compat["tier"],
                "compatibility_reason": plan_compat["reason"],
                "query_family": plan_compat["query_family"],
                "seen_family": plan_compat["seen_family"],
                "seen_profile": seen_profile,
                "seen_goal_state": seen_goal_state,
            }
        )
    ranked.sort(reverse=True, key=lambda item: item["score"])
    if use_v3 or use_v4 or use_plan:
        ranked = _select_diverse_ranked_items(ranked, top_k)
    else:
        ranked = ranked[:top_k]
    ranked = _attention_bias_for_ranked_items(ranked)
    _write_retrieval_audit(ranking_metric, query_task, query_profile, ranked)
    return ranked

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
                query_geometry = dict(query_geometry)
                query_geometry["task_key"] = type(self).__name__
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
                    use_v2=_is_v2_ranking(ranking_metric) or _is_v3_ranking(ranking_metric) or _is_v4_ranking(ranking_metric) or _is_plan_ranking(ranking_metric),
                    use_v3=_is_v3_ranking(ranking_metric),
                    use_v4=_is_v4_ranking(ranking_metric),
                    use_plan=_is_plan_ranking(ranking_metric),
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


def _compact_summary_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value[:4]) or "unknown"
    if isinstance(value, dict):
        return ", ".join(f"{key}={val}" for key, val in list(value.items())[:4]) or "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _scene_summary(task_instruction, task_key, geometry, affordance, profile):
    geometry = _canonical_geometry(task_key, geometry, affordance)
    manipulated_object = geometry.get("manipulated_object") or task_key
    primary_shape = geometry.get("primary_shape") or "unknown shape"
    key_geometry = geometry.get("geometry_tags") or []
    action = geometry.get("action_primitive") or profile.get("motion_sequence") or "unknown"
    contact = geometry.get("contact_region") or profile.get("contact_strategy") or "unknown"
    target_relation = geometry.get("constraint_type") or profile.get("target_relation") or "unknown"
    axis = geometry.get("motion_axis") or profile.get("axis_constraint") or "unknown"
    precision = geometry.get("alignment_requirement") or profile.get("precision_driver") or "unknown"
    caution = profile.get("transfer_caution") or "match the contact mode before copying coordinates"
    return (
        f"Task '{task_instruction}' manipulates {manipulated_object}. "
        f"Relevant shape/parts: {_compact_summary_value(primary_shape)}. "
        f"Geometric cues: {_compact_summary_value(key_geometry)}. "
        f"Target relation: {_compact_summary_value(target_relation)}. "
        f"Action primitive/direction: {_compact_summary_value(action)}. "
        f"Contact point/region: {_compact_summary_value(contact)}. "
        f"Axis or orientation constraint: {_compact_summary_value(axis)}. "
        f"Precision risk: {_compact_summary_value(precision)}. "
        f"Transfer caution: {_compact_summary_value(caution)}."
    )


def _format_augmented_demo(
    rank,
    task_name,
    episode_id,
    retrieval_item,
    include_geometry,
    include_affordance,
    use_v2=False,
    include_scene_summary=False,
    include_trajectory=True,
    include_retrieval_metadata=True,
):
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
    ]
    if include_retrieval_metadata:
        lines.extend([
            (
            f"Retrieval scores: score={retrieval_item['score']:.4f}, "
            f"S_dyn={retrieval_item['s_dyn']:.4f}, "
            f"S_geo={retrieval_item['s_geo']:.4f}, "
            f"S_profile={retrieval_item['s_profile']:.4f}, "
            f"transfer_penalty={retrieval_item['penalty']:.4f}"
            ),
        ])
        if retrieval_item.get("compatibility_tier") not in {None, "not_plan_scored"}:
            cap = retrieval_item.get("score_cap")
            cap_text = "none" if cap is None else f"{cap:.4f}"
            lines.append(
                "Plan compatibility: "
                f"tier={retrieval_item.get('compatibility_tier')}, "
                f"S_plan={retrieval_item.get('s_plan', 0.0):.4f}, "
                f"raw_score={retrieval_item.get('raw_score', retrieval_item['score']):.4f}, "
                f"score_cap={cap_text}, "
                f"seen_family={retrieval_item.get('seen_family', 'unknown')}, "
                f"query_family={retrieval_item.get('query_family', 'unknown')}, "
                f"reason={retrieval_item.get('compatibility_reason', '')}"
            )
        lines.append("")
    else:
        lines.append("")
    if use_v2 and include_retrieval_metadata:
        lines.extend(
            [
                f"Attention bias: {retrieval_item.get('attention_bias', 1.0):.2f}",
                "Use this demo as a primary analogy only if its attention bias is high; use low-bias demos only as weak fallback context.",
                "",
                _format_profile_block("Precise interaction signature p_i", retrieval_item.get("seen_profile", {})),
                "",
            ]
        )
    seen_contact_hints = review.get("contact_hints_i") or review.get("affordance_a_i") or {}
    seen_geometry = _canonical_geometry(task_name, review.get("geometry_g_i") or {}, seen_contact_hints)
    if include_geometry:
        lines.extend([_format_feature_block("Primitive geometry/action descriptor g_i", seen_geometry, GEOMETRY_FIELDS), ""])
        seen_goal_state = retrieval_item.get("seen_goal_state") or _goal_state_descriptor(
            task_name,
            seen_geometry,
            seen_contact_hints,
            retrieval_item.get("seen_profile", {}),
            raw_goal=(
                review.get("goal_state_h_i")
                or review.get("target_pose_h_i")
                or review.get("target_pose_i")
                or review.get("target_pose_descriptor_i")
                or review.get("goal_state_descriptor_i")
            ),
            use_as="seen_demo_goal_state",
        )
        lines.extend([_format_feature_block("Goal-state/contact-pose descriptor h_i", seen_goal_state, GOAL_STATE_FIELDS), ""])
    if include_affordance:
        lines.extend([_format_feature_block("RoboPoint contact hints c_i", seen_contact_hints, CONTACT_HINT_FIELDS), ""])
    if include_scene_summary:
        lines.extend(
            [
                "Scene summary s_i:",
                _scene_summary(
                    demo["task_instruction"],
                    task_name,
                    seen_geometry,
                    seen_contact_hints,
                    retrieval_item.get("seen_profile", {}),
                ),
                "",
            ]
        )

    if not include_trajectory:
        return "\n".join(lines).rstrip()

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


def _v3_action_guidance(query_profile):
    family = _family_name(query_profile)
    common = [
        "V3 contact-mode guidance:",
        "- Output 3 to 6 key actions when the task requires grasp, move, align, lower, and release phases.",
        "- The 7th action value is the gripper state and must be binary: 1=open, 0=closed.",
    ]
    if family == "hole_over_vertical_stand":
        common.extend(
            [
                "- Required mechanics: grasp the roll body, lift it, align the central hole above the vertical stand or holder axis, lower along that vertical axis, then open/release.",
                "- Do not treat the stand base as the insertion target; target the vertical holder/peg axis.",
            ]
        )
    elif family == "object_into_shelf":
        common.extend(
            [
                "- Required mechanics: grasp the selected book body, lift it clear, move through the front shelf opening with enough clearance, keep the book oriented for the shelf, then release inside the shelf.",
                "- Do not copy cup-holder placement unless the action also enters the shelf opening.",
            ]
        )
    elif family == "round_object_into_open_goal":
        common.extend(
            [
                "- Required mechanics: grasp the ball, lift above the hoop, align the ball center over the hoop ring center, then open/release so the ball can pass through the ring.",
                "- Do not stack the ball on a rim or cup-like surface; the target is the open goal center.",
            ]
        )
    elif family == "tool_scoop_under_object":
        common.extend(
            [
                "- Required mechanics: grasp the spatula handle, keep the blade low, slide the flat blade under the cube, then lift while maintaining tool-object contact.",
                "- Do not grasp the cube directly and do not use a twisting or stacking action pattern.",
            ]
        )
    else:
        common.append("- Choose actions that match the query contact mode even if a high-scoring demo has a different action rhythm.")
    return common


def _format_v4_user_prompt(
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
    query_profile = _interaction_profile(query_task_key, query_geometry, query_affordance)
    query_goal_state = _goal_state_descriptor(
        query_task_key,
        query_geometry,
        query_affordance,
        query_profile,
        raw_goal=(
            query_geometry.get("goal_state_h_j")
            or query_geometry.get("target_pose_h_j")
            or query_geometry.get("target_pose_j")
            or query_geometry.get("target_pose")
            or query_geometry.get("goal_state_descriptor_j")
        ),
        use_as="query_goal_state",
    )
    query_scene_summary = _scene_summary(
        query_task_instruction,
        query_task_key,
        query_geometry,
        query_affordance,
        query_profile,
    )

    stage1 = [
        "<<<V4_STAGE1_PROMPT>>>",
        f"You will receive {len(ranked)} top-k retrieved seen demonstrations and one unseen query.",
        "",
        "Your job is to answer, in simple grounded words, what physical manipulation the Franka Panda robot should perform in the current unseen scene.",
        "Convert the unseen task into a clean semantic manipulation plan before any 7D actions are predicted.",
        "",
        "Use the retrieved seen demonstrations only as analogies for contact mode, target relation, motion direction, and gripper behavior. Do not copy their 7D coordinates in this stage.",
        "Their object identity and final goal may differ from the unseen query; prioritize the unseen current observation, primitive geometry/signature, and h_j goal-state/contact-pose descriptor when they conflict with a retrieved demo.",
        "",
        "Important rules:",
        "- Do not output 7D actions in Stage 1.",
        "- Do not use unseen future observations, unseen ground-truth actions, after-states, or unseen demonstrations.",
        "- Prefer descriptors and scene summaries that match the unseen contact mechanics over demos with superficially similar object names.",
        "- h_i is each seen demo's final success state; h_j is the unseen query's final success state. Use h_i only as an analogy and obey h_j.",
        "- Copy the target object's current coordinate exactly from the Current observation into target_current_coordinate.",
        "- Choose the active reference part from the Current observation, not a generic support object. For example, use holder/peg/post instead of stand_base when the task requires placing a hole over a stand.",
        "- Copy the active reference part's coordinate exactly from the Current observation into reference_coordinate.",
        "- Make action_primitive and motion_direction concrete, such as pull straight outward, rotate clockwise, lift then place, slide along +x, press downward, align over vertical post, or insert along socket axis.",
        "- Output exactly one compact JSON object with these fields: target_object, target_current_coordinate, reference_object, active_reference_part, reference_coordinate, target_location_relation, target_orientation, action_primitive, motion_direction, contact_point, gripper_plan, success_relation, constraints, demo_use_hint.",
        "",
        "Retrieved seen demonstrations summarized for semantic transfer:",
    ]
    if include_geometry:
        stage1.extend(["", _format_goal_state_field_guide(), ""])

    for rank, item in enumerate(ranked, start=1):
        selected_idx = item["index"]
        task_name, episode_id = _task_episode_from_path(all_demo_paths[selected_idx])
        stage1.extend(
            [
                _format_augmented_demo(
                    rank,
                    task_name,
                    episode_id,
                    item,
                    include_geometry,
                    include_affordance,
                    use_v2=True,
                    include_scene_summary=True,
                    include_trajectory=False,
                ),
                "",
            ]
        )

    stage1.extend(
        [
            "Unseen query for semantic planning:",
            "Task instruction:",
            query_task_instruction,
            "Task key:",
            query_task_key,
            "",
            "Current observation:",
            query_observation,
            "",
        ]
    )
    if include_geometry:
        stage1.extend([_format_feature_block("Primitive geometry/action descriptor g_j", query_geometry, GEOMETRY_FIELDS), ""])
        stage1.extend([_format_feature_block("Goal-state/contact-pose descriptor h_j", query_goal_state, GOAL_STATE_FIELDS), ""])
    if include_affordance:
        stage1.extend([_format_feature_block("RoboPoint contact hints c_j", query_affordance, CONTACT_HINT_FIELDS), ""])
    stage1.extend(
        [
            _format_profile_block("Precise interaction signature p_j", query_profile),
            "",
            "Scene summary s_j:",
            query_scene_summary,
            "",
            "Return only the Stage 1 semantic manipulation plan JSON:",
        ]
    )

    stage2 = [
        "<<<V4_STAGE2_CONTEXT>>>",
        f"You will now receive the same {len(ranked)} retrieved seen demonstrations with their paper-faithful per-key-action observation/action trajectories.",
        "",
        "Your job is to use the Stage 1 semantic manipulation plan as a clean intent bottleneck, then predict the unseen task's key 7D action sequence.",
        "Follow the VLA-style question directly: what action should the robot take to complete the task from the current observation?",
        "",
        "Important rules:",
        "- Each seen demonstration includes per-key-action observations paired with the corresponding 7D action.",
        "- The unseen query includes only the current/initial observation and task instruction; use the inserted Stage 1 plan for descriptor-derived intent.",
        "- Do not use unseen future observations, unseen ground-truth actions, after-states, or unseen demonstrations.",
        "- Adapt one or a few seen trajectory rhythms that match the Stage 1 plan; do not average together conflicting demos.",
        "- First write relative_action_sketch: short simple motion phrases such as approach target, close gripper, lift +z, move toward reference_coordinate, align with axis, lower/place, release, retreat.",
        "- The relative_action_sketch must be scene-relative and contact-aware. It should mention target/reference/axis directions, gripper open/close timing, and whether the task is pull, push, press, rotate, insert, place, slide, scoop, or lift.",
        "- Then use that relative_action_sketch to produce key_actions_7d. The 7D actions must be integer lists [x, y, z, roll, pitch, yaw, gripper]. The 7th value is binary: 1=open, 0=closed.",
        "- Do not put numeric 7D-like lists inside relative_action_sketch; put all final action numbers only in key_actions_7d.",
        "- Return only one compact JSON object with exactly these fields: relative_action_sketch, key_actions_7d.",
        "",
        "Retrieved seen demonstrations with key observation-action trajectories:",
    ]

    for rank, item in enumerate(ranked, start=1):
        selected_idx = item["index"]
        task_name, episode_id = _task_episode_from_path(all_demo_paths[selected_idx])
        stage2.extend(
            [
                _format_augmented_demo(
                    rank,
                    task_name,
                    episode_id,
                    item,
                    include_geometry=False,
                    include_affordance=False,
                    use_v2=False,
                    include_scene_summary=False,
                    include_trajectory=True,
                    include_retrieval_metadata=False,
                ),
                "",
            ]
        )

    stage2.extend(
        [
            "Unseen query:",
            "Task instruction:",
            query_task_instruction,
            "Task key:",
            query_task_key,
            "",
            "Current observation:",
            query_observation,
            "",
        ]
    )
    stage2.extend(
        [
            "<<<V4_STAGE2_PLAN_INSERT_HERE>>>",
            "The agent will insert the Stage 1 semantic manipulation plan here before asking for the relative sketch and final 7D actions.",
            "",
            "After reading the inserted semantic plan, return only this JSON shape:",
            '{"relative_action_sketch":["approach target ...","close gripper ...","move toward reference ..."],"key_actions_7d":[[x,y,z,roll,pitch,yaw,gripper],...]}',
        ]
    )
    return "\n".join(stage1 + [""] + stage2)


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
    use_v2=False,
    use_v3=False,
    use_v4=False,
    use_plan=False,
):
    if use_v4:
        return _format_v4_user_prompt(
            ranked,
            all_demo_paths,
            query_observation,
            query_task_key,
            query_task_instruction,
            query_geometry,
            query_affordance,
            include_geometry=include_geometry,
            include_affordance=include_affordance,
        )

    query_profile = _interaction_profile(query_task_key, query_geometry, query_affordance)
    query_goal_state = _goal_state_descriptor(
        query_task_key,
        query_geometry,
        query_affordance,
        query_profile,
        raw_goal=(
            query_geometry.get("goal_state_h_j")
            or query_geometry.get("target_pose_h_j")
            or query_geometry.get("target_pose_j")
            or query_geometry.get("target_pose")
            or query_geometry.get("goal_state_descriptor_j")
        ),
        use_as="query_goal_state",
    )
    lines = [
        f"You will receive {len(ranked)} top-k retrieved seen demonstrations from the AGNOSTOS seen-task training set. Use all of them as in-context examples for the current unseen query.",
        "",
        "The retrieved demonstrations were selected because they may share similar action primitive, motion, contact-region, or geometric constraint patterns. Their object identity and final goal may be different from the unseen query.",
        "Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to the retrieved seen demonstrations using action trends, primitive manipulation geometry, and the unseen goal-state/contact-pose descriptor.",
        "",
        "Important rules:",
        "- Each seen demonstration includes per-key-action observations paired with the corresponding 7D action.",
        "- Treat each retrieved seen demonstration as an analogy, not as the exact target. Transfer the action rhythm only when it is compatible with the unseen query.",
        "- The unseen query includes only the current/initial observation, task instruction, primitive geometry/action descriptor, and goal-state/contact-pose descriptor.",
        "- RoboPoint/oracle contact points, when present, are final-action hints only; they were not used to retrieve the demonstrations.",
        "- The unseen h_j descriptor is the desired final success state. If a seen h_i goal conflicts with h_j, follow h_j and use that seen demo only as weak motion evidence.",
        "- Do not use unseen demonstrations, unseen future frames, unseen ground-truth actions, or after-states.",
        "- Preserve the X-ICM output format: only a list of 7D action lists, such as [[x, y, z, roll, pitch, yaw, gripper], ...].",
    ]
    if include_geometry:
        lines.extend(["", _format_geometry_field_guide(), "", _format_goal_state_field_guide()])
    if include_affordance:
        lines.extend(["", _format_contact_field_guide()])
    if use_v2:
        lines.extend(
            [
                "- Each seen demonstration has an Attention bias from 0.00 to 1.00.",
                "- Treat demos with attention bias >= 0.75 as primary analogies, demos from 0.40 to 0.75 as supporting evidence, and demos below 0.40 as weak fallback context.",
                "- If a low-bias demo conflicts with the unseen query or a high-bias demo, ignore the low-bias action trend.",
                "- Use the precise interaction signatures to distinguish similar words with different mechanics, such as docking vs shape insertion, hinged pushing vs slot insertion, and pulling out vs putting in.",
            ]
        )
    if use_v3:
        lines.extend(
            [
                "- For v3, prioritize contact-mode compatibility over raw visual/dynamic similarity when they disagree.",
                "- Prefer demos whose interaction family, target relation, required contact region, and axis constraint match the unseen query.",
                "- If retrieved demos repeat the same wrong contact mode, use them only as weak coordinate hints.",
            ]
        )
    if use_plan:
        lines.extend(
            [
                "- Plan compatibility was applied during retrieval. Exact and near demos are primary analogies; weak demos are fallback evidence only.",
                "- If a demo is marked blocked, do not copy its action rhythm or final relation. Use it only for generic coordinate scale if no better demo covers that detail.",
                "- The query interaction family and h_j goal-state/contact-pose descriptor are authoritative when any retrieved h_i conflicts with them.",
            ]
        )
    lines.append("")

    for rank, item in enumerate(ranked, start=1):
        selected_idx = item["index"]
        task_name, episode_id = _task_episode_from_path(all_demo_paths[selected_idx])
        lines.extend([
            _format_augmented_demo(
                rank,
                task_name,
                episode_id,
                item,
                include_geometry,
                include_affordance,
                use_v2=use_v2,
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
        lines.extend([_format_feature_block("Primitive geometry/action descriptor g_j", query_geometry, GEOMETRY_FIELDS), ""])
        lines.extend([_format_feature_block("Goal-state/contact-pose descriptor h_j", query_goal_state, GOAL_STATE_FIELDS), ""])
    if include_affordance:
        lines.extend([_format_feature_block("RoboPoint contact hints c_j", query_affordance, CONTACT_HINT_FIELDS), ""])
    if use_v2:
        lines.extend([_format_profile_block("Precise interaction signature p_j", query_profile), ""])
    if use_v3:
        lines.extend(_v3_action_guidance(query_profile))
        lines.append("")
    lines.append("Predict the key 7D action sequence for the unseen task. Return only a Python-style list of 7D action lists:")
    return "\n".join(lines)




if __name__ == "__main__":
    pass
