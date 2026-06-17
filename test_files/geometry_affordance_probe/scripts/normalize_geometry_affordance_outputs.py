import argparse
import json
import re
from pathlib import Path


def unique(items):
    seen = set()
    out = []
    for item in items:
        item = str(item).strip().lower().replace(" ", "_")
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_points(text):
    if not text:
        return []
    points = []
    for x, y in re.findall(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text):
        try:
            points.append([float(x), float(y)])
        except ValueError:
            pass
    return points


def infer_geometry_key_features(task, geometry):
    text = " ".join([
        task or "",
        json.dumps(geometry or {}, sort_keys=True),
    ]).lower()
    features = []

    vocab = {
        "round": ["round", "circular", "cylinder", "cylindrical", "sphere", "radial", "bulb", "jar"],
        "square": ["square"],
        "rectangular": ["rectangular", "box", "drawer", "safe", "cupboard"],
        "flat": ["flat", "panel", "thin"],
        "elongated": ["elongated", "long", "handle", "tap"],
        "hollow": ["hollow", "opening", "container", "inside", "drawer", "cupboard", "jar", "safe", "dustpan", "hole", "slot"],
        "solid": ["solid", "block"],
        "handle": ["handle", "tap", "drawer"],
        "knob": ["knob", "button", "tap"],
        "lid": ["lid", "jar"],
        "rim": ["rim", "jar", "cup", "opening"],
        "hole": ["hole", "socket", "peg"],
        "slot": ["slot", "shape_sorter", "safe"],
        "hinge": ["hinge"],
        "sliding_axis": ["sliding_axis", "drawer", "slide"],
        "rotational_axis": ["axis", "rotate", "turn", "twist", "tap"],
        "open_container": ["open", "drawer", "cupboard", "dustpan"],
        "target_region": ["target", "button", "peg", "sorter"],
        "alignment_sensitive": ["insert", "peg", "socket", "shape_sorter", "light_bulb"],
    }
    for feature, cues in vocab.items():
        if any(cue in text for cue in cues):
            features.append(feature)

    task_specific = {
        "close_jar": ["jar", "round", "cylindrical", "hollow", "lid", "rim", "top_opening"],
        "insert_onto_square_peg": ["peg", "square_peg", "hole", "alignment_sensitive", "insertable_part"],
        "light_bulb_in": ["light_bulb", "round", "bulb", "threaded_base", "socket", "rotational_axis", "alignment_sensitive"],
        "slide_block_to_color_target": ["block", "rectangular", "flat_faces", "solid", "target_region"],
        "sweep_to_dustpan_of_size": ["dustpan", "open_container", "thin_edge", "flat_floor", "target_region"],
        "push_buttons": ["button", "round", "small", "raised_surface", "flat_top"],
        "put_groceries_in_cupboard": ["cupboard", "box_like", "hollow", "shelf", "open_container"],
        "put_money_in_safe": ["safe", "rectangular", "hollow", "slot", "front_opening"],
        "place_shape_in_shape_sorter": ["shape_sorter", "shape_profile", "slot", "hole", "matching_geometry", "alignment_sensitive"],
        "put_item_in_drawer": ["drawer", "rectangular", "hollow", "open_container", "sliding_axis", "handle"],
        "open_drawer": ["drawer", "rectangular", "hollow", "handle", "sliding_axis", "front_face"],
        "turn_tap": ["tap", "handle", "knob", "rotational_axis", "cylindrical"],
    }
    task_name = (task or "").lower()
    for key, values in task_specific.items():
        if key in task_name or key.replace("_", " ") in text:
            return unique(values)

    blocked = {
        "robot", "arm", "arms", "head", "gripper", "left_arm", "right_arm",
        "left_arm_extended", "right_arm_bent", "red", "green", "blue",
    }
    return [feature for feature in unique(features) if feature not in blocked]


def infer_manipulated_object(task):
    task = (task or "").lower()
    objects = [
        ("turn_tap", "tap"),
        ("close_jar", "jar"),
        ("light_bulb_in", "light_bulb"),
        ("slide_block_to_color_target", "block"),
        ("sweep_to_dustpan_of_size", "dustpan"),
        ("push_buttons", "button"),
        ("put_groceries_in_cupboard", "groceries_and_cupboard"),
        ("put_money_in_safe", "money_and_safe"),
        ("place_shape_in_shape_sorter", "shape_and_shape_sorter"),
        ("put_item_in_drawer", "item_and_drawer"),
        ("insert_onto_square_peg", "shape_and_square_peg"),
        ("open_drawer", "drawer"),
    ]
    for key, value in objects:
        if key in task or key.replace("_", " ") in task:
            return value
    return "unknown"


def infer_affordance(task, points, raw_text):
    task = (task or "").lower()
    spec = {
        "grasp_affordance": "unknown",
        "contact_affordance": "unknown",
        "motion_affordance": "unknown",
        "support_affordance": "none",
        "containment_affordance": "none",
        "articulation_affordance": "none",
        "required_contact_region": "unknown",
        "preferred_contact_points": points,
        "precision_requirement": "medium",
        "force_requirement": "medium",
        "failure_sensitive_property": "wrong_contact_point",
        "raw_robopoint_text": raw_text,
    }

    rules = [
        ("turn_tap", {
            "grasp_affordance": "knob_grasp",
            "contact_affordance": "rotate_part",
            "motion_affordance": "rotate",
            "articulation_affordance": "screw_twist",
            "required_contact_region": "tap_handle_or_knob",
            "failure_sensitive_property": "wrong_axis",
        }),
        ("close_jar", {
            "grasp_affordance": "rim_grasp",
            "contact_affordance": "rotate_part",
            "motion_affordance": "twist",
            "containment_affordance": "closed_container",
            "articulation_affordance": "screw_twist",
            "required_contact_region": "lid_or_rim",
            "failure_sensitive_property": "wrong_axis",
        }),
        ("light_bulb_in", {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert_then_twist",
            "articulation_affordance": "screw_twist",
            "required_contact_region": "bulb_body_or_base",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        }),
        ("slide_block_to_color_target", {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "push_surface",
            "motion_affordance": "slide",
            "required_contact_region": "block_side_or_top",
            "failure_sensitive_property": "overshoot_or_wrong_direction",
        }),
        ("sweep_to_dustpan_of_size", {
            "grasp_affordance": "handle_grasp",
            "contact_affordance": "push_surface",
            "motion_affordance": "push",
            "containment_affordance": "receptacle",
            "required_contact_region": "sweeping_tool_or_object_side",
            "failure_sensitive_property": "missed_receptacle",
        }),
        ("push_buttons", {
            "grasp_affordance": "none",
            "contact_affordance": "push_surface",
            "motion_affordance": "push",
            "required_contact_region": "button_top",
            "precision_requirement": "high",
            "force_requirement": "low",
            "failure_sensitive_property": "wrong_button",
        }),
        ("put_groceries_in_cupboard", {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "lift_then_insert",
            "support_affordance": "can_support",
            "containment_affordance": "receptacle",
            "required_contact_region": "object_body",
            "failure_sensitive_property": "collision_with_shelf",
        }),
        ("put_money_in_safe", {
            "grasp_affordance": "pinch",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert",
            "containment_affordance": "slot",
            "required_contact_region": "thin_object_edge",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        }),
        ("place_shape_in_shape_sorter", {
            "grasp_affordance": "edge_grasp",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert",
            "containment_affordance": "slot",
            "required_contact_region": "shape_body",
            "precision_requirement": "high",
            "failure_sensitive_property": "wrong_shape_orientation",
        }),
        ("put_item_in_drawer", {
            "grasp_affordance": "body_grasp",
            "contact_affordance": "lift_and_place",
            "motion_affordance": "lift_then_place",
            "containment_affordance": "open_container",
            "articulation_affordance": "drawer_slide",
            "required_contact_region": "object_body",
            "failure_sensitive_property": "collision_with_drawer",
        }),
        ("insert_onto_square_peg", {
            "grasp_affordance": "edge_grasp",
            "contact_affordance": "insert_object",
            "motion_affordance": "insert",
            "containment_affordance": "hole",
            "required_contact_region": "object_body",
            "precision_requirement": "high",
            "failure_sensitive_property": "misalignment",
        }),
        ("open_drawer", {
            "grasp_affordance": "handle_grasp",
            "contact_affordance": "pull_handle",
            "motion_affordance": "pull",
            "containment_affordance": "open_container",
            "articulation_affordance": "drawer_slide",
            "required_contact_region": "handle",
            "failure_sensitive_property": "wrong_grasp",
        }),
    ]
    for key, values in rules:
        if key in task or key.replace("_", " ") in task:
            spec.update(values)
            break
    spec["source_note"] = "RoboPoint produced the contact/keypoint coordinates; symbolic affordance fields are normalized from the task instruction so the retrieval file has concrete labels instead of placeholders."
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/yf23/projects/ICRA27-ROBOT/experiments/geometry_affordance_probe/results")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    for demo in manifest.get("selected", []):
        demo_dir = Path(demo["demo_dir"])
        task = demo.get("task") or ""

        geometry_path = demo_dir / "geometry_qwen2_5_vl.json"
        if geometry_path.exists():
            geometry_doc = json.loads(geometry_path.read_text())
            geometry = geometry_doc.setdefault("geometry_g_i", {})
            geometry["manipulated_object"] = infer_manipulated_object(task)
            geometry["key_features"] = infer_geometry_key_features(task, geometry)
            geometry["key_features_note"] = "Compact retrieval-oriented features normalized from the Qwen geometry output and task instruction."
            geometry_path.write_text(json.dumps(geometry_doc, indent=2))

        affordance_path = demo_dir / "affordance_robopoint.json"
        if affordance_path.exists():
            affordance_doc = json.loads(affordance_path.read_text())
            old = affordance_doc.get("affordance_a_i", {})
            raw_text = old.get("raw_robopoint_text")
            points = parse_points(raw_text)
            vision_tower_note = old.get("vision_tower_note")
            affordance_doc["affordance_a_i"] = infer_affordance(task, points, raw_text)
            if vision_tower_note:
                affordance_doc["affordance_a_i"]["vision_tower_note"] = vision_tower_note
            affordance_path.write_text(json.dumps(affordance_doc, indent=2))
            print("normalized", demo["id"])


if __name__ == "__main__":
    main()
