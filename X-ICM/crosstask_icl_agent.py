from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
import shutil
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, quaternion_to_discrete_euler, CAMERAS
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from vllm import LLM, SamplingParams 
from qwen_vl_utils import process_vision_info


class CrossTaskICLAgent(Agent):
    def __init__(self, task_name, demo_num_per_icl=10, seed=0, ranking_method="lang_vis.out"):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.demo_num_per_icl = demo_num_per_icl
        self.front_rgb_path=None
        self.vl_front_rgb_path=None
        self.vl_overhead_rgb_path=None
        self.seed=seed
        self.ranking_method=ranking_method
        self.phase_query_affordance = {}
        self.phase_query_profile = {}
        self.phase_query_geometry = {}

        if self._is_phase_model():
            self.SYSTEM_PROMPT = (
                "You are a Franka Panda robot with a parallel gripper. "
                "You are running Action-Chain X-ICM phase mode. "
                "You will receive the current unseen task observation, an explicit action chain, phase memory, "
                "the current phase to execute, role-labeled current-scene contact/target anchors, and retrieved seen phase examples. "
                "Predict actions only for the current phase. Do not solve the whole task at once and do not repeat completed phases unless the prompt says the previous phase failed. "
                "Return only one compact JSON object with fields current_phase and phase_actions_7d. "
                "phase_actions_7d must be a short mini-trajectory: a list of [x,y,z,roll,pitch,yaw,gripper] integer lists."
            )
        elif any(tag in ranking_method for tag in ["geo_aff", ".geo", ".aff"]):
            self.SYSTEM_PROMPT = (
                "You are a Franka Panda robot with a parallel gripper. "
                "You will receive the top-k retrieved in-context demonstrations from seen robot manipulation tasks. "
                "Each seen demonstration contains a task instruction, per-key-action observations, the corresponding 7D actions, "
                "and optional geometry/affordance descriptions depending on the ablation. "
                "You will then receive one unseen query with only its current observation, task instruction, and the same descriptor types. "
                "The retrieved demonstrations are action, contact, motion, or geometry analogies; their object identity and final goal may differ from the unseen query. "
                "Use the unseen query descriptors, especially its goal-state/contact-pose descriptor when present, as the desired success state. "
                "Your job is to infer the unseen task's key 7D action sequence by comparing the current unseen scene to compatible retrieved seen demonstrations. "
                "Do not use future observations, after-states, unseen demonstrations, or ground-truth unseen actions. "
                "Return only a Python-style list of 7D action lists. Do not output anything else."
            )
            if self._is_closed_loop():
                self.SYSTEM_PROMPT += (
                    " In closed-loop mode, predict only the next useful primitive action for the current observation. "
                    "After that primitive executes, you will observe the scene again and correct the next action."
                )
        else:
            self.SYSTEM_PROMPT = "You are a Franka Panda robot with a parallel gripper. We provide you with some demos from some seen tasks, in the format of [task_instruction, observation]>[ 7-dim action_1, 7-dim action_2, ..., 7-dim action_N ]. Then you will receive an unseen task instruction with a new observation, and you need to output a list of 7-dim actions that match the trends in the demos. Do not output anything else."


    def _is_v4(self):
        return "v4" in self.ranking_method

    def _is_closed_loop(self):
        normalized = self.ranking_method.replace("-", "_")
        return "closed_loop" in normalized or ".cl" in normalized

    def _is_phase_model(self):
        normalized = self.ranking_method.replace("-", "_").lower()
        tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
        return "phase" in tokens or "phases" in tokens or "action_chain" in normalized or "actionchain" in normalized

    def _phase_chain_max_steps(self):
        return max(1, int(os.environ.get("XICM_PHASE_MAX_PROMPTS", "8")))

    def _phase_terminal_retract_delta(self):
        return max(0, int(os.environ.get("XICM_PHASE_TERMINAL_RETRACT_Z", "8")))

    def _phase_terminal_max_z(self):
        return int(np.clip(int(os.environ.get("XICM_PHASE_TERMINAL_MAX_Z", "75")), 0, 99))

    def _phase_tool_surface_min_z(self):
        return int(np.clip(int(os.environ.get("XICM_PHASE_TOOL_SURFACE_MIN_Z", "15")), 0, 99))

    def _phase_approach_clearance_z(self):
        return max(0, int(os.environ.get("XICM_PHASE_APPROACH_CLEARANCE_Z", "12")))

    def _phase_transport_clearance_z(self):
        return max(0, int(os.environ.get("XICM_PHASE_TRANSPORT_CLEARANCE_Z", "12")))

    def _phase_verify_dist(self):
        return max(0.0, float(os.environ.get("XICM_PHASE_VERIFY_DIST", "13")))

    def _phase_verify_xy_dist(self):
        return max(0.0, float(os.environ.get("XICM_PHASE_VERIFY_XY_DIST", "18")))

    def _phase_verify_strict_gripper(self):
        value = os.environ.get("XICM_PHASE_VERIFY_STRICT_GRIPPER", "0").strip().lower()
        return value not in {"0", "false", "off", "no"}

    def _phase_verify_min_motion(self):
        return max(0.0, float(os.environ.get("XICM_PHASE_VERIFY_MIN_MOTION", "8")))

    def _phase_verify_min_lift_z(self):
        return max(0.0, float(os.environ.get("XICM_PHASE_VERIFY_MIN_LIFT_Z", "10")))

    def _phase_verify_min_rot_delta(self):
        return max(0.0, float(os.environ.get("XICM_PHASE_VERIFY_MIN_ROT_DELTA", "8")))

    def _closed_loop_max_replans(self):
        return max(1, int(os.environ.get("XICM_CLOSED_LOOP_MAX_REPLANS", "4")))

    def _closed_loop_should_replan(self):
        return self._is_closed_loop() and self.closed_loop_replans < self._closed_loop_max_replans()

    def _low_dim_state_summary(self, observation):
        state = observation.get("low_dim_state")
        if state is None:
            return "unknown"
        try:
            values = state.squeeze().detach().cpu().numpy().astype(float).tolist()
        except Exception:
            try:
                values = np.array(state).squeeze().astype(float).tolist()
            except Exception:
                return "unknown"
        if not isinstance(values, list) or len(values) < 22:
            return "unknown"
        gripper_open = values[14]
        gripper_pose = values[15:22]
        pose_text = ", ".join(f"{value:.4f}" for value in gripper_pose)
        return f"gripper_open={gripper_open:.3f}; gripper_pose_xyzquat=[{pose_text}]"

    def _closed_loop_prompt_suffix(self, step, observation):
        history = self.closed_loop_history[-6:]
        history_lines = [
            f"- step {item['step']}: action_7d={item['action_7d']}"
            for item in history
        ] or ["- none"]
        return "\n".join(
            [
                "",
                "Closed-loop execution mode:",
                f"- Current environment step: {step}",
                f"- Current robot state: {self._low_dim_state_summary(observation)}",
                "- Previously executed primitive actions in this episode:",
                *history_lines,
                "- Re-observe the current scene, infer the current subgoal, and output only the next primitive 7D action.",
                "- The next action should make immediate progress from the current state, not replay the whole original plan.",
                "- If the object is already grasped, do not predict another grasp; move toward the target relation.",
                "- If the object is already at the target relation, output a small release/retract or no-op-like finishing primitive.",
                "- Return only one compact JSON object with fields current_subgoal and next_action_7d, where next_action_7d is [x,y,z,roll,pitch,yaw,gripper].",
            ]
        )

    def _continuous_action_to_discrete(self, continuous_action):
        action = np.asarray(continuous_action, dtype=float)
        bounds = SCENE_BOUNDS
        res = (bounds[3:] - bounds[:3]) / 100
        trans = np.floor((action[:3] - bounds[:3]) / res).astype(int)
        trans = np.clip(trans, 0, 99).tolist()
        try:
            rot = quaternion_to_discrete_euler(action[3:7]).astype(int).tolist()
        except Exception:
            rot = [0, 0, 0]
        grip_index = 7 if len(action) > 7 else 6
        gripper = int(round(float(action[grip_index]))) if len(action) > grip_index else 1
        return [*trans, *rot, gripper]

    def _phase_budget_fallback_action(self):
        primitive = str((self.current_phase or {}).get("primitive", "")).lower()
        if self._is_phase_model() and self.phase_prompts_used >= self._phase_chain_max_steps():
            open_gripper = primitive in {"place_or_release", "release_or_stop", "release_or_retract"}
            return self._phase_terminal_action(open_gripper=open_gripper, retract=False)
        if self.phase_history:
            last_action = self.phase_history[-1].get("action_7d")
            if isinstance(last_action, list) and len(last_action) == 7:
                return [int(round(float(value))) for value in last_action]
        if getattr(self, "_last_discrete_actions", None):
            last_action = self._last_discrete_actions[-1]
            if isinstance(last_action, list) and len(last_action) == 7:
                return [int(round(float(value))) for value in last_action]
        return [50, 50, 50, 0, 36, 50, 1]

    def _phase_last_discrete_action(self):
        if self.phase_history:
            for item in reversed(self.phase_history):
                last_action = item.get("action_7d")
                if isinstance(last_action, list) and len(last_action) == 7:
                    return self._clip_discrete_action(last_action)
        if getattr(self, "_last_discrete_actions", None):
            for last_action in reversed(self._last_discrete_actions):
                if isinstance(last_action, list) and len(last_action) == 7:
                    return self._clip_discrete_action(last_action)
        return None

    def _phase_last_action_for_primitives(self, primitives):
        wanted = {str(primitive).lower() for primitive in primitives}
        for item in reversed(getattr(self, "phase_history", []) or []):
            primitive = str(item.get("primitive", "")).lower()
            last_action = item.get("action_7d")
            if primitive in wanted and isinstance(last_action, list) and len(last_action) == 7:
                return self._clip_discrete_action(last_action)
        return None

    def _phase_terminal_action(self, open_gripper=False, retract=True):
        last_action = self._phase_last_discrete_action()
        if last_action is None:
            last_action = [50, 50, 50, 0, 36, 50, 1]
        action = self._clip_discrete_action(last_action)
        if retract:
            max_z = self._phase_terminal_max_z()
            next_z = min(max_z, action[2] + self._phase_terminal_retract_delta())
            action[2] = int(np.clip(max(action[2], next_z), 0, 99))
        if open_gripper:
            action[6] = 1
        return self._clip_discrete_action(action)

    def _phase_anchor_guard_enabled(self):
        value = os.environ.get("XICM_PHASE_ANCHOR_GUARD", "1").strip().lower()
        return value not in {"0", "false", "off", "no"}

    def _phase_role_points(self):
        affordance = getattr(self, "phase_query_affordance", {}) or {}
        points = affordance.get("role_labeled_points") or []
        lookup = {}
        if not isinstance(points, list):
            return lookup
        for point in points:
            if not isinstance(point, dict):
                continue
            role = self._phase_canonical_role(point.get("role", ""))
            voxel = point.get("voxel_xyz")
            if role and role not in lookup and isinstance(voxel, list) and len(voxel) >= 3:
                lookup[role] = [int(round(float(value))) for value in voxel[:3]]
        memory = getattr(self, "phase_anchor_memory", {}) or {}
        primitive = str((self.current_phase or {}).get("primitive", "")).lower()
        for role in ("goal_region", "constraint_reference"):
            if role in memory:
                lookup[role] = list(memory[role])
        if primitive in {"approach", "grasp", "grasp_or_contact", "grasp_or_contact_tool"}:
            for role in ("manipulated_object_contact", "tool_working_edge", "secondary_object_to_move"):
                if role in memory:
                    lookup[role] = list(memory[role])
        return lookup

    def _phase_canonical_role(self, role):
        role = str(role).lower()
        if role == "constraint_region":
            return "constraint_reference"
        return role

    def _phase_update_anchor_memory(self):
        points = self._phase_role_points_from_affordance(getattr(self, "phase_query_affordance", {}) or {})
        if not points:
            return
        if not hasattr(self, "phase_anchor_memory"):
            self.phase_anchor_memory = {}
        primitive = str((self.current_phase or {}).get("primitive", "")).lower()
        static_roles = {"goal_region", "constraint_reference"}
        pregrasp_roles = {"manipulated_object_contact", "tool_working_edge", "secondary_object_to_move"}
        for role, voxel in points.items():
            if role in static_roles:
                self.phase_anchor_memory.setdefault(role, list(voxel))
                continue
            if role in pregrasp_roles:
                if role not in self.phase_anchor_memory:
                    self.phase_anchor_memory[role] = list(voxel)
                elif primitive not in {"approach", "grasp", "grasp_or_contact", "grasp_or_contact_tool"}:
                    self.phase_anchor_memory[role] = list(voxel)

    def _phase_role_points_from_affordance(self, affordance):
        points = affordance.get("role_labeled_points") or []
        lookup = {}
        if not isinstance(points, list):
            return lookup
        for point in points:
            if not isinstance(point, dict):
                continue
            role = self._phase_canonical_role(point.get("role", ""))
            voxel = point.get("voxel_xyz")
            if role and role not in lookup and isinstance(voxel, list) and len(voxel) >= 3:
                lookup[role] = [int(round(float(value))) for value in voxel[:3]]
        return lookup

    def _phase_current_gripper_state(self, observation):
        state = observation.get("low_dim_state") if isinstance(observation, dict) else None
        if state is None:
            return {}
        try:
            values = state.squeeze().detach().cpu().numpy().astype(float).tolist()
        except Exception:
            try:
                values = np.array(state).squeeze().astype(float).tolist()
            except Exception:
                return {}
        if not isinstance(values, list) or len(values) < 22:
            return {}
        pose = values[15:22]
        try:
            discrete = self._continuous_action_to_discrete([*pose, values[14]])
        except Exception:
            discrete = None
        return {
            "open": float(values[14]),
            "pose": pose,
            "discrete": discrete,
        }

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def _phase_trace_root(self):
        root = os.environ.get("XICM_PHASE_REVIEW_TRACE_DIR", "").strip()
        if not root or not self._is_phase_model():
            return None
        return root

    def _safe_trace_label(self, value):
        label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")
        return label[:80] or "unknown"

    def _write_phase_trace_json(self, path, payload):
        with open(path, "w") as handle:
            json.dump(self._json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _write_phase_prompt_trace(self, step, user_prompt, messages, output_text):
        root = self._phase_trace_root()
        if not root:
            return
        try:
            current_phase = self.current_phase or {}
            primitive = self._safe_trace_label(current_phase.get("primitive", "unknown"))
            trace_step = max(int(step), int(getattr(self, "step", step)), 0)
            task_dir = self._safe_trace_label(self.task_name)
            episode_dir = os.path.join(root, task_dir, f"episode_{self.episode_id:03d}")
            prompt_dir = os.path.join(
                episode_dir,
                f"step_{trace_step:03d}_phase_{int(self.phase_index):02d}_{primitive}",
            )
            os.makedirs(prompt_dir, exist_ok=True)

            for label, path in [
                ("front", self.vl_front_rgb_path),
                ("overhead", self.vl_overhead_rgb_path),
            ]:
                if path and os.path.exists(path):
                    shutil.copy2(path, os.path.join(prompt_dir, f"query_{label}.png"))

            with open(os.path.join(prompt_dir, "system_prompt.txt"), "w") as handle:
                handle.write(self.SYSTEM_PROMPT.rstrip() + "\n")
            with open(os.path.join(prompt_dir, "user_prompt.txt"), "w") as handle:
                handle.write(str(user_prompt).rstrip() + "\n")
            with open(os.path.join(prompt_dir, "model_output_raw.txt"), "w") as handle:
                handle.write(str(output_text).rstrip() + "\n")

            self._write_phase_trace_json(
                os.path.join(prompt_dir, "module_state.json"),
                {
                    "task_name": self.task_name,
                    "episode_id": self.episode_id,
                    "step": trace_step,
                    "raw_runner_step": int(step),
                    "phase_index": self.phase_index,
                    "current_phase": current_phase,
                    "completed_phases": self.completed_phases,
                    "pending_phase_completion": getattr(self, "pending_phase_completion", None),
                    "phase_history": self.phase_history,
                    "phase_chain": self.phase_chain,
                    "query_geometry_g_j": self.phase_query_geometry,
                    "query_goal_or_contact_hints_c_j": self.phase_query_affordance,
                    "query_interaction_profile_p_j": self.phase_query_profile,
                    "ranked_phase_items": getattr(self.handler, "last_ranked_phase_items", []),
                    "messages_preview": messages,
                    "query_images": {
                        "front": self.vl_front_rgb_path,
                        "overhead": self.vl_overhead_rgb_path,
                    },
                },
            )
            self._last_phase_trace_dir = prompt_dir
        except Exception as exc:
            print("Phase prompt trace write failed:", exc)

    def _write_phase_parsed_action_trace(self, actions):
        root = self._phase_trace_root()
        if not root:
            return
        prompt_dir = getattr(self, "_last_phase_trace_dir", None)
        if not prompt_dir:
            return
        try:
            self._write_phase_trace_json(
                os.path.join(prompt_dir, "parsed_actions.json"),
                {
                    "phase_index": self.phase_index,
                    "current_phase": self.current_phase,
                    "parsed_actions_7d": actions,
                },
            )
        except Exception as exc:
            print("Phase parsed-action trace write failed:", exc)

    def _phase_refresh_query_state(self, obs, **kwargs):
        if not self._is_phase_model():
            return
        try:
            mask_id_to_sim_name = {}
            mask_dict = {}
            point_cloud_dict = {}
            lang_goal = kwargs["lang_goal"]

            front_rgb_img = obs["front_rgb"]
            front_rgb_img = front_rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            front_rgb_img = np.clip((front_rgb_img).astype(np.uint8), 0, 255)
            front_rgb_dir = os.path.join(self.savedir, "rgb_dir", "front", str(self.episode_id))
            os.makedirs(front_rgb_dir, exist_ok=True)
            self.front_rgb_path = os.path.join(front_rgb_dir, "rgb.png")
            Image.fromarray(front_rgb_img).save(self.front_rgb_path)

            for camera in CAMERAS:
                if camera in {"front", "overhead"}:
                    rgb_img = obs[f"{camera}_rgb"]
                    rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
                    rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)
                    query_view_dir = os.path.join(self.savedir, "rgb_dir", "query_views", str(self.episode_id))
                    os.makedirs(query_view_dir, exist_ok=True)
                    query_view_path = os.path.join(query_view_dir, f"{camera}.png")
                    Image.fromarray(rgb_img).save(query_view_path)
                    if camera == "front":
                        self.vl_front_rgb_path = query_view_path
                    elif camera == "overhead":
                        self.vl_overhead_rgb_path = query_view_path

                mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])
                mask_dict[camera] = obs[f"{camera}_mask"].squeeze().cpu().numpy()
                point_cloud_dict[camera] = obs[f"{camera}_point_cloud"].cpu().squeeze().permute(1, 2, 0).numpy()

            self.handler.get_user_prompt_ranking(
                mask_dict,
                mask_id_to_sim_name,
                point_cloud_dict,
                custom_num_demos=self.demo_num_per_icl,
                taskname=lang_goal,
                image_path=self.front_rgb_path,
                seed=self.seed,
                ranking_metric=self.ranking_method,
                current_phase_index=self.phase_index,
                completed_phases=self.completed_phases,
                phase_history=self.phase_history,
            )
            self.phase_chain = getattr(self.handler, "last_phase_chain", self.phase_chain)
            self.current_phase = getattr(self.handler, "last_current_phase", self.current_phase)
            self.phase_query_affordance = getattr(self.handler, "last_query_affordance", self.phase_query_affordance)
            self.phase_query_profile = getattr(self.handler, "last_query_profile", self.phase_query_profile)
            self.phase_query_geometry = getattr(self.handler, "last_query_geometry", self.phase_query_geometry)
            self._phase_update_anchor_memory()
        except Exception as exc:
            print("Phase query-state refresh failed:", exc)

    def _phase_dist(self, a, b, dims=3):
        if not a or not b:
            return None
        try:
            dims = int(np.clip(dims, 1, 3))
            return float(np.linalg.norm(np.array(a[:dims], dtype=float) - np.array(b[:dims], dtype=float)))
        except Exception:
            return None

    def _phase_actions_for_phase(self, phase_id=None, primitive=None):
        actions = []
        primitive = str(primitive or "").lower()
        for item in getattr(self, "phase_history", []) or []:
            if phase_id is not None and item.get("phase_id") != phase_id:
                continue
            if primitive and str(item.get("primitive", "")).lower() != primitive:
                continue
            action = item.get("action_7d")
            if isinstance(action, list) and len(action) == 7:
                actions.append(self._clip_discrete_action(action))
        return actions

    def _phase_circular_delta(self, a, b, period=72):
        diff = abs(float(a) - float(b)) % period
        return min(diff, period - diff)

    def _phase_command_motion_stats(self, phase_id=None, primitive=None):
        actions = self._phase_actions_for_phase(phase_id=phase_id, primitive=primitive)
        if not actions and getattr(self, "_last_discrete_actions", None):
            actions = [
                self._clip_discrete_action(action)
                for action in self._last_discrete_actions
                if isinstance(action, list) and len(action) == 7
            ]
        if not actions:
            return {}
        xyz = np.array([action[:3] for action in actions], dtype=float)
        rot = np.array([action[3:6] for action in actions], dtype=float)
        first_xyz = xyz[0]
        last_xyz = xyz[-1]
        final_gripper = actions[-1][6]
        rot_delta = 0.0
        if len(rot) > 1:
            for row in rot[1:]:
                rot_delta = max(
                    rot_delta,
                    max(self._phase_circular_delta(row[index], rot[0][index]) for index in range(3)),
                )
        return {
            "count": len(actions),
            "first": actions[0],
            "last": actions[-1],
            "xyz_delta": float(np.linalg.norm(last_xyz - first_xyz)),
            "xy_delta": float(np.linalg.norm(last_xyz[:2] - first_xyz[:2])),
            "z_delta": float(np.max(xyz[:, 2]) - np.min(xyz[:, 2])),
            "final_z_delta": float(last_xyz[2] - first_xyz[2]),
            "rot_delta": float(rot_delta),
            "final_gripper_open": final_gripper >= 1,
            "final_gripper_closed": final_gripper < 1,
            "any_gripper_open": any(action[6] >= 1 for action in actions),
            "any_gripper_closed": any(action[6] < 1 for action in actions),
        }

    def _phase_anchor_candidates(self, primitive, raw_points=None):
        candidates = []

        def add(point):
            if isinstance(point, list) and len(point) >= 3 and point not in candidates:
                candidates.append(point)

        add(self._phase_relevant_anchor(primitive))
        raw_points = raw_points or {}
        primitive = str(primitive).lower()
        if primitive in {"move_or_align", "place_or_release", "insert", "lower"}:
            add(raw_points.get("goal_region"))
            add(raw_points.get("constraint_reference"))
        elif primitive in {
            "slide_or_sweep",
            "pull_or_drag",
            "pull_or_extract",
            "push",
            "press",
            "push_or_rotate_hinge",
            "pull_open_hinge",
            "rotate",
            "tilt_or_pour",
            "untilt_or_stop",
        }:
            add(raw_points.get("secondary_object_to_move"))
            add(raw_points.get("goal_region"))
            add(raw_points.get("constraint_reference"))
            add(raw_points.get("manipulated_object_contact"))
        else:
            add(raw_points.get("manipulated_object_contact"))
            add(raw_points.get("tool_working_edge"))
        return candidates

    def _phase_best_anchor_distance(self, reference, candidates, dims):
        best_dist = None
        best_anchor = candidates[0] if candidates else None
        for candidate in candidates:
            dist = self._phase_dist(reference, candidate, dims=dims)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_anchor = candidate
        return best_dist, best_anchor

    def _phase_relevant_anchor(self, primitive=None):
        role_points = self._phase_role_points()
        primitive = primitive or str((self.current_phase or {}).get("primitive", "")).lower()
        if primitive in {"move_or_align", "place_or_release", "insert", "lower"}:
            return role_points.get("goal_region") or role_points.get("constraint_reference")
        if primitive in {
            "slide_or_sweep",
            "pull_or_drag",
            "pull_or_extract",
            "push",
            "press",
            "push_or_rotate_hinge",
            "pull_open_hinge",
            "rotate",
            "tilt_or_pour",
            "untilt_or_stop",
        }:
            return (
                role_points.get("secondary_object_to_move")
                or role_points.get("goal_region")
                or role_points.get("constraint_reference")
                or role_points.get("manipulated_object_contact")
            )
        return role_points.get("manipulated_object_contact") or role_points.get("tool_working_edge")

    def _phase_completion_observation(self, observation):
        current_phase = self.current_phase or {}
        primitive = str(current_phase.get("primitive", "")).lower()
        if primitive == "finish_or_retract":
            return True, "terminal phase; no anchor verification required", None, None
        phase_id = current_phase.get("phase_id", self.phase_index)
        last_action = self._phase_last_discrete_action()
        gripper_state = self._phase_current_gripper_state(observation)
        actual = gripper_state.get("discrete")
        profile = getattr(self, "phase_query_profile", {}) or {}
        family = str(profile.get("interaction_family", "")).lower()
        raw_points = self._phase_role_points_from_affordance(getattr(self, "phase_query_affordance", {}) or {})
        observed_manipulated = raw_points.get("manipulated_object_contact") or raw_points.get("tool_working_edge")
        anchor_candidates = self._phase_anchor_candidates(primitive, raw_points)
        if primitive in {"move_or_align", "place_or_release", "insert", "lower"}:
            reference = observed_manipulated or actual or last_action
        else:
            reference = actual or last_action
        distance_dims = 2 if primitive == "move_or_align" else 3
        dist, anchor = self._phase_best_anchor_distance(reference, anchor_candidates, distance_dims)
        threshold = self._phase_verify_xy_dist() if distance_dims == 2 else self._phase_verify_dist()
        gripper_open = gripper_state.get("open")
        stats = self._phase_command_motion_stats(phase_id=phase_id, primitive=primitive)
        commanded_dist = None
        commanded_anchor = None
        commanded_start_dist = None
        commanded_start_anchor = None
        commanded_tool_edge_dist = None
        commanded_tool_edge = None
        if isinstance(stats.get("last"), list):
            commanded_dist, commanded_anchor = self._phase_best_anchor_distance(
                stats.get("last"), anchor_candidates, distance_dims
            )
            if isinstance(stats.get("first"), list):
                commanded_start_dist, commanded_start_anchor = self._phase_best_anchor_distance(
                    stats.get("first"), anchor_candidates, distance_dims
                )
            if primitive == "move_or_align" and family == "pour_to_target":
                manipulated = raw_points.get("manipulated_object_contact")
                tool_edge = raw_points.get("tool_working_edge")
                goal = raw_points.get("goal_region") or raw_points.get("constraint_reference")
                if manipulated and tool_edge and goal:
                    try:
                        offset = np.array(tool_edge[:3], dtype=float) - np.array(manipulated[:3], dtype=float)
                        commanded_tool_edge = (
                            np.array(stats.get("last")[:3], dtype=float) + offset
                        ).round().astype(int).tolist()
                        commanded_tool_edge_dist = self._phase_dist(commanded_tool_edge, goal, dims=2)
                    except Exception:
                        commanded_tool_edge_dist = None

        close_enough = dist is None or dist <= threshold
        commanded_close_enough = commanded_dist is not None and commanded_dist <= threshold
        commanded_start_close_enough = commanded_start_dist is not None and commanded_start_dist <= threshold
        commanded_tool_edge_close = commanded_tool_edge_dist is not None and commanded_tool_edge_dist <= threshold
        open_enough = gripper_open is None or gripper_open >= 0.5
        closed_enough = gripper_open is None or gripper_open < 0.5
        command_open_enough = bool(stats.get("final_gripper_open", False))
        command_closed_enough = bool(stats.get("final_gripper_closed", False))
        motion_enough = max(stats.get("xyz_delta", 0.0), stats.get("xy_delta", 0.0)) >= self._phase_verify_min_motion()
        lift_enough = (
            stats.get("z_delta", 0.0) >= self._phase_verify_min_lift_z()
            and stats.get("final_z_delta", 0.0) >= 0.0
        )
        rot_enough = stats.get("rot_delta", 0.0) >= self._phase_verify_min_rot_delta()
        previous_phase_verified = any(
            item.get("phase_id") == phase_id - 1
            and str(item.get("status", "")).startswith("verified")
            for item in getattr(self, "completed_phases", []) or []
            if isinstance(item, dict)
        )

        if primitive == "approach":
            ok = close_enough and (
                open_enough or command_open_enough or not self._phase_verify_strict_gripper()
            )
        elif primitive in {"grasp", "grasp_or_contact", "grasp_or_contact_tool"}:
            ok = close_enough and (
                closed_enough or command_closed_enough or not self._phase_verify_strict_gripper()
            )
        elif primitive in {"move_or_align", "insert", "lower"}:
            ok = close_enough or commanded_close_enough or commanded_tool_edge_close
        elif primitive == "place_or_release":
            ok = close_enough and (
                open_enough or command_open_enough or not self._phase_verify_strict_gripper()
            )
        elif primitive in {"slide_or_sweep", "pull_or_drag", "pull_or_extract", "push", "press", "push_or_rotate_hinge"}:
            ok = motion_enough and (close_enough or commanded_start_close_enough)
        elif primitive in {"lift", "lift_or_stop", "pull_open_hinge"}:
            ok = lift_enough and (
                closed_enough or command_closed_enough or not self._phase_verify_strict_gripper()
            )
        elif primitive in {"rotate", "tilt_or_pour", "untilt_or_stop"}:
            ok = rot_enough or motion_enough
        elif primitive in {"release_or_stop", "release_or_retract"}:
            ok = previous_phase_verified and (
                open_enough or command_open_enough or not self._phase_verify_strict_gripper()
            )
        else:
            ok = close_enough or motion_enough

        reason = []
        if dist is not None:
            metric = "xy_dist_to_anchor" if distance_dims == 2 else "dist_to_anchor"
            reason.append(f"{metric}={dist:.1f}; threshold={threshold:.1f}")
        else:
            reason.append("dist_to_anchor=unknown")
        if commanded_dist is not None:
            metric = "command_xy_dist_to_anchor" if distance_dims == 2 else "command_dist_to_anchor"
            reason.append(f"{metric}={commanded_dist:.1f}")
        if commanded_start_dist is not None:
            metric = "command_start_xy_dist_to_anchor" if distance_dims == 2 else "command_start_dist_to_anchor"
            reason.append(f"{metric}={commanded_start_dist:.1f}")
        if commanded_tool_edge_dist is not None:
            reason.append(
                f"command_tool_edge_xy_dist_to_goal={commanded_tool_edge_dist:.1f}; "
                f"command_tool_edge={commanded_tool_edge}"
            )
        if stats:
            reason.append(
                "command_delta="
                f"xyz={stats.get('xyz_delta', 0.0):.1f},"
                f"xy={stats.get('xy_delta', 0.0):.1f},"
                f"z={stats.get('z_delta', 0.0):.1f},"
                f"rot={stats.get('rot_delta', 0.0):.1f},"
                f"final_gripper={'open' if stats.get('final_gripper_open') else 'closed'}"
            )
            reason.append(
                "required="
                f"motion>={self._phase_verify_min_motion():.1f},"
                f"lift_z>={self._phase_verify_min_lift_z():.1f},"
                f"rot>={self._phase_verify_min_rot_delta():.1f}"
            )
        if primitive in {"release_or_stop", "release_or_retract"}:
            reason.append(f"previous_phase_verified={previous_phase_verified}")
        if len(anchor_candidates) > 1:
            reason.append(f"anchor_candidates={anchor_candidates}")
        if observed_manipulated:
            reason.append(f"observed_manipulated={observed_manipulated}")
        if gripper_open is not None:
            reason.append(f"gripper_open={gripper_open:.3f}")
        return ok, "; ".join(reason), anchor or commanded_anchor, actual

    def _phase_retry_limit(self):
        return max(0, int(os.environ.get("XICM_PHASE_RETRY_LIMIT", "1")))

    def _phase_requires_verified_completion(self, primitive):
        return str(primitive).lower() in {
            "move_or_align",
            "place_or_release",
            "insert",
            "lower",
            "lift",
            "lift_or_stop",
            "pull_open_hinge",
            "pull_or_extract",
            "pull_or_drag",
            "slide_or_sweep",
            "push",
            "press",
            "push_or_rotate_hinge",
            "rotate",
            "tilt_or_pour",
            "untilt_or_stop",
            "release_or_stop",
            "release_or_retract",
        }

    def _phase_validate_pending_completion(self, observation):
        pending = getattr(self, "pending_phase_completion", None)
        if not pending or not self._is_phase_model():
            return
        ok, reason, anchor, actual = self._phase_completion_observation(observation)
        primitive = pending.get("primitive", "unknown")
        phase_id = pending.get("phase_id", self.phase_index)
        can_advance_after_retry = (
            pending.get("attempt", 0) >= self._phase_retry_limit()
            and not self._phase_requires_verified_completion(primitive)
        )
        if ok or can_advance_after_retry:
            status = "verified_observation" if ok else "advanced_after_retry_limit"
            self.completed_phases.append(
                {
                    "phase_id": phase_id,
                    "primitive": primitive,
                    "status": status,
                    "check": reason,
                }
            )
            self.phase_index += 1
            if primitive == "finish_or_retract":
                self.phase_prompts_used = self._phase_chain_max_steps()
            self.pending_phase_completion = None
            print(
                f"Phase completion accepted: phase_id={phase_id}; primitive={primitive}; "
                f"status={status}; {reason}; anchor={anchor}; actual={actual}"
            )
            return

        self.pending_phase_completion = {
            **pending,
            "attempt": int(pending.get("attempt", 0)) + 1,
            "last_failed_check": reason,
        }
        print(
            f"Phase completion rejected: phase_id={phase_id}; primitive={primitive}; "
            f"retry={self.pending_phase_completion['attempt']}; {reason}; anchor={anchor}; actual={actual}"
        )

    def _clip_discrete_action(self, action):
        fixed = [int(round(float(value))) for value in action[:7]]
        fixed[:3] = [int(np.clip(value, 0, 99)) for value in fixed[:3]]
        fixed[3:6] = [int(np.clip(value, 0, 71)) for value in fixed[3:6]]
        fixed[6] = 1 if fixed[6] >= 1 else 0
        return fixed

    def _phase_guard_discrete_actions(self, actions):
        if not self._phase_anchor_guard_enabled():
            return actions
        if not actions:
            return actions
        current_phase = self.current_phase or {}
        primitive = str(current_phase.get("primitive", "")).lower()
        if primitive == "finish_or_retract":
            return [self._phase_terminal_action(open_gripper=False, retract=False)]

        role_points = self._phase_role_points()
        manipulated = role_points.get("manipulated_object_contact")
        tool_edge = role_points.get("tool_working_edge")
        secondary = role_points.get("secondary_object_to_move")
        constraint = role_points.get("constraint_reference")
        goal = role_points.get("goal_region") or role_points.get("constraint_reference")
        guarded = [self._clip_discrete_action(action) for action in actions]
        profile = getattr(self, "phase_query_profile", {}) or {}
        geometry = getattr(self, "phase_query_geometry", {}) or {}
        family = str(profile.get("interaction_family", "")).lower()
        axis_hint = " ".join(
            str(value).lower()
            for value in [
                profile.get("axis_constraint", ""),
                profile.get("motion_sequence", ""),
                geometry.get("motion_axis", ""),
                geometry.get("motion_type", ""),
            ]
        )

        def with_xyz(action, xyz, gripper=None):
            updated = list(action)
            updated[:3] = [int(np.clip(value, 0, 99)) for value in xyz[:3]]
            if gripper is not None:
                updated[6] = int(gripper)
            return self._clip_discrete_action(updated)

        def with_rot(action, rot, gripper=None):
            updated = list(action)
            updated[3:6] = [int(value) % 72 for value in rot[:3]]
            if gripper is not None:
                updated[6] = int(gripper)
            return self._clip_discrete_action(updated)

        knob_start_rot = [67, 42, 31]
        knob_mid_rot = [6, 36, 49]
        knob_end_rot = [5, 30, 67]

        def lateral_proposal_delta():
            if len(guarded) < 2:
                return np.zeros(3, dtype=int)
            delta = np.array(guarded[-1][:3], dtype=int) - np.array(guarded[0][:3], dtype=int)
            if "horizontal" in axis_hint or "linear_socket" in axis_hint:
                delta[2] = 0
            return delta

        def default_motion_target(start_xyz):
            start = np.array(start_xyz, dtype=int)
            delta = lateral_proposal_delta()
            if primitive in {"pull_or_extract", "pull_or_drag"}:
                if constraint:
                    raw = start - np.array(constraint, dtype=int)
                    if np.linalg.norm(raw[:2]) >= 2:
                        delta = raw
                if np.linalg.norm(delta[:2]) < 6:
                    delta = np.array([18 if start[0] >= 50 else -18, 0, 0], dtype=int)
                delta[2] = 0
                return np.clip(start + np.round(1.25 * delta).astype(int), 0, 99).tolist()
            if primitive == "push_or_rotate_hinge":
                if family == "hinged_panel_close" or "horizontal_hinge" in axis_hint:
                    return [int(start[0]), int(start[1]), int(np.clip(start[2] - 18, 0, 99))]
                direction = 18 if start[1] < 50 else -18
                return [int(start[0]), int(np.clip(start[1] + direction, 0, 99)), int(start[2])]
            if primitive in {"push", "press"}:
                if "surface_normal" in axis_hint or "button" in family:
                    return [int(start[0]), int(start[1]), int(np.clip(start[2] - 10, 0, 99))]
                direction = 14 if start[1] < 50 else -14
                return [int(start[0]), int(np.clip(start[1] + direction, 0, 99)), int(start[2])]
            if primitive == "pull_open_hinge":
                return [int(start[0]), int(start[1]), int(np.clip(start[2] + 18, 0, 99))]
            return start.tolist()

        if primitive == "approach" and manipulated:
            hover = list(manipulated)
            hover[2] = int(np.clip(manipulated[2] + self._phase_approach_clearance_z(), 0, 99))
            base = guarded[-1]
            return [with_xyz(base, hover, 1)]

        if primitive in {"grasp", "grasp_or_contact", "grasp_or_contact_tool"} and manipulated:
            base = guarded[-1]
            return [
                with_xyz(base, manipulated, 1),
                with_xyz(base, manipulated, 0),
            ]

        if primitive in {
            "slide_or_sweep",
            "pull_or_drag",
            "pull_or_extract",
            "push",
            "press",
            "push_or_rotate_hinge",
            "pull_open_hinge",
        }:
            if primitive in {"pull_or_drag", "pull_or_extract"}:
                target = role_points.get("goal_region") or secondary
            else:
                target = goal or secondary
            if manipulated and tool_edge and target:
                handle = np.array(manipulated, dtype=int)
                edge = np.array(tool_edge, dtype=int)
                target_edge = np.array(target, dtype=int)
                handle_from_edge = handle - edge
                delta = target_edge - edge
                through_edge = np.clip(target_edge + np.round(0.35 * delta).astype(int), 0, 99)
                mid_edge = np.clip(np.round((edge + target_edge) / 2).astype(int), 0, 99)
                blade_z = int(np.clip(max(self._phase_tool_surface_min_z(), min(edge[2], target_edge[2])), 0, 99))
                base = guarded[0]

                def handle_for_edge(edge_xyz):
                    edge_xyz = np.array(edge_xyz, dtype=int)
                    edge_xyz[2] = blade_z
                    return np.clip(edge_xyz + handle_from_edge, 0, 99)

                return [
                    with_xyz(base, handle_for_edge(mid_edge), 0),
                    with_xyz(base, handle_for_edge(target_edge), 0),
                    with_xyz(base, handle_for_edge(through_edge), 0),
                ]
            if manipulated and target:
                start = np.array(manipulated, dtype=int)
                end = np.array(target, dtype=int)
                if (
                    primitive in {"pull_or_drag", "pull_or_extract"}
                    and np.linalg.norm((end - start)[:2]) < self._phase_verify_min_motion()
                ):
                    end = np.array(default_motion_target(start), dtype=int)
                delta = end - start
                through = np.clip(end + np.round(0.25 * delta).astype(int), 0, 99)
                mid = np.clip(np.round((start + end) / 2).astype(int), 0, 99)
                base = guarded[0]
                z_low = int(np.clip(min(start[2], end[2]), 0, 99))
                mini = [
                    with_xyz(base, [mid[0], mid[1], z_low], 0),
                    with_xyz(base, [end[0], end[1], z_low], 0),
                    with_xyz(base, [through[0], through[1], z_low], 0),
                ]
                return mini[: max(2, min(3, len(mini)))]
            if manipulated and primitive in {"pull_or_drag", "pull_or_extract", "push", "press", "push_or_rotate_hinge", "pull_open_hinge"}:
                start = np.array(manipulated, dtype=int)
                end = np.array(default_motion_target(start), dtype=int)
                mid = np.clip(np.round((start + end) / 2).astype(int), 0, 99)
                base = guarded[0]
                gripper = 0 if primitive in {"pull_or_drag", "pull_or_extract", "pull_open_hinge"} else base[6]
                return [
                    with_xyz(base, start, gripper),
                    with_xyz(base, mid, gripper),
                    with_xyz(base, end, gripper),
                ]
            if primitive in {"pull_or_drag", "pull_or_extract", "push", "press", "push_or_rotate_hinge", "pull_open_hinge"}:
                start_xyz = target or guarded[0][:3]
                start = np.array(start_xyz, dtype=int)
                end = np.array(default_motion_target(start), dtype=int)
                mid = np.clip(np.round((start + end) / 2).astype(int), 0, 99)
                base = guarded[0]
                gripper = 0 if primitive in {"pull_or_drag", "pull_or_extract", "pull_open_hinge"} else base[6]
                return [
                    with_xyz(base, start, gripper),
                    with_xyz(base, mid, gripper),
                    with_xyz(base, end, gripper),
                ]
            if target:
                return [with_xyz(action, target, 0) for action in guarded[: max(1, min(3, len(guarded)))]]

        if primitive in {"lift", "lift_or_stop"}:
            previous_slide = self._phase_last_action_for_primitives({"slide_or_sweep"})
            if family == "tool_scoop_under_object" and previous_slide:
                anchor = previous_slide[:3]
                base = previous_slide
            else:
                anchor = manipulated or secondary or (guarded[-1][:3] if guarded else None)
                base = guarded[0]
            if anchor:
                z0 = int(anchor[2])
                z_values = [
                    int(np.clip(max(z0 + 10, 28), 0, 99)),
                    int(np.clip(max(z0 + 20, 38), 0, 99)),
                    int(np.clip(max(z0 + 30, 48), 0, 99)),
                ]
                return [with_xyz(base, [anchor[0], anchor[1], z], 0) for z in z_values]

        if primitive in {"rotate", "tilt_or_pour", "untilt_or_stop"}:
            previous_move = self._phase_last_action_for_primitives({"move_or_align"})
            previous_tilt = self._phase_last_action_for_primitives({"tilt_or_pour"})
            if primitive == "tilt_or_pour":
                pivot = (previous_move[:3] if previous_move else None) or manipulated or secondary or goal
                previous = previous_move or self._phase_last_action_for_primitives(
                    {"grasp", "grasp_or_contact", "grasp_or_contact_tool"}
                )
            elif primitive == "untilt_or_stop":
                pivot = (
                    (previous_tilt[:3] if previous_tilt else None)
                    or (previous_move[:3] if previous_move else None)
                    or manipulated
                    or secondary
                    or goal
                )
                previous = previous_tilt or previous_move or self._phase_last_action_for_primitives(
                    {"grasp", "grasp_or_contact", "grasp_or_contact_tool"}
                )
            else:
                pivot = manipulated or secondary or goal
                previous = self._phase_last_action_for_primitives(
                    {"grasp", "grasp_or_contact", "grasp_or_contact_tool"}
                )
            if pivot:
                base = previous or guarded[0]
                if primitive == "rotate" and family == "knob_or_handle_rotation":
                    first = with_xyz(with_rot(base, knob_start_rot, 0), pivot, 0)
                    second = with_xyz(with_rot(first, knob_mid_rot, 0), pivot, 0)
                    third = with_xyz(with_rot(first, knob_end_rot, 0), pivot, 0)
                    return [first, second, third]
                rot_axis = 5
                if primitive in {"tilt_or_pour", "untilt_or_stop"} or "tilt" in axis_hint:
                    rot_axis = 4
                first = with_xyz(base, pivot, 0)
                second = list(first)
                third = list(first)
                direction = -1 if primitive == "untilt_or_stop" else 1
                second[rot_axis] = int((second[rot_axis] + direction * 18) % 72)
                third[rot_axis] = int((third[rot_axis] + direction * 30) % 72)
                second[6] = 0
                third[6] = 0
                if primitive == "tilt_or_pour" and family == "pour_to_target":
                    return [
                        self._clip_discrete_action(second),
                        self._clip_discrete_action(third),
                        self._clip_discrete_action(third),
                    ]
                return [self._clip_discrete_action(second), self._clip_discrete_action(third)]

        if primitive in {"move_or_align", "insert", "lower"}:
            target = goal or secondary
            if family == "pour_to_target" and target and manipulated and tool_edge:
                previous = self._phase_last_action_for_primitives({"grasp", "grasp_or_contact", "grasp_or_contact_tool"})
                start = np.array((previous[:3] if previous else manipulated), dtype=int)
                handle = np.array(manipulated, dtype=int)
                edge = np.array(tool_edge, dtype=int)
                target_edge = np.array(target, dtype=int)
                handle_from_edge = handle - edge
                end = np.clip(target_edge + handle_from_edge, 0, 99)
                end[2] = int(np.clip(max(end[2], target_edge[2] + 4), 0, 99))
                clearance = self._phase_transport_clearance_z()
                travel_z = int(np.clip(max(start[2], end[2], target_edge[2]) + clearance, 0, 99))
                mid = np.clip(np.round((start + end) / 2).astype(int), 0, 99)
                base = previous or guarded[0]
                return [
                    with_xyz(base, [mid[0], mid[1], travel_z], 0),
                    with_xyz(base, [end[0], end[1], travel_z], 0),
                    with_xyz(base, end, 0),
                ]
            if target:
                previous = self._phase_last_action_for_primitives({"grasp", "grasp_or_contact", "grasp_or_contact_tool"})
                start = manipulated or (previous[:3] if previous else guarded[0][:3])
                start = np.array(start, dtype=int)
                end = np.array(target, dtype=int)
                clearance = self._phase_transport_clearance_z()
                travel_z = int(np.clip(max(start[2], end[2]) + clearance, 0, 99))
                mid = np.clip(np.round((start + end) / 2).astype(int), 0, 99)
                base = previous or guarded[0]
                return [
                    with_xyz(base, [mid[0], mid[1], travel_z], 0),
                    with_xyz(base, [end[0], end[1], travel_z], 0),
                ]

        if primitive in {"place_or_release", "release_or_stop", "release_or_retract"}:
            target = goal or secondary
            if target:
                base = guarded[-1]
                return [
                    with_xyz(base, target, 0),
                    with_xyz(base, target, 1),
                ]
            return [self._phase_terminal_action(open_gripper=True)]

        return guarded

    def _use_query_images(self):
        return bool(getattr(self, "components", {}).get("is_vl_model", False))

    def _messages_have_images(self, messages):
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                if any(isinstance(item, dict) and item.get("type") == "image" for item in content):
                    return True
        return False

    def _generate_text(self, messages, max_tokens=256):
        prompt = self.components["processor"].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        llm_inputs = {
            "prompt": prompt
        }
        if self._messages_have_images(messages):
            image_inputs, video_inputs = process_vision_info(messages)
            multi_modal_data = {}
            if image_inputs:
                multi_modal_data["image"] = image_inputs
            if video_inputs:
                multi_modal_data["video"] = video_inputs
            if multi_modal_data:
                llm_inputs["multi_modal_data"] = multi_modal_data

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=max_tokens,
            stop_token_ids=[],
        )

        outputs = self.components["llm"].generate([llm_inputs], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def _build_final_messages(self, user_prompt):
        system_prompt = self.SYSTEM_PROMPT
        if not self._use_query_images():
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        system_prompt = (
            f"{system_prompt} You may also receive front and overhead RGB images "
            "of the current unseen query. Use those images only as the current "
            "initial observation. If the text prompt includes geometry, goal-state, "
            "or contact descriptors, treat those descriptors as the authoritative "
            "structured hints for the current ablation."
        )
        content = [
            {
                "type": "text",
                "text": (
                    "Current unseen query images are attached below. "
                    "First image: front RGB view. Second image: overhead/top RGB view."
                ),
            }
        ]
        if self.vl_front_rgb_path:
            content.append({"type": "image", "image": self.vl_front_rgb_path})
        if self.vl_overhead_rgb_path:
            content.append({"type": "image", "image": self.vl_overhead_rgb_path})
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _split_v4_prompt(self, user_prompt):
        stage1_marker = "<<<V4_STAGE1_PROMPT>>>"
        stage2_marker = "<<<V4_STAGE2_CONTEXT>>>"
        if stage1_marker not in user_prompt or stage2_marker not in user_prompt:
            raise ValueError("v4 prompt is missing expected stage markers")
        stage1_text, _, stage2_text = user_prompt.partition(stage2_marker)
        stage1_text = stage1_text.replace(stage1_marker, "", 1).strip()
        return stage1_text, stage2_text.strip()

    def _run_v4_semantic_bottleneck(self, user_prompt):
        stage1_prompt, stage2_context = self._split_v4_prompt(user_prompt)

        stage1_system_prompt = (
            "You are a Franka Panda robot planner. First answer the high-level question: "
            "what physical manipulation should the robot perform in the current unseen scene? "
            "Use retrieved demos only as analogies for contact mode, relation, motion direction, and gripper timing. "
            "Ground the plan in the unseen current observation: copy target_current_coordinate and reference_coordinate exactly when present, "
            "and name the active_reference_part rather than a generic support object. "
            "Return only one compact JSON object with the requested semantic fields. Do not output 7D actions in this stage."
        )
        stage2_system_prompt = (
            "You are a Franka Panda robot with a parallel gripper. "
            "Use the semantic manipulation plan, the current unseen observation, and compatible retrieved trajectories to answer: "
            "what action should the robot take? "
            "First write a short relative_action_sketch in simple robot motion language, then use that sketch to accurately predict key_actions_7d. "
            "Return only one compact JSON object with exactly two fields: relative_action_sketch and key_actions_7d. "
            "key_actions_7d must be a list of [x, y, z, roll, pitch, yaw, gripper] integer lists, where gripper is 1=open and 0=closed."
        )

        print(stage1_system_prompt)
        print()
        print(stage1_prompt)

        semantic_plan = self._generate_text(
            [
                {"role": "system", "content": stage1_system_prompt},
                {"role": "user", "content": stage1_prompt},
            ],
            max_tokens=384,
        )
        print("Semantic plan:", semantic_plan)

        plan_insert = (
            "Stage 1 semantic manipulation plan:\n"
            f"{semantic_plan}\n\n"
            "Use this plan as the primary task intent. Before final 7D actions, write a relative action sketch that follows the plan, "
            "then convert that sketch into the final key_actions_7d using the current observation and compatible retrieved trajectory rhythms."
        )
        stage2_prompt = stage2_context.replace("<<<V4_STAGE2_PLAN_INSERT_HERE>>>", plan_insert, 1)
        stage2_prompt = stage2_prompt.replace(
            "The agent will insert the Stage 1 semantic manipulation plan here before asking for the final 7D actions.",
            "",
            1,
        )

        print()
        print(stage2_system_prompt)
        print()
        print(stage2_prompt)

        output_text = self._generate_text(
            [
                {"role": "system", "content": stage2_system_prompt},
                {"role": "user", "content": stage2_prompt},
            ],
            max_tokens=512,
        )
        print(f"Prediction:", output_text)
        return output_text

    def _preprocess(self, obs, step, **kwargs):
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        lang_goal = kwargs['lang_goal']

        front_rgb_img = obs['front_rgb']
        front_rgb_img=front_rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        front_rgb_img = np.clip((front_rgb_img).astype(np.uint8), 0, 255)

        front_rgb_img = Image.fromarray(front_rgb_img)
        front_rgb_dir = os.path.join(self.savedir, 'rgb_dir', 'front', str(self.episode_id))
        os.makedirs(front_rgb_dir, exist_ok=True)
        front_rgb_img.save(os.path.join(front_rgb_dir, 'rgb.png'))
        self.front_rgb_path=os.path.join(front_rgb_dir, 'rgb.png')

        for camera in CAMERAS:
            rgb_img = obs[f'{camera}_rgb']
            rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
            rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)

            rgb_dict[camera] = rgb_img
            if camera in {"front", "overhead"}:
                query_view_dir = os.path.join(self.savedir, 'rgb_dir', 'query_views', str(self.episode_id))
                os.makedirs(query_view_dir, exist_ok=True)
                query_view_path = os.path.join(query_view_dir, f'{camera}.png')
                Image.fromarray(rgb_img).save(query_view_path)
                if camera == "front":
                    self.vl_front_rgb_path = query_view_path
                elif camera == "overhead":
                    self.vl_overhead_rgb_path = query_view_path

            mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

            mask = obs[f'{camera}_mask']
            mask = mask.squeeze().cpu().numpy() 

            mask_dict[camera] = mask

            point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
            point_cloud_dict[camera] = point_cloud

        if len(self.actions) == 0 or (self._closed_loop_should_replan() and not self._is_phase_model()):
            user_prompt = self.handler.get_user_prompt_ranking(
                mask_dict,
                mask_id_to_sim_name,
                point_cloud_dict,
                custom_num_demos=self.demo_num_per_icl,
                taskname=lang_goal,
                image_path=self.front_rgb_path,
                seed=self.seed,
                ranking_metric=self.ranking_method,
                current_phase_index=self.phase_index,
                completed_phases=self.completed_phases,
                phase_history=self.phase_history,
            )
            if self._is_phase_model():
                self.phase_chain = getattr(self.handler, "last_phase_chain", self.phase_chain)
                self.current_phase = getattr(self.handler, "last_current_phase", self.current_phase)
                self.phase_query_affordance = getattr(self.handler, "last_query_affordance", self.phase_query_affordance)
                self.phase_query_profile = getattr(self.handler, "last_query_profile", self.phase_query_profile)
                self.phase_query_geometry = getattr(self.handler, "last_query_geometry", self.phase_query_geometry)
                self._phase_update_anchor_memory()
            if self._closed_loop_should_replan() and not self._is_phase_model():
                user_prompt += self._closed_loop_prompt_suffix(step, obs)

            if self._is_v4():
                return self._run_v4_semantic_bottleneck(user_prompt)

            print(self.SYSTEM_PROMPT) 

            print()

            print(user_prompt)

            messages = self._build_final_messages(user_prompt)


            ########################### vllm local deploy #####################################
            max_tokens = 512 if self._is_phase_model() else 256
            output_text = self._generate_text(messages, max_tokens=max_tokens)

            print(f"Prediction:", output_text)
            if self._is_phase_model():
                self._write_phase_prompt_trace(step, user_prompt, messages, output_text)
            return output_text
    
    def re_match(self, text):
        pattern = r'\[([^\[\]]+\d[^\[\]]*)\]'
        matches = re.findall(pattern, text)
        
        valid_lists = []
        for match in matches:
            items = [int(x.strip()) for x in match.split(',')]
            if len(items) == 7:
                valid_lists.append(items)
        return valid_lists

    def _extract_key_actions_7d(self, text):
        cleaned = str(text).strip()
        if "```" in cleaned:
            cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "next_action_7d" in parsed:
                action = parsed["next_action_7d"]
                if isinstance(action, list) and len(action) == 7:
                    return [[int(round(float(value))) for value in action]]
            if isinstance(parsed, dict) and "action_7d" in parsed:
                action = parsed["action_7d"]
                if isinstance(action, list) and len(action) == 7:
                    return [[int(round(float(value))) for value in action]]
            if isinstance(parsed, dict) and "key_actions_7d" in parsed:
                actions = parsed["key_actions_7d"]
                if isinstance(actions, list):
                    return [
                        [int(round(float(value))) for value in action]
                        for action in actions
                        if isinstance(action, list) and len(action) == 7
                    ]
            if isinstance(parsed, dict) and "phase_actions_7d" in parsed:
                actions = parsed["phase_actions_7d"]
                if isinstance(actions, list):
                    return [
                        [int(round(float(value))) for value in action]
                        for action in actions
                        if isinstance(action, list) and len(action) == 7
                    ]
            if isinstance(parsed, dict) and "actions_7d" in parsed:
                actions = parsed["actions_7d"]
                if isinstance(actions, list):
                    return [
                        [int(round(float(value))) for value in action]
                        for action in actions
                        if isinstance(action, list) and len(action) == 7
                    ]
            if isinstance(parsed, list):
                return [
                    [int(round(float(value))) for value in action]
                    for action in parsed
                    if isinstance(action, list) and len(action) == 7
                ]
        except Exception:
            pass
        return self.re_match(cleaned)

    def _postprocess(self, output_text):
        try:
            actions = self._extract_key_actions_7d(str(output_text))
            bypass_phase_guard = bool(getattr(self, "_phase_bypass_anchor_guard_once", False))
            if self._is_phase_model() and not bypass_phase_guard:
                actions = self._phase_guard_discrete_actions(actions)
            if bypass_phase_guard:
                self._phase_bypass_anchor_guard_once = False
            self._last_discrete_actions = actions
            if self._is_phase_model():
                self._write_phase_parsed_action_trace(actions)
            print("parsed actions: ", actions)
        except Exception as e:
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            self._last_discrete_actions = actions
            print(e)
            print('Error when parsing actions. Falling back to default.')
        
        if len(np.array(actions).shape) == 1:
            actions = [actions]

        output = []
        for action in actions:
            if len(action) != 7:
                print("error:::", actions)
                if len(action)==6:
                    action.append(1)
                else:
                    action = [57, 49, 87, 0, 39, 0, 1]
            trans_indicies = np.array(action[:3])
            rot_and_grip_indicies = np.array(action[3:6])
            is_gripper_open = action[6]

            bounds = SCENE_BOUNDS
            res = (bounds[3:] - bounds[:3]) / 100
            attention_coordinate = bounds[:3] + res * trans_indicies + res / 2
            quat = discrete_euler_to_quaternion(rot_and_grip_indicies)
            
            continuous_action = np.concatenate([
                attention_coordinate,
                quat,
                [is_gripper_open],
                [1],
            ])
            output.append(continuous_action)
        
        # get subsequent predicted actions
        return output[:26]
        

    def act(self, step: int, observation: dict,
            deterministic=False, **kwargs) -> ActResult:
        # inference
        if self._closed_loop_should_replan() and not self._is_phase_model():
            output_text = self._preprocess(observation, step, **kwargs)
            output = self._postprocess(output_text)
            if len(output) == 0:
                output = [[57, 49, 87, 0, 39, 0, 1]]
            continuous_action = output[0]
            self.closed_loop_replans += 1
            discrete_action = (
                self._last_discrete_actions[0]
                if getattr(self, "_last_discrete_actions", None)
                else self._continuous_action_to_discrete(continuous_action)
            )
            self.closed_loop_history.append(
                {
                    "step": int(self.step),
                    "action_7d": discrete_action,
                }
            )
        else:
            if self._is_phase_model() and len(self.actions) == 0:
                if (
                    getattr(self, "pending_phase_completion", None)
                    and self.phase_prompts_used < self._phase_chain_max_steps()
                ):
                    self._phase_refresh_query_state(observation, **kwargs)
                    self._phase_validate_pending_completion(observation)
            if len(self.actions) == 0:
                if self._is_phase_model() and self.phase_prompts_used >= self._phase_chain_max_steps():
                    fallback = self._phase_budget_fallback_action()
                    print(
                        "Phase prompt budget exhausted; using fallback terminal action:",
                        fallback,
                    )
                    self._phase_bypass_anchor_guard_once = True
                    output = self._postprocess(json.dumps({"phase_actions_7d": [fallback]}))
                else:
                    output_text = self._preprocess(observation, step, **kwargs)
                    if self._is_phase_model():
                        self.phase_prompts_used += 1
                    output = self._postprocess(output_text)
                if len(output) == 0:
                    if self._is_phase_model():
                        self._phase_bypass_anchor_guard_once = True
                    output = self._postprocess(
                        json.dumps({"phase_actions_7d": [self._phase_budget_fallback_action()]})
                    )
                self.actions = output
                self.discrete_actions = [
                    self._clip_discrete_action(action)
                    for action in getattr(self, "_last_discrete_actions", [])[: len(output)]
                    if isinstance(action, list) and len(action) == 7
                ]
	            
            continuous_action = self.actions.pop(0)
            if self._is_phase_model():
                if getattr(self, "discrete_actions", None):
                    discrete_action = self.discrete_actions.pop(0)
                else:
                    discrete_action = self._continuous_action_to_discrete(continuous_action)
                current_phase = self.current_phase or {}
                self.phase_history.append(
                    {
                        "phase_id": current_phase.get("phase_id", self.phase_index),
                        "primitive": current_phase.get("primitive", "unknown"),
                        "action_7d": discrete_action,
                    }
                )
                if len(self.actions) == 0 and self.phase_prompts_used < self._phase_chain_max_steps():
                    existing_pending = getattr(self, "pending_phase_completion", None) or {}
                    existing_phase_id = existing_pending.get("phase_id")
                    current_phase_id = current_phase.get("phase_id", self.phase_index)
                    attempt = (
                        int(existing_pending.get("attempt", 0))
                        if existing_phase_id == current_phase_id
                        else 0
                    )
                    self.pending_phase_completion = {
                        "phase_id": current_phase_id,
                        "primitive": current_phase.get("primitive", "unknown"),
                        "attempt": attempt,
                        "last_failed_check": existing_pending.get("last_failed_check"),
                    }

        self.step += 1
        
        # copy_obs = {k: v.cpu() for k, v in observation.items()}
        copy_obs={}
        for k, v in observation.items():
            # print(k, type(v))
            if k=='lang_goal':
                copy_obs[k]=v
            else:
                copy_obs[k]=v.cpu()
        return ActResult(continuous_action,
                         observation_elements=copy_obs,
                         info=None)
    
    def act_summaries(self) -> List[Summary]:
        return []

    def reset(self):
        super().reset()
        self.step = 0
        self.episode_id += 1
        self._prev_action = None
        self.actions = []
        self.discrete_actions = []
        self.closed_loop_replans = 0
        self.closed_loop_history = []
        self._last_discrete_actions = []
        self._phase_bypass_anchor_guard_once = False
        self.phase_chain = []
        self.current_phase = None
        self.phase_index = 0
        self.completed_phases = []
        self.phase_history = []
        self.phase_prompts_used = 0
        self.pending_phase_completion = None
        self.phase_anchor_memory = {}
        self.phase_query_affordance = {}
        self.phase_query_profile = {}
        self.phase_query_geometry = {}

    def load_weights(self, savedir: str, components={}):
        # no weight to load
        # only build task handler
        self.savedir = savedir
        
        self.components=components

        from form_icl_demonstrations_crosstask_ranking import create_task_handler

        self.handler = create_task_handler(self.task_name)
        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return
