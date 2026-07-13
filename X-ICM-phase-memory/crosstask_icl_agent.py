from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
import glob
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, quaternion_to_discrete_euler, CAMERAS
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from vllm import LLM, SamplingParams 
from qwen_vl_utils import process_vision_info
from phase_memory_pipeline import PhaseMemoryController


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

        if self._is_phase_memory_mode():
            self.SYSTEM_PROMPT = (
                "You are a Franka Panda phase-step policy with a parallel gripper. "
                "You receive one current phase, current scene images, long-term memory from retrieved seen demonstrations, "
                "and short-term memory from this episode. Decide if the current phase is done, failed, or needs exactly one next 7D action. "
                "Never output a full trajectory and never try to repair a failed phase indefinitely."
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


    def _is_closed_loop(self):
        normalized = self.ranking_method.replace("-", "_")
        return "closed_loop" in normalized or ".cl" in normalized

    def _is_phase_anchor_replay(self):
        normalized = self.ranking_method.lower().replace("-", "_")
        return "phase_anchor" in normalized

    def _is_phase_memory_mode(self):
        normalized = self.ranking_method.lower().replace("-", "_")
        return "phase_memory" in normalized

    def _phase_anchor_review_root(self):
        root = os.environ.get("XICM_PHASE_ANCHOR_REVIEW_ROOT", "")
        if not root:
            raise RuntimeError("XICM_PHASE_ANCHOR_REVIEW_ROOT must point to a phase-anchor review folder.")
        if not os.path.isdir(root):
            raise RuntimeError(f"XICM_PHASE_ANCHOR_REVIEW_ROOT does not exist: {root}")
        return root

    def _phase_anchor_task_dir(self):
        root = self._phase_anchor_review_root()
        episode_id = max(0, int(self.episode_id))
        patterns = [
            os.path.join(root, f"*_{self.task_name}_episode{episode_id}"),
            os.path.join(root, f"{self.task_name}_episode{episode_id}"),
            os.path.join(root, f"*{self.task_name}*episode{episode_id}*"),
        ]
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern))
        matches = [path for path in matches if os.path.isdir(path)]
        if not matches:
            raise RuntimeError(
                f"No phase-anchor task folder found for task={self.task_name}, episode={episode_id} under {root}"
            )
        return sorted(matches)[0]

    def _load_phase_anchor_actions_7d(self):
        task_dir = self._phase_anchor_task_dir()
        action_path = os.path.join(task_dir, "06_compiled_open_loop_actions.json")
        if not os.path.exists(action_path):
            if os.environ.get("XICM_PHASE_ANCHOR_STRICT", "0") == "1":
                raise RuntimeError(f"Missing compiled phase-anchor actions: {action_path}")
            print(f"PhaseAnchorReplay warning: missing compiled actions, using no-op fallback: {action_path}")
            return [[50, 50, 70, 0, 36, 0, 1]]
        with open(action_path) as handle:
            data = json.load(handle)
        actions = data.get("actions_7d") or []
        if not actions:
            if os.environ.get("XICM_PHASE_ANCHOR_STRICT", "0") == "1":
                raise RuntimeError(f"No actions_7d found in {action_path}")
            print(f"PhaseAnchorReplay warning: no compiled actions, using no-op fallback: {action_path}")
            return [[50, 50, 70, 0, 36, 0, 1]]
        parsed = []
        for action in actions:
            if not isinstance(action, list) or len(action) != 7:
                raise RuntimeError(f"Bad action in {action_path}: {action}")
            parsed.append([int(round(float(value))) for value in action])
        print(f"PhaseAnchorReplay loaded {len(parsed)} actions from {action_path}")
        print("PhaseAnchorReplay actions:", parsed)
        return parsed

    def _phase_memory_review_root(self):
        root = (
            os.environ.get("XICM_PHASE_MEMORY_REVIEW_ROOT", "")
            or os.environ.get("XICM_PHASE_ANCHOR_REVIEW_ROOT", "")
        )
        if not root:
            raise RuntimeError(
                "XICM_PHASE_MEMORY_REVIEW_ROOT or XICM_PHASE_ANCHOR_REVIEW_ROOT must point to phase packets."
            )
        if not os.path.isdir(root):
            raise RuntimeError(f"Phase-memory review root does not exist: {root}")
        return root

    def _phase_memory_retrieval_method(self):
        return os.environ.get(
            "XICM_PHASE_MEMORY_RETRIEVAL_METHOD",
            "lang_vis.out.geo.aff_v3.rerank_top50",
        )

    def _phase_memory_retrieval_k(self):
        default_k = min(4, max(1, int(self.demo_num_per_icl or 4)))
        return int(os.environ.get("XICM_PHASE_MEMORY_RETRIEVAL_K", str(default_k)))

    def _build_phase_memory_retrieval_prompt(
        self,
        mask_dict,
        mask_id_to_sim_name,
        point_cloud_dict,
        lang_goal,
    ):
        retrieval_method = self._phase_memory_retrieval_method()
        try:
            return self.handler.get_user_prompt_ranking(
                mask_dict,
                mask_id_to_sim_name,
                point_cloud_dict,
                custom_num_demos=self._phase_memory_retrieval_k(),
                taskname=lang_goal,
                image_path=self.front_rgb_path,
                seed=self.seed,
                ranking_metric=retrieval_method,
            )
        except Exception as exc:
            print(f"PhaseMemory warning: retrieval prompt failed, using empty long-term retrieval memory: {exc}")
            return ""

    def _ensure_phase_memory_controller(
        self,
        mask_dict,
        mask_id_to_sim_name,
        point_cloud_dict,
        lang_goal,
    ):
        controller = getattr(self, "phase_memory_controller", None)
        if controller is not None and controller.episode_id == max(0, int(self.episode_id)):
            return controller
        retrieval_prompt = self._build_phase_memory_retrieval_prompt(
            mask_dict,
            mask_id_to_sim_name,
            point_cloud_dict,
            lang_goal,
        )
        self.phase_memory_controller = PhaseMemoryController(
            task_name=self.task_name,
            episode_id=max(0, int(self.episode_id)),
            review_root=self._phase_memory_review_root(),
            savedir=self.savedir,
            retrieval_method=self._phase_memory_retrieval_method(),
            retrieval_prompt=retrieval_prompt,
            max_actions_per_phase=int(os.environ.get("XICM_PHASE_MEMORY_MAX_ACTIONS_PER_PHASE", "2")),
            max_repeated_voxel=int(os.environ.get("XICM_PHASE_MEMORY_MAX_REPEATED_VOXEL", "1")),
            anchor_distance_limit=float(os.environ.get("XICM_PHASE_MEMORY_ANCHOR_DISTANCE_LIMIT", "32")),
        )
        return self.phase_memory_controller

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

        if len(self.actions) == 0 or self._closed_loop_should_replan():
            if self._is_phase_memory_mode():
                controller = self._ensure_phase_memory_controller(
                    mask_dict,
                    mask_id_to_sim_name,
                    point_cloud_dict,
                    lang_goal,
                )
                action = controller.next_action(
                    observation_summary=self._low_dim_state_summary(obs),
                    front_rgb_path=self.vl_front_rgb_path,
                    overhead_rgb_path=self.vl_overhead_rgb_path,
                    generate_text=self._generate_text,
                )
                return json.dumps(
                    {
                        "phase_memory_mode": True,
                        "next_action_7d": action,
                    }
                )

            if self._is_phase_anchor_replay():
                actions = self._load_phase_anchor_actions_7d()
                return json.dumps(actions)

            user_prompt = self.handler.get_user_prompt_ranking(mask_dict, mask_id_to_sim_name, point_cloud_dict, custom_num_demos=self.demo_num_per_icl, taskname=lang_goal, image_path=self.front_rgb_path, seed=self.seed, ranking_metric=self.ranking_method)   
            if self._closed_loop_should_replan():
                user_prompt += self._closed_loop_prompt_suffix(step, obs)

            print(self.SYSTEM_PROMPT) 

            print()

            print(user_prompt)

            messages = self._build_final_messages(user_prompt)


            ########################### vllm local deploy #####################################
            output_text = self._generate_text(messages, max_tokens=256)

            print(f"Prediction:", output_text)
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
            self._last_discrete_actions = actions
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
        if self._is_phase_memory_mode():
            output_text = self._preprocess(observation, step, **kwargs)
            output = self._postprocess(output_text)
            if len(output) == 0:
                output = self._postprocess(json.dumps({"next_action_7d": [50, 50, 80, 0, 36, 0, 1]}))
            continuous_action = output[0]
        elif self._closed_loop_should_replan():
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
            output_text = self._preprocess(observation, step, **kwargs)
            if len(self.actions) == 0:
                output = self._postprocess(output_text)
                self.actions = output
            
            continuous_action = self.actions.pop(0)

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
        self.closed_loop_replans = 0
        self.closed_loop_history = []
        self._last_discrete_actions = []
        self.phase_memory_controller = None

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
