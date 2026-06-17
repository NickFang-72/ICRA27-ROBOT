from typing import List
import re
from yarr.agents.agent import Agent, Summary, ActResult
import json
import numpy as np
from PIL import Image
import os
from utils import SCENE_BOUNDS, ROTATION_RESOLUTION, discrete_euler_to_quaternion, CAMERAS
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from vllm import LLM, SamplingParams 
from qwen_vl_utils import process_vision_info

from utils import normalize_quaternion, quaternion_to_discrete_euler, point_to_voxel_index, convert_to_euler
from collections import deque



TASK_MAPPING = {
    "close_jar": {
        "jar_lid0": "lid",
        "jar0": "jar",
    },
    "open_drawer": {
        "drawer_bottom": "drawer",
    },
    "slide_block_to_color_target": {
        "target1": "target",
        "block": "block"
    },
    "sweep_to_dustpan_of_size": {
        "dustpan_tall": "dustpan",
        "broom_holder": "broom holder"
    },
    "meat_off_grill": {
        "chicken_visual": "chicken",
        "grill_visual": "grill"
    },
    "turn_tap": {
        "tap_left_visual": "left tap",
        "tap_right_visual": "right tap"
    },
    "put_item_in_drawer": {
        "item": "item",
        "drawer_frame": "drawer"
    },
    "stack_blocks": {
        "stack_blocks_target0": "first block",
        "stack_blocks_target1": "second block",
        "stack_blocks_target2": "third block",
        "stack_blocks_target3": "fourth block",
        "stack_blocks_target_plane": "plane",
    },
    "light_bulb_in": {
        "bulb0": "blub",
        "lamp_screw": "lamp screw",
    },
    "put_money_in_safe": {
        "dollar_stack": "money",
        "safe_body": "shelf",
    },
    "place_wine_at_rack_location": {
        "wine_bottle_visual": "wine",
        "rack_top_visual": "rack",
    },
    "put_groceries_in_cupboard": {
        "cupboard": "cupboard",
        "crackers_visual": "cracker",
    },
    "place_shape_in_shape_sorter": {
        "cube": "cube",
        "shape_sorter_visual": "shape sorter",
    },
    "push_buttons": {
        "target_button_wrap0": "button",
    },
    "insert_onto_square_peg": {
        "square_ring": "ring",
        "pillar0": "first spok",
        "pillar1": "second spok",
        "pillar2": "third spok"
    },
    "stack_cups": {
        "cup1_visual": "first cup",
        "cup2_visual": "second cup",
        "cup3_visual": "third cup",
    },
    "place_cups": {
        "mug_visual1": "first cup",
        "mug_visual0": "second cup",
        "mug_visual2": "third cup",
        "mug_visual3": "forth cup",
        "place_cups_holder_spoke0": "first holder",
        "place_cups_holder_spoke1": "second holder",
        "place_cups_holder_spoke2": "third holder"
    },
    "reach_and_drag": {
        "stick": "stick",
        "cube": "cube"
    },



    "basketball_in_hoop": {
        "basket_ball_hoop_visual": "hoop",
        "ball": "ball"
    },
    "scoop_with_spatula": {
        "Cuboid": "cube",
        "spatula_visual": "spatula"
    },
    "straighten_rope": {
        "head": "rope head",
        "head_target": "rope head target",
        "head_tail": "rope head tail",
        "tail": "rope tail"
    },
    "turn_oven_on": {
        "oven_door": "oven door",
        "oven_knob_8": "first oven knob",
        "oven_knob_9": "second oven knob"
    },
    "beat_the_buzz": {
        "Cuboid": "cube",
        "wand_visual": "pole",
        "wand_visual_sub": "pole head"
    },
    "water_plants": {
        "waterer_visual": "waterer",
        "plant_visual": "plant",
        "base_visual": "waterer base"
    },
    "unplug_charger": {
        "charger_visual": "charger",
        "task_wall": "wall"
    },
    "phone_on_base": {
        "phone_visual": "phone",
        "phone_case_visual": "phone case"
    },
    "toilet_seat_down": {
        "toilet_seat_up_toilet": "toilet seat up_toilet",
        "toilet_seat_up_seat": "toilet seat up_seat",
        "toilet": "toilet"
    },
    "lamp_off": {
        "push_button_target": "button",
        "target_button_topPlate": "button topPlate",
        "target_button_wrap": "button wrap"
    },
    "lamp_on": {
        "push_button_target": "button",
        "target_button_topPlate": "button topPlate",
        "target_button_wrap": "button wrap"
    },
    "put_books_on_bookshelf": {
        "book0_visual": "first book",
        "book1_visual": "second book",
        "book2_visual": "third book",
        "bookshelf_visual": "bookshelf"
    },
    "put_umbrella_in_umbrella_stand": {
        "umbrella_visual": "umbrella",
        "stand_visual": "umbrella stand"
    },
    "open_grill": {
        "grill_visual": "grill",
        "lid_visual": "lid",
        "handle_visual": "handle"
    },
    "put_rubbish_in_bin": {
        "rubbish_visual": "rubbish",
        "bin_visual": "bin"
    },
    "take_usb_out_of_computer": {
        "computer_visual": "computer",
        "usb_visual": "usb"
    },
    "take_lid_off_saucepan": {
        "saucepan_visual": "saucepan",
        "saucepan_lid_visual": "saucepan lid"
    },
    "take_plate_off_colored_dish_rack": {
        "plate_visual": "plate",
        "dish_rack_pillar0": "first dish rack",
        "dish_rack_pillar1": "second dish rack"
    },
    "close_fridge": {
        "fridge_base_visual": "fridge base",
        "door_top_visual": "fridge top door",
        "door_bottom_visual": "fridge bottom door"
    },
    "close_microwave": {
        "microwave_door": "microwave door",
        "microwave_frame_vis": "microwave frame"
    },
    "close_laptop_lid": {
        "lid_visual": "lid",
        "laptop_holder": "laptop holder",
        "base_visual": "laptop base"
    },
    "put_toilet_roll_on_stand": {
        "toilet_roll_visual": "toilet roll",
        "holder_visual": "holder",
        "stand_base": "stand_base"
    },
    "put_knife_on_chopping_board": {
        "knife_visual": "knife",
        "chopping_board_visual": "chopping board"
    }
}

def form_obs(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict):
    
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
    
    return str(real_name_to_avg_coord)




class QwenSFTAgent(Agent):
    def __init__(self, task_name, start_seed=0, history_length=1, target_length=1):
        self.episode_id = -1
        self.device = 'cuda'
        self.task_name = task_name
        self.start_seed = start_seed
        self.history_length = history_length
        self.target_length = target_length

        self.obtained_init_action = False

        self.sim_name_to_real_name = TASK_MAPPING[task_name]

    def _preprocess(self, obs, step, **kwargs):
        rgb_dict = {}
        mask_id_to_sim_name = {}
        mask_dict = {}
        point_cloud_dict = {}
        lang_goal = kwargs['lang_goal']

        gripper_pose=obs['gripper_pose'][0,0].cpu().numpy()
        gripper_open=obs['gripper_open'][0,0].cpu().numpy()
        
        quat = normalize_quaternion(gripper_pose[3:])
        if quat[-1] < 0:
            quat = -quat
        disc_rot = quaternion_to_discrete_euler(quat)
        trans_indicies = []

        index = point_to_voxel_index(gripper_pose[:3])
        trans_indicies.extend(index.tolist())

        rot_and_grip_indicies = disc_rot.tolist()
        rot_and_grip_indicies.extend([int(gripper_open)])
        current_eef = trans_indicies + rot_and_grip_indicies

        if not self.obtained_init_action:
            self.history_actions = deque([current_eef] * self.history_length, maxlen=self.history_length)
            self.obtained_init_action = True
        else:
            self.history_actions.append(current_eef)
            assert len(self.history_actions) == self.history_length
        

        rgb_dir = os.path.join(self.savedir, 'rgb_dir')
        os.makedirs(rgb_dir, exist_ok=True)

        front_rgb_img = obs['front_rgb']
        front_rgb_img=front_rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        front_rgb_img = np.clip((front_rgb_img).astype(np.uint8), 0, 255)
        front_rgb_img = Image.fromarray(front_rgb_img)
        front_rgb_img.save(os.path.join(rgb_dir, 'front_rgb.png'))
        self.front_rgb_path=os.path.join(rgb_dir, 'front_rgb.png')

        # overhead_rgb_img = obs['overhead_rgb']
        # overhead_rgb_img=overhead_rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        # overhead_rgb_img = np.clip((overhead_rgb_img).astype(np.uint8), 0, 255)
        # overhead_rgb_img = Image.fromarray(overhead_rgb_img)
        # overhead_rgb_img.save(os.path.join(rgb_dir, 'overhead_rgb.png'))
        # self.overhead_rgb_path=os.path.join(rgb_dir, 'overhead_rgb.png')

        wrist_rgb_img = obs['wrist_rgb']
        wrist_rgb_img=wrist_rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
        wrist_rgb_img = np.clip((wrist_rgb_img).astype(np.uint8), 0, 255)
        wrist_rgb_img = Image.fromarray(wrist_rgb_img)
        wrist_rgb_img.save(os.path.join(rgb_dir, 'wrist_rgb.png'))
        self.wrist_rgb_path=os.path.join(rgb_dir, 'wrist_rgb.png')


        if len(self.actions) == 0:

            for camera in CAMERAS:
                # rgb_img = obs[f'{camera}_rgb']
                # rgb_img = rgb_img.squeeze().permute(1, 2, 0).cpu().numpy()
                # rgb_img = np.clip(((rgb_img + 1.0) / 2 * 255).astype(np.uint8), 0, 255)

                # rgb_dict[camera] = rgb_img

                mask_id_to_sim_name.update(kwargs["mapping_dict"][f"{camera}_mask_id_to_name"])

                mask = obs[f'{camera}_mask']
                mask = mask.squeeze().cpu().numpy() 

                mask_dict[camera] = mask

                point_cloud = obs[f'{camera}_point_cloud'].cpu().squeeze().permute(1, 2, 0).numpy()
                point_cloud_dict[camera] = point_cloud

            mask_id_to_real_name = {mask_id: self.sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                            if name in self.sim_name_to_real_name}
            object_info = form_obs(mask_dict, mask_id_to_real_name, point_cloud_dict)
            print("please check object_info!!!!!!!!", object_info)


            # prompt_text = f"You are a Franka robot arm and use 7-dof end-effector control. The task description is '{lang_goal}'.\n"
            # prompt_text += "I give you the current multi-view observations <image>\n<image>\n, "
            # prompt_text += f"and the current 7-dof action {list(self.history_actions)}.\n"
            # prompt_text += f"What should be the next action for the robot?"
            
            # prompt_text = f"The task description is '{lang_goal}'.\nAnd the current multi-view observations are <image>\n<image>\n. What should be the next action?"
            # prompt_text = f"The task description is '{lang_goal}'.\nI give you the current multi-view observations are <image>\n<image>\n, and the current 7-dof action {list(self.history_actions)}.\nWhat should be the next action for the robot?"
            # prompt_text = f"The task description is '{lang_goal}'.\nI give you the current multi-view observations are <image>\n<image>\n, and the current 7-dof action {list(self.history_actions)}.\nWhat should be the next action for the robot?"
            prompt_text = f"The task description is '{lang_goal}'.\nI give you the current multi-view observations are <image>\n<image>\n, the current object names and 3D coordinates {object_info}, and the current 7-dof action {list(self.history_actions)}.\nWhat should be the next action for the robot?"
            


            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.front_rgb_path},
                        {"type": "image", "image": self.wrist_rgb_path},
                        {"type": "text", "text": prompt_text},
                    ]
                }
            ]


            print(messages)
            


            prompt = self.components["processor"].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            
            
            # ########################## vllm local deploy #####################################
            # mm_data = {}
            # if image_inputs is not None:
            #     mm_data["image"] = image_inputs
            # if video_inputs is not None:
            #     mm_data["video"] = video_inputs

            # llm_inputs = {
            #     "prompt": prompt,
            #     "multi_modal_data": mm_data,
            # }
            
            # # llm_inputs = {
            # #     "prompt": prompt
            # # }
            

            # sampling_params = SamplingParams(
            #     temperature=0.1,
            #     top_p=0.001,
            #     repetition_penalty=1.05,
            #     max_tokens=256,
            #     stop_token_ids=[],
            # )

            # outputs = self.components["mllm"].generate([llm_inputs], sampling_params=sampling_params)
            # output_text = outputs[0].outputs[0].text




            ########### Transformers  Chat Template #####################################
            inputs = self.components["processor"](
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.components["mllm"].device)

            # Inference: Generation of the output
            generated_ids = self.components["mllm"].generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.components["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

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

    def _postprocess(self, output_text):
        try:
            # # 修复尾部多余括号的输入
            # output_text = str(output_text)
            # if output_text[:3]=="['[" and output_text[-2:]=="']":
            #     output_text = output_text.strip()[2:-2]+", 1]]"
            # elif output_text.count('[') > output_text.count(']'):
            #     output_text = output_text.strip() + "]"
            # elif output_text.count('[') < output_text.count(']'):
            #     output_text = output_text.strip().rstrip(']')

            # print("fixed output: ", output_text)

            # # 尝试正则匹配并解析合法 JSON 格式
            # regex = r'^json(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*$'
            # match = re.search(regex, output_text)
            # if match:
            #     actions = np.array(json.loads(match.group(1)))
            # else:
            #     regex = r'^(\s*\[\s*(?:\[(?:\d+\s*,\s*){6}\d+\]\s*,\s*)*\[(?:\d+\s*,\s*){6}\d+\]\s*\])\s*$'
            #     match = re.search(regex, output_text)
            #     if match:
            #         actions = np.array(json.loads(match.group(1)))
            #     else:
            #         try:
            #             # 尝试直接解析原始输入
            #             actions = np.array(json.loads(output_text))
            #         except Exception:
            #             # 处理缺少外层括号的输入
            #             fixed_text = f"[{output_text.strip().rstrip(',')}]"
            #             actions = np.array(json.loads(fixed_text))
                            
            actions = self.re_match(str(output_text))
            print("parsed actions: ", actions)
        except Exception as e:
            # 默认动作列表
            actions = [[57, 49, 87, 0, 39, 0, 1] for _ in range(26)]
            print(e)
            print('Error when parsing actions. Falling back to default.')
        
        # 确保 actions 的形状是二维的
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

    def load_weights(self, savedir: str, components={}):
        # no weight to load
        self.savedir = savedir
        self.components=components

        return

    def build(self, training: bool, device=None):
        return

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}
    
    def update_summaries(self) -> List[Summary]:
        return []

    def save_weights(self, savedir: str):
        return