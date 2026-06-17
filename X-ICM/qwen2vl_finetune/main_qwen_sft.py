import gc
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import hydra
from omegaconf import DictConfig, OmegaConf
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from pyrep.const import RenderMode

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from vllm import LLM, SamplingParams 
from qwen_vl_utils import process_vision_info

from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.rollout_generator import RolloutGenerator

from utils import CAMERAS, SCENE_BOUNDS, ROTATION_RESOLUTION, VOXEL_SIZE, IMAGE_SIZE
import multiprocessing
import torch
from torch.multiprocessing import Manager
torch.multiprocessing.set_sharing_strategy('file_system')

from qwen_sft_agent import QwenSFTAgent
import sys
PROJECT_ROOT=os.path.dirname(os.path.abspath(__file__))


IMAGE_SIZE = [256, 256]
# IMAGE_SIZE = [128, 128]

def create_obs_config():
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=True,
        depth=False,
        image_size=IMAGE_SIZE,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in CAMERAS:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config

def eval_seed(eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config,
              components) -> None:


    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    agent = QwenSFTAgent(
        eval_cfg.rlbench.task_name,
        eval_cfg.framework.start_seed,
        history_length=eval_cfg.rlbench.history_length,
        target_length=eval_cfg.rlbench.target_length
        )

    stat_accum = SimpleAccumulator(eval_video_fps=30)

    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=0,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=eval_cfg.framework.logdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=1,
        multi_task=multi_task,
        components=components)

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()
    
    env_runner.start({"task": eval_cfg.framework.logdir}, save_load_lock, writer_lock,
                              env_config, 0,
                              eval_cfg.framework.eval_save_metrics,
                              eval_cfg.cinematic_recorder)


    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()



def load_weight(savedir, deploy_type="naive_local"):
    components={}

    if deploy_type=="vllm_local":
        ################ vllm local deploy ########################
        model_name = "Qwen/Qwen2-7B-Instruct"
        if "Qwen2.7B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
        elif "Qwen2.72B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/snapshots/c867f763ef53f2ea9d9b31ee8501273dedd391eb"
        elif "Qwen2.5.7B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
        elif "Qwen2.5.14B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
        elif "Qwen2.5.32B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
        elif "Qwen2.5.72B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
        elif "Qwen2.VL.7B.instruct" in savedir:
            model_name="/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f"
        elif "Qwen2.VL.72B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2-VL-72B-Instruct/snapshots/f400120e59a6196b024298b7d09fb517f742db7d"
        elif "Qwen2.5.VL.72B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--Qwen--Qwen2-VL-72B-Instruct/snapshots/f400120e59a6196b024298b7d09fb517f742db7d"
        elif "InternLM3.8B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--internlm--internlm3-8b-instruct/snapshots/28c99415adaf61767bd1c619f4f99f308fdfd223"
        elif "Llama3.0.8B.instruct" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/Meta-Llama-3-8B-Instruct"
        elif "DeepSeek.R1.Distill.Qwen.7B" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
        elif "Ministral.8B.Instruct.2410" in savedir:
            model_name = "/remote-home/jiamingz/projects/huggingface/hub/Ministral-8B-Instruct-2410"
        else:
            model_name = savedir

        multimodel=True
        
        # multimodel=False
        # if ".VL." in savedir:
        #     multimodel=True
        
        print("loading %s"%model_name)
        # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L45
        # https://docs.vllm.ai/en/stable/dev/offline_inference/llm.html
        if multimodel:
            llm = LLM(
                model=model_name,
                limit_mm_per_prompt={"image": 20, "video": 0},
                tensor_parallel_size = torch.cuda.device_count(),
                # max_model_len=16464,
                gpu_memory_utilization=0.8,
                )
        else:
            llm = LLM(
                model=model_name,
                tensor_parallel_size = torch.cuda.device_count(),
                # max_model_len=16464,
                gpu_memory_utilization=0.8,
                trust_remote_code=True if "internlm" in model_name else False,
            )
        processor = AutoProcessor.from_pretrained(model_name,
            trust_remote_code=True if "internlm" in model_name else False)
        
        components['mllm'] = llm
        components['processor']=processor

    elif deploy_type=="naive_local":
        ################ naive local deploy ########################
        model_name = "Qwen/Qwen2-7B-Instruct"
        if "Qwen2.7B.instruct" in savedir:
            model_name = "Qwen/Qwen2-7B-Instruct"
        elif "Qwen2.72B.instruct" in savedir:
            model_name = "Qwen/Qwen2-72B-Instruct"
        elif "Qwen2.5.7B.instruct" in savedir:
            model_name = "Qwen/Qwen2.5-7B-Instruct"
        elif "Qwen2.5.14B.instruct" in savedir:
            model_name = "Qwen/Qwen2.5-14B-Instruct"
        elif "Qwen2.5.32B.instruct" in savedir:
            model_name = "Qwen/Qwen2.5-32B-Instruct"
        elif "Qwen2.5.72B.instruct" in savedir:
            model_name = "Qwen/Qwen2.5-72B-Instruct"
        elif "Qwen2.VL.7B.instruct" in savedir:
            model_name = "Qwen/Qwen2-VL-7B-Instruct"
        elif "Qwen2.VL.72B.instruct" in savedir:
            model_name = "Qwen/Qwen2-VL-72B-Instruct"
        elif "Qwen2.5.VL.72B.instruct" in savedir:
            model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
        else:
            model_name = savedir
        
        print("loading %s"%model_name)

        multimodel=True
        
        # multimodel=False
        # if ".VL." in savedir:
        #     multimodel=True
        
        if not multimodel:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            components['mllm']=model
            components['tokenizer']=tokenizer
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                # torch_dtype="auto",
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            # default processer
            # min_pixels = 16*28*28
            # max_pixels = 1280*28*28
            # processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
            processor = AutoProcessor.from_pretrained(model_name)

            components['mllm']=model
            components['processor']=processor
    else:
        pass

    return components


@hydra.main(config_name='config', config_path='.')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    # eval_cfg.rlbench.demo_path = os.path.join(PROJECT_ROOT, eval_cfg.rlbench.demo_path)
    eval_cfg.framework.logdir = os.path.join(PROJECT_ROOT, eval_cfg.framework.logdir)

    env_device = 'cuda'
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = CAMERAS
    
    obs_config = create_obs_config()      

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    tasks = eval_cfg.rlbench.tasks
    multi_task = False

    print("tasks:", tasks)

    components=load_weight(os.path.join(eval_cfg.framework.logdir,
                        eval_cfg.method.name))

    for task_id, task in enumerate(tasks):

        eval_cfg.rlbench.task_name=task

        logdir = os.path.join(eval_cfg.framework.logdir,
                            eval_cfg.method.name, "eval",
                            eval_cfg.rlbench.task_name,
                            'seed%d' % start_seed)

        print(f"evaluating: task_id: {task_id}, task_name: {task}")
        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = task_file_to_task_class(task)

        env_config = (task_class,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      True,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)

        logging.info('Evaluating seed %d.' % start_seed)
        eval_seed(eval_cfg,
                    logdir,
                    eval_cfg.rlbench.cameras,
                    env_device,
                    multi_task, start_seed,
                    env_config,
                    components)

if __name__ == '__main__':
    main()
