# 🚀 AGNOSTOS: Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization

<div align="center">
  <a href="https://jiaming-zhou.github.io/AGNOSTOS/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/pdf/2505.15660"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/Jiaming2472/AGNOSTOS"><img src="https://img.shields.io/badge/🤗-Data-yellow.svg" alt="Hugging Face Data"></a>
  <a href="https://huggingface.co/Jiaming2472/X-ICM"><img src="https://img.shields.io/badge/🤗-Model-orange.svg" alt="Hugging Face Model"></a>
  <a href="https://youtu.be/5MKlijK1gKI"><img src="https://img.shields.io/badge/Video-YouTube-red.svg" alt="Video"></a>
</div>

<div align="center">
  <a href="https://jiaming-zhou.github.io/">Jiaming Zhou</a><sup>1</sup>, <a href="https://yipko.com/about/">Ke Ye</a><sup>1</sup>, <a href="https://www.jiayi-liu.cn/">Jiayi Liu</a><sup>1</sup>, <a href="https://teleema.github.io/">Teli Ma</a><sup>1</sup>, <a href="https://zifanw.notion.site/">Zifan Wang</a><sup>1</sup>, <a href="https://github.com/ConnerQiu">Ronghe Qiu</a><sup>1</sup>, <a href="https://kunyulin.github.io/">Kun-Yu Lin</a><sup>2</sup>, <a href="https://lawliet-zzl.github.io/">Zhilin Zhao</a><sup>3</sup>, <a href="https://junweiliang.me/">Junwei Liang</a><sup>1,4</sup><br>
  <sup>1</sup>HKUST (Guangzhou), <sup>2</sup>HKU, <sup>3</sup>SYSU, <sup>4</sup>HKUST
</div>


## 📝 Overview
The project introduces AGNOSTOS, a simulation manipulation benchmark designed to rigorously evaluate cross-task zero-shot generalization of Vision-Language-Action models, and proposes Cross-Task In-Context Manipulation (X-ICM), a method that significantly improves cross-task generalization capabilities.

<img src="docs/agnostos_benchmark.gif" width="100%">

## 🔧 Environment Setup

### 🐳 Option 1: Using Docker
Please refer to [INSTALL_docker.md](./docs/INSTALL_docker.md) to initialize your environment.

### ⚙️ Option 2: Local Installation

For simplified installation using modern package management, we recommend **Pixi**. 
Install it via the [official guide](https://pixi.sh/latest/#installation), and you can set up dependencies with minimal commands:

```bash
git clone https://github.com/jiaming-zhou/X-ICM.git && cd X-ICM
pixi shell  # Install dependencies and enter virtual environment
pixi run setup_env  # Install additional system dependencies (like xvfb, CoppeliaSim and flash-attention etc.)
```

For more options, run `pixi run --list`.

⚠️ **Important**: You need to install **CUDA 12.4** before running the commands above.

💡 **For mainland China users**: In `pixi.toml`, comment out the default lines and uncomment the mirror lines in `[workspace]` and `[pypi-options]` sections for faster installation.


## 📊 AGNOSTOS Benchmark Data
The benchmark consists of two parts. To download the data, use the following Pixi tasks:

```bash
pixi run get_seen_tasks  # Downloads and extracts 18 seen tasks (140G)
pixi run get_unseen_tasks  # Downloads and extracts 23 unseen tasks (20.2GB)
```

Data will be placed in the `data/` directory. For manual download instructions, see [MANUAL_DATA_DOWNLOAD.md](./docs/MANUAL_DATA_DOWNLOAD.md).

## 🤖 X-ICM Method

### 1️⃣ Model Download
To download the pre-trained dynamics diffusion model, run:

```bash
pixi run get_model
```

The model will be extracted to `data/dynamics_diffusion/`. For manual download instructions, see [MANUAL_DATA_DOWNLOAD.md](./docs/MANUAL_DATA_DOWNLOAD.md).

### 2️⃣ Evaluation

<details>
<summary><b>Parameters (Click to expand)</b></summary>

```bash
### set seed numbers for different runs
seeds: [example: "0,99"]
### set the number of rollouts for each run
episodes: [example: 25]
### set the method of LLM
modelname: [example: Qwen2.5.7B.instruct]
### set the number of cross-task in-context samples
num_icls: [example: 18]
### set the gpu list
gpu_ids: [example: 0,1]
### set the in-context sample selection method
ranking_method: [example: "lang_vis.out"]
```

</details>

For dynamics-guided in-context manipulation:
```bash
pixi run eval_xicm "0,99" 25 Qwen2.5.7B.instruct 18 0,1 "lang_vis.out"
```

For random selection of cross-task samples:
```bash
pixi run eval_xicm "0,99" 25 Qwen2.5.7B.instruct 18 0,1 "random"
```

After testing, use [`gather_score.py`](./gather_score.py) to collect and analyze results.

💡 **Note**: Download required models (Stable-Diffusion, Qwen-LLM) from HuggingFace and configure paths in [`main.py`](./main.py#L132) and [`rlbench_inference_dynamics_diffusion.py`](./rlbench_inference_dynamics_diffusion.py#L136).

## 🎯 Benchmarking Results over all 23 unseen tasks
We provide the testing results of our `X-ICM (7B)` and `X-ICM (72B)` models under the sub-folder `logs/`.
- X-ICM (7B) achieves 23.5% average success rate and X-ICM (72B) achieves 30.1% average success rate, both versions outperform all existing VLA models;
- X-ICM (7B) fails on only two tasks, while X-ICM (72B) succeeds on all tasks;

## 🔬 Benchmarking Your VLA Model on AGNOSTOS

### 1️⃣ Fine-tuning
Due to the embodiment gap, existing VLA models need to be fine-tuned on RLBench data. 

Please follow your VLA model's fine-tuning guidelines to fine-tune your models on our 18 seen tasks, and then test the models on our 23 unseen tasks.

<details>
<summary><b>Example: Fine-tuning Qwen2.5-VL (Click to expand)</b></summary>

We provide a complete fine-tuning pipeline for **Qwen2.5-VL**, using the [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) framework:

```bash
# Download seen tasks data (18 tasks, ~140GB)
pixi run get_seen_tasks

# Basic training with default parameters
bash scripts/train_qwen2.5VL_sft.sh

# Custom training: bash scripts/train_qwen2.5VL_sft.sh [LR_LLM] [LR_VISION] [LR_MERGER] [EPOCHS] [BS] [GPUS]
# LR_LLM: LLM learning rate (default: 1e-4)
# LR_VISION: Vision tower learning rate (default: 2e-5)
# LR_MERGER: MLP learning rate (default: 1e-5)
# EPOCHS: Training epochs (default: 20)
# BS: Batch size per GPU (default: 128)
# GPUS: GPU IDs (default: 0,1,2,3,4,5,6,7)

# Example with custom parameters:
bash scripts/train_qwen2.5VL_sft.sh 1e-4 2e-5 1e-5 20 128 0,1,2,3,4,5,6,7
```

</details>

### 2️⃣ Testing Fine-tuned VLA Models

#### For Generic VLA Models
Modify the [`custom_agent.py`](./custom_agent.py) file:
1. Load your VLA model in the [`load_weights`](./custom_agent.py#L116) function;
2. Implement VLA model inference in the [`_inference`](./custom_agent.py#L21) function, including input construction and output format conversion;
3. Run the evaluation:
    ```bash
    bash scripts/eval_CustomModel.sh seeds episodes gpu_ids
    ```

    Example:
    ```bash
    bash scripts/eval_CustomModel.sh "0,99" 25 0,1
    ```

💡 **Note**: Different VLA models may require different input image sizes (default is 256x256). Please modify [`IMAGE_SIZE`](./main_custom.py#L32) in [`main_custom.py`](./main_custom.py) accordingly.

<details>
<summary><b>Example: Testing Qwen2.5-VL on AGNOSTOS (Click to expand)</b></summary>

After fine-tuning, evaluate your Qwen2.5-VL model on AGNOSTOS:

```bash
# bash scripts/eval_qwen2.5VL_sft.sh [MODE] [CHECKPOINT] [SEEDS] [EPISODES] [GPU_ID] [H_LEN] [T_LEN] [STEPS] [START] [NUM]
# MODE: ood (23 unseen tasks) or withintask (18 seen tasks)
# CHECKPOINT: Path to model checkpoint (e.g., outputs/checkpoint-1860)
# SEEDS: Random seeds (e.g., 0,1,2)
# EPISODES: Number of rollouts per task
# GPU_ID: GPU device ID
# H_LEN: History length | T_LEN: Target length | STEPS: Episode steps
# START: Start task index | NUM: Number of tasks to evaluate

# OOD evaluation on all 23 unseen tasks
bash scripts/eval_qwen2.5VL_sft.sh ood outputs/checkpoint-1860 0 25 0 1 1 25 0 23

# WithinTask evaluation on all 18 seen tasks
bash scripts/eval_qwen2.5VL_sft.sh withintask outputs/checkpoint-1860 0 25 0 1 1 25 0 18

# OOD with multiple seeds
bash scripts/eval_qwen2.5VL_sft.sh ood outputs/checkpoint-1860 0,1,2 25 0 1 1 25 0 23
```

</details>



## 📂 Repository Structure

<details>
<summary><b>Directory Overview (Click to expand)</b></summary>

```
X-ICM/
├── data/                          # Dataset and models
│   ├── seen_tasks/                # 18 seen training tasks (~140GB)
│   ├── unseen_tasks/              # 23 unseen evaluation tasks (~20.2GB)
│   └── dynamics_diffusion/        # Pre-trained dynamics diffusion model
├── scripts/                       # Training and evaluation scripts
│   ├── train_qwen2.5VL_sft.sh    # Qwen2.5-VL fine-tuning script
│   ├── eval_qwen2.5VL_sft.sh     # Qwen2.5-VL evaluation script
│   └── eval_XICM.sh              # X-ICM evaluation script
├── qwen2vl_finetune/              # Qwen2.5-VL fine-tuning module
├── RLBench/                       
├── YARR/                          
├── PyRep/                         
├── CoppeliaSim/                   # CoppeliaSim simulator
├── main.py                        # X-ICM inference entry point
├── main_custom.py                 # Generic VLA model evaluation
├── custom_agent.py                # Custom VLA agent template
├── rlbench_inference_dynamics_diffusion.py
└── gather_score.py                # Result aggregation script
```

</details>

## 🙏 Acknowledgments
This repository is built upon the [RoboPrompt](https://github.com/davidyyd/roboprompt) and [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune). Some resources from [RVT](https://github.com/NVlabs/RVT) and [RLBench](https://github.com/stepjam/RLBench) are used in this work.

## 📄 Citation
If you find our work helpful to your research, please kindly give us a star and cite our paper.
```bibtex
@inproceedings{
zhou2025exploring,
    title={Exploring the Limits of Vision-Language-Action Manipulation in Cross-task Generalization},
    author={Jiaming Zhou and Ke Ye and Jiayi Liu and Teli Ma and Zifan Wang and Ronghe Qiu and Kun-Yu Lin and Zhilin Zhao and Junwei Liang},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=h6xQClTm4W}
}
```
