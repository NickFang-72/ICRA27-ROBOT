# X-ICM / AGNOSTOS Setup Status

Date: 2026-06-16

## Local Environment

Conda environment:

```bash
conda activate zero-shot
```

Python:

```text
Python 3.10.19
```

Repository:

```text
/Users/nicholas/Documents/ICRA27 ROBOT/X-ICM
```

The repository was downloaded from the GitHub source archive because `git clone` stalled during pack indexing.

## Installed Successfully In `zero-shot`

Core Python/model stack:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `transformers==4.51.3`
- `diffusers==0.33.1`
- `qwen-vl-utils==0.0.11`
- `accelerate==1.7.0`
- `datasets==3.5.1`
- `peft==0.15.2`
- `trl==0.17.0`
- `openai==1.82.0`
- `clip==1.0`
- `hydra-core==1.3.2`
- `omegaconf==2.3.0`
- `scipy==1.15.3`
- `pandas==1.4.1`
- `opencv-python`
- `matplotlib`
- `tensorboard`
- `tensorboardX`

Local editable packages:

- `RLBench`
- `YARR`

Smoke test passed for:

```text
torch 2.6.0
torchvision 0.21.0
transformers 4.51.3
diffusers 0.33.1
qwen_vl_utils
yarr
clip
openai
```

Torch reports Apple MPS is available.

`pip check` reports no broken Python package requirements.

## Not Installed Locally

These are required by the official X-ICM runtime, but are Linux/CUDA/simulator-specific:

- `pyrep`
- `vllm`
- `flash_attn`
- `bitsandbytes`
- `deepspeed`
- Ubuntu CoppeliaSim
- `xvfb`
- CUDA 12.4

The official `pixi.toml` declares:

```text
platforms = ["linux-64"]
cuda = "12.4"
python = "3.11.12"
torch = CUDA 12.4 wheel
```

So full X-ICM evaluation cannot run natively on this macOS arm64 machine.

## Data / Model Artifacts

Current disk space:

```text
12 GiB available
```

Required artifacts:

- Dynamics diffusion model: about 10.4 GB tar before extraction.
- Unseen AGNOSTOS tasks: about 20.2 GB tar before extraction.
- Seen AGNOSTOS tasks: about 140 GB split tar before extraction.

Because the disk has only 12 GiB free, the benchmark/model artifacts were not downloaded.

The repo currently contains only the small included:

```text
data/train.json
```

## Practical Next Step

For full AGNOSTOS/X-ICM evaluation, use either:

1. A Linux CUDA 12.4 machine with enough storage, preferably 250+ GB free.
2. The official Docker image from the repo docs:

```bash
docker pull yipko/x-icm:snapshot-20251031
```

Then download:

```bash
pixi run get_model
pixi run get_unseen_tasks
pixi run get_seen_tasks
```

For our geometry/affordance retrieval experiment on this Mac, we can still begin with the repo code, `data/train.json`, and offline descriptor/retrieval scripts that do not require PyRep, CoppeliaSim, CUDA, or simulator rollouts.
