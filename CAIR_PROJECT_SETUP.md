# CAIR Project Setup Runbook

This runbook describes how to set up a new CAIR project, connect it to GitHub under `NickFang-72`, and run training in a reproducible way. It is written for this project session, but the same pattern can be reused for future projects.

## Current Project Values

Use these values for this repository:

```bash
export PROJECT_NAME=ICRA27-ROBOT
export GITHUB_OWNER=NickFang-72
export GITHUB_REPO=git@github.com:NickFang-72/ICRA27-ROBOT.git
export CAIR_PROJECT_ROOT=/data/yf23/projects/ICRA27-ROBOT
export CAIR_ENV_NAME=icra27-robot
export CAIR_ENV=/data/yf23/conda/envs/$CAIR_ENV_NAME
export CAIR_DATA=/data/yf23/datasets/ICRA27-ROBOT
export CAIR_RUNS=/data/yf23/runs/ICRA27-ROBOT
export CAIR_CHECKPOINTS=/data/yf23/checkpoints/ICRA27-ROBOT
```

For a new future project, replace `PROJECT_NAME`, `GITHUB_REPO`, `CAIR_ENV_NAME`, and the matching `/data/yf23/...` paths.

Run this block in every fresh CAIR shell or tmux session before using commands that reference these variables. SSH sessions do not remember exports from previous logins.

## 1. Connect To CAIR

From this Mac, Tailscale must be connected. The configured SSH route is:

```text
Mac -> Thor -> CAIR
```

Use these aliases:

```bash
ssh thor
ssh cair
```

Verify the GPU host:

```bash
ssh cair 'hostname; whoami; nvidia-smi -L'
```

Expected CAIR account:

```text
yf23
```

Expected GPU host:

```text
dclap-p1746-cair.abudhabi.nyu.edu
```

Do not store passwords in project files, Markdown notes, shell history, Git commits, or training configs.

## 2. Create The CAIR Project Folders

Log in to CAIR:

```bash
ssh cair
```

Create a clean project layout under `/data/yf23`:

```bash
export PROJECT_NAME=ICRA27-ROBOT
export CAIR_PROJECT_ROOT=/data/yf23/projects/$PROJECT_NAME
export CAIR_ENV_NAME=icra27-robot
export CAIR_ENV=/data/yf23/conda/envs/$CAIR_ENV_NAME
export CAIR_DATA=/data/yf23/datasets/$PROJECT_NAME
export CAIR_RUNS=/data/yf23/runs/$PROJECT_NAME
export CAIR_CHECKPOINTS=/data/yf23/checkpoints/$PROJECT_NAME

mkdir -p /data/yf23/projects
mkdir -p /data/yf23/conda/envs
mkdir -p "$CAIR_DATA" "$CAIR_RUNS" "$CAIR_CHECKPOINTS"
```

Keep source code in `/data/yf23/projects`, datasets in `/data/yf23/datasets`, run logs in `/data/yf23/runs`, and model outputs in `/data/yf23/checkpoints`.

## 3. Create The Conda Environment On CAIR

Set the project variables in the CAIR shell:

```bash
export PROJECT_NAME=ICRA27-ROBOT
export CAIR_ENV_NAME=icra27-robot
export CAIR_ENV=/data/yf23/conda/envs/$CAIR_ENV_NAME
```

Load conda if it is already installed:

```bash
if [ -f /data/yf23/miniforge3/etc/profile.d/conda.sh ]; then
  source /data/yf23/miniforge3/etc/profile.d/conda.sh
elif [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
  source ~/miniforge3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
else
  echo "Conda was not found. Install Miniforge under /data/yf23/miniforge3 first."
  return 1 2>/dev/null || exit 1
fi
```

If conda is not installed, install Miniforge into `/data/yf23`:

```bash
cd /tmp
curl -L -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /data/yf23/miniforge3
source /data/yf23/miniforge3/etc/profile.d/conda.sh
```

Create and activate the project environment:

```bash
conda create -p "$CAIR_ENV" python=3.10 -y
conda activate "$CAIR_ENV"
python -m pip install --upgrade pip setuptools wheel
```

Use `conda activate "$CAIR_ENV"` every time before installing dependencies or running training.

## 4. Link The Project Code To GitHub

For this repository, the GitHub remote is:

```bash
git@github.com:NickFang-72/ICRA27-ROBOT.git
```

For a new future repository:

1. Open `https://github.com/new`.
2. Owner: `NickFang-72`.
3. Repository name: use the project name.
4. Choose private or public as needed.
5. If the local repository already has code, do not initialize the GitHub repo with a README, license, or `.gitignore`.

From the local Mac project folder, link the code:

```bash
cd "/Users/nicholas/Documents/ICRA27 ROBOT"
git remote -v
git remote set-url origin "$GITHUB_REPO"
```

If there is no `origin` yet:

```bash
git remote add origin "$GITHUB_REPO"
```

Push the first branch:

```bash
git push -u origin "$(git branch --show-current)"
```

On CAIR, clone the GitHub repo into `/data/yf23/projects`:

```bash
ssh cair
export PROJECT_NAME=ICRA27-ROBOT
export GITHUB_REPO=git@github.com:NickFang-72/ICRA27-ROBOT.git
export CAIR_PROJECT_ROOT=/data/yf23/projects/$PROJECT_NAME
mkdir -p /data/yf23/projects
cd /data/yf23/projects
git clone "$GITHUB_REPO" "$PROJECT_NAME"
cd "$CAIR_PROJECT_ROOT"
git status
```

If the project already exists on CAIR:

```bash
ssh cair
export PROJECT_NAME=ICRA27-ROBOT
export CAIR_PROJECT_ROOT=/data/yf23/projects/$PROJECT_NAME
cd "$CAIR_PROJECT_ROOT"
git fetch origin
git pull --ff-only
```

## 5. Install This Project On CAIR

Activate the environment:

```bash
export PROJECT_NAME=ICRA27-ROBOT
export CAIR_PROJECT_ROOT=/data/yf23/projects/$PROJECT_NAME
export CAIR_ENV_NAME=icra27-robot
export CAIR_ENV=/data/yf23/conda/envs/$CAIR_ENV_NAME
export CAIR_DATA=/data/yf23/datasets/$PROJECT_NAME
export CAIR_RUNS=/data/yf23/runs/$PROJECT_NAME
export CAIR_CHECKPOINTS=/data/yf23/checkpoints/$PROJECT_NAME

source /data/yf23/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$CAIR_ENV"
cd "$CAIR_PROJECT_ROOT"
```

This runbook uses conda because the CAIR workflow for this project is conda-based. The `X-ICM` subtree also includes `pixi.toml` and `pixi.lock`; if a run needs to reproduce that locked environment exactly, use pixi inside `X-ICM` or translate the pixi lock into the CAIR conda environment before training. Keep one environment strategy per run.

PyRep checks `COPPELIASIM_ROOT` during installation, so configure CoppeliaSim before installing `X-ICM/PyRep` or `X-ICM/RLBench`:

```bash
export COPPELIASIM_ROOT=/data/yf23/tools/CoppeliaSim_Edu
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT${QT_QPA_PLATFORM_PLUGIN_PATH:+:$QT_QPA_PLATFORM_PLUGIN_PATH}"
test -d "$COPPELIASIM_ROOT/system" || {
  echo "Install CoppeliaSim at $COPPELIASIM_ROOT before installing PyRep."
  return 1 2>/dev/null || exit 1
}
```

Install the local project components that exist in this repository:

```bash
python -m pip install -r X-ICM/PyRep/requirements.txt
python -m pip install -e X-ICM/PyRep

python -m pip install -r X-ICM/RLBench/requirements.txt
python -m pip install -e X-ICM/RLBench

python -m pip install -r X-ICM/YARR/requirements.txt
python -m pip install -e X-ICM/YARR

python -m pip install -r X-ICM/qwen2vl_finetune/requirements.txt
```

The `qwen2vl_finetune` subtree is installed from requirements only here. Add an editable install for it only if a future entry point needs it as an importable local package.

Add these exports to the training script or job launcher, not to committed code unless the path is parameterized.

## 6. Start A Training Session On CAIR

Use `tmux` so training survives SSH disconnects:

```bash
ssh cair
tmux new -s train-icra27
```

Inside tmux:

```bash
export PROJECT_NAME=ICRA27-ROBOT
export CAIR_PROJECT_ROOT=/data/yf23/projects/$PROJECT_NAME
export CAIR_ENV_NAME=icra27-robot
export CAIR_ENV=/data/yf23/conda/envs/$CAIR_ENV_NAME
export CAIR_DATA=/data/yf23/datasets/$PROJECT_NAME
export CAIR_RUNS=/data/yf23/runs/$PROJECT_NAME
export CAIR_CHECKPOINTS=/data/yf23/checkpoints/$PROJECT_NAME

source /data/yf23/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate "$CAIR_ENV"
cd "$CAIR_PROJECT_ROOT"
git status
git rev-parse HEAD
nvidia-smi
```

Create a run folder that records the exact Git commit:

```bash
export RUN_ID=$(date +%Y%m%d-%H%M%S)
export RUN_DIR="$CAIR_RUNS/$RUN_ID"
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git status --short > "$RUN_DIR/git_status.txt"
conda env export > "$RUN_DIR/conda_env.yml"
```

Run training with logs captured:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 \
python -m YOUR_TRAINING_MODULE --config path/to/config.yaml \
  2>&1 | tee "$RUN_DIR/train.log"
```

Replace `YOUR_TRAINING_MODULE` and `path/to/config.yaml` with the actual training entry point for the experiment.

For the current `X-ICM` entry point, run:

```bash
cd "$CAIR_PROJECT_ROOT/X-ICM"
set -o pipefail
CUDA_VISIBLE_DEVICES=0 \
python main.py framework.logdir="$RUN_DIR/logs" \
  2>&1 | tee "$RUN_DIR/train.log"
```

Detach from tmux:

```bash
Ctrl-b d
```

Reattach later:

```bash
tmux attach -t train-icra27
```

## 7. Commit Rules For This Session

Before any release, training run, or major experiment, the code should be committed and pushed.

Use this local workflow on the Mac:

```bash
cd "/Users/nicholas/Documents/ICRA27 ROBOT"
git status --short --branch
git diff
git add <files>
git diff --cached --stat
umask 077
git diff --cached > .git/claudecode-staged.diff
claude -p "$(printf '%s\n\n' 'Review this staged Git diff before commit. Focus on bugs, regressions, secrets, unsafe behavior, missing tests, and obvious maintainability issues. Return blocking findings first, then non-blocking notes. Diff:'; cat .git/claudecode-staged.diff)"
rm -f .git/claudecode-staged.diff
git commit -m "Describe the change"
git push origin main
```

If Claude Code reports a blocking issue, fix it before committing. Do not commit secrets, passwords, private SSH keys, datasets, checkpoints, or raw training outputs.

Use this release pattern:

```bash
export VERSION=v0.1.0
git tag -a "$VERSION" -m "Release $VERSION"
git push origin main --tags
```

For training milestones, use descriptive tags:

```bash
export TRAIN_TAG=train-$(date +%Y%m%d-%H%M)
git tag -a "$TRAIN_TAG" -m "Training snapshot $TRAIN_TAG"
git push origin main --tags
```

## 8. Pull Updates On CAIR Before Training

Before each CAIR run:

```bash
ssh cair
export PROJECT_NAME=ICRA27-ROBOT
export CAIR_PROJECT_ROOT=/data/yf23/projects/$PROJECT_NAME
export CAIR_ENV_NAME=icra27-robot
export CAIR_ENV=/data/yf23/conda/envs/$CAIR_ENV_NAME
cd "$CAIR_PROJECT_ROOT"
git fetch origin
git pull --ff-only
git rev-parse HEAD
```

If there are local CAIR changes, do not overwrite them. Commit them on a branch or copy them into a patch before pulling:

```bash
git status --short
git switch -c cair/local-work-$(date +%Y%m%d-%H%M)
git add <files>
git commit -m "Save CAIR local work"
```

## 9. What Belongs In Git

Commit:

- Source code.
- Config files needed to reproduce a run.
- Small scripts and launchers.
- Documentation and setup notes.
- Lightweight metadata that describes experiments.

Do not commit:

- Passwords or tokens.
- SSH private keys.
- Raw datasets.
- Checkpoints and model weights.
- Large logs, videos, generated rollouts, or cache folders.
- Machine-specific absolute paths unless they are examples in documentation.

Prefer storing large artifacts under:

```text
/data/yf23/datasets/ICRA27-ROBOT
/data/yf23/runs/ICRA27-ROBOT
/data/yf23/checkpoints/ICRA27-ROBOT
```

## 10. Handoff Checklist

For a future user taking over this project:

```bash
ssh cair 'hostname; whoami; nvidia-smi -L'
ssh cair 'test -d /data/yf23/projects/ICRA27-ROBOT && echo project-found'
ssh cair 'source /data/yf23/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true; conda activate /data/yf23/conda/envs/icra27-robot && python --version'
git -C "/Users/nicholas/Documents/ICRA27 ROBOT" status --short --branch
git -C "/Users/nicholas/Documents/ICRA27 ROBOT" remote -v
```

The project is ready for training when SSH is passwordless, the CAIR conda environment activates, the GitHub repo is reachable from both the Mac and CAIR, and `git status` is clean before launching the run.
