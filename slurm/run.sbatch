#!/bin/bash
#SBATCH --job-name=qwen-embed
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

# Submit from the repo root: sbatch slurm/run.sbatch my_script.py [args]
# Put every call in that one script so each model loads once.
set -euo pipefail

export HF_HOME=/scratch/ad158/qwen
export HF_HUB_OFFLINE=1
export UV_PROJECT_ENVIRONMENT=/scratch/ad158/qwen/venv
export UV_CACHE_DIR=/scratch/ad158/uv-cache

cd "$SLURM_SUBMIT_DIR"
uv run python "${1:-my_script.py}" "${@:2}"
