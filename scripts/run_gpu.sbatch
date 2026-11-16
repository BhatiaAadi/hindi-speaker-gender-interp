#!/bin/bash
#SBATCH --job-name=hi-gpu
#SBATCH --partition=u22
#SBATCH --reservation=cs3406
#SBATCH --nodelist=node13
#SBATCH --account=cs3406
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch scripts/run_gpu.sbatch sanity behaviour lens patching probes diagnose heads
set -euo pipefail
source $HOME/hindi_interp/scripts/env.sh
cd $REPO
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -m hindi_interp.data --tokenizer google/gemma-3-4b-pt --out data/speaker_gender.jsonl
python -m hindi_interp.experiments "$@" --out results/gemma-3-4b-pt --work $WORK/acts
