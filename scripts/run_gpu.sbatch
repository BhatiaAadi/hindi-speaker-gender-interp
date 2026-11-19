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
#        TASK=honorific sbatch --export=ALL scripts/run_gpu.sbatch behaviour   (or TASK=kinship)
set -euo pipefail
source $HOME/hindi_interp/scripts/env.sh
cd $REPO
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
TASK=${TASK:-speaker_gender}
case $TASK in
  speaker_gender) MOD=data; OUT=results/gemma-3-4b-pt ;;
  honorific) MOD=data_honorific; OUT=results/gemma-3-4b-pt-honorific ;;
  kinship) MOD=data_kinship; OUT=results/gemma-3-4b-pt-kinship ;;
esac
python -m hindi_interp.$MOD --tokenizer google/gemma-3-4b-pt --out data/$TASK.jsonl
python -m hindi_interp.experiments "$@" --data data/$TASK.jsonl --out $OUT --work $WORK/acts-$TASK
