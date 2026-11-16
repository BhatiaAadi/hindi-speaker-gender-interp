#!/bin/bash
#SBATCH --job-name=hi-setup
#SBATCH --partition=u22
#SBATCH --reservation=cs3406
#SBATCH --nodelist=node13
#SBATCH --account=cs3406
#SBATCH -c 2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
# Builds the venv and downloads Gemma-3-4B onto node13 scratch (CPU only).
set -euo pipefail
source $HOME/hindi_interp/scripts/env.sh
cat /proc/driver/nvidia/version || true
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.11 "$VENV"
fi
. "$VENV/bin/activate"
uv pip install -r $REPO/requirements.txt
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
from huggingface_hub import snapshot_download
p = snapshot_download("google/gemma-3-4b-pt", allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
print("downloaded to", p)
PY
du -sh $WORK/*
