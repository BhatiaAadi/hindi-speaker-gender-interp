# Shared environment for all jobs on turing node13.
# Heavy files (venv, HF cache, uv cache) live on node13's local /scratch because
# the team home directory is nearly full. Code and small results stay in $HOME.
export WORK=/scratch/cs3406c_hindi
export REPO=$HOME/hindi_interp
export HF_HOME=$WORK/hf
export UV_CACHE_DIR=$WORK/uvcache
export UV_PYTHON_INSTALL_DIR=$WORK/python
export VENV=$WORK/venv
export PYTHONPATH=$REPO/src
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK" "$HF_HOME" "$UV_CACHE_DIR"
# Load HF_TOKEN without echoing it, even under `set -x`.
{ _x=$-; set +x; } 2>/dev/null
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
case $_x in *x*) set -x;; esac
if [ -x "$VENV/bin/python" ]; then . "$VENV/bin/activate"; fi
