# Where Does the Model Look? Speaker Gender in Hindi First-Person Verbs

Mechanistic-interpretability mini-project (IIIT Hyderabad). Hindi first-person verbs mark the
speaker's gender (`मैं कल बाज़ार गई` / `गया`); English (`I went`) does not. We ask where
Gemma-3-4B (pretrained) reads the speaker's gender from the context, which components carry it to
the verb, and why it fails when it fails.

## Layout
- `src/hindi_interp/data.py` — builds the Hindi minimal-pair dataset (`data/speaker_gender.jsonl`)
  and checks tokenization (clean/corrupt prompts aligned token by token).
- `src/hindi_interp/model.py` — Gemma-3 loading, activation caching and patching with PyTorch hooks.
- `src/hindi_interp/experiments.py` — the pipeline: `sanity`, `behaviour`, `lens`, `patching`,
  `probes`, `diagnose`, `heads`.
- `scripts/` — Slurm jobs for the GPU server.
- `results/` — outputs (CSV/JSON) and figures.
- `proposal/` — project proposal and slides.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...            # access to google/gemma-3-4b-pt
python -m hindi_interp.data --tokenizer google/gemma-3-4b-pt
PYTHONPATH=src python -m hindi_interp.experiments sanity behaviour lens patching probes diagnose heads
```
