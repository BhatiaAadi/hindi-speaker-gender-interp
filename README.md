# Where Does the Model Look? Speaker Gender in Hindi First-Person Verbs

Mechanistic-interpretability mini-project (IIIT Hyderabad). Hindi first-person verbs mark the
speaker's gender (`मैं कल बाज़ार गई थी` / `गया था`); English (`I went`) does not. We ask where
Gemma-3-4B (pretrained) reads the speaker's gender from the context, which components carry it to
the verb, and why it fails when it fails.

## Layout
- `src/hindi_interp/data.py` — builds the Hindi minimal-pair dataset (`data/speaker_gender.jsonl`)
  and checks tokenization (clean/corrupt prompts aligned token by token).
- `src/hindi_interp/model.py` — Gemma-3 loading, activation caching and patching with PyTorch hooks.
- `src/hindi_interp/experiments.py` — the pipeline: `sanity`, `behaviour`, `lens`, `patching`,
  `probes`, `heads`, `ambig`, `route`, `diagnose` (see the module docstring).
- `src/hindi_interp/plots.py` — report figures from the results directory.
- `scripts/` — Slurm jobs for the GPU server.
- `results/gemma-3-4b-pt/` — all outputs (CSV/JSON) of the Gemma-3-4B run.
- `report/` — the ACL-format report (`report.tex`, figures, bibliography).
- `proposal/` — project proposal and slides.

## Main result
The speaker's gender travels name → `मैं` → answer: attention layer 14 (heads L14H1, L14H3) copies it
from the name onto the pronoun `मैं`, MLP 18 processes it there, and head L25H6 moves it to the answer
position (75% of the patching effect on its own). Late MLPs 26, 29, 30 and 32 write a masculine
default. Errors come from female names that are also nouns (`कविता`, poem), whose gender never
reaches the answer position; steering at layer 26 repairs them.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...            # access to google/gemma-3-4b-pt
python -m hindi_interp.data --tokenizer google/gemma-3-4b-pt
PYTHONPATH=src python -m hindi_interp.experiments sanity behaviour lens patching probes heads ambig route diagnose \
    --out results/gemma-3-4b-pt --work /path/to/scratch
PYTHONPATH=src python -m hindi_interp.plots --res results/gemma-3-4b-pt --fig report/figures
cd report && xelatex report && bibtex report && xelatex report && xelatex report
```
The full run takes about 1.5 h on one L40S (bfloat16).
