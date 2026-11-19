# Who Is Speaking, and to Whom? Speaker Gender, Honorifics and Kinship in Hindi inside Gemma-3

Mechanistic-interpretability mini-project (IIIT Hyderabad). Hindi makes a model choose what English
leaves open: the speaker's gender in first-person verbs (`मैं कल बाज़ार गई थी` / `गया था`), the
formality of "you" (`आप` / `तुम`) and the side of the family (`मामा` / `चाचा`). We ask where
Gemma-3-4B (pretrained) reads each cue from the context, which components and SAE features carry it
to the answer, and why it fails when it fails.

## Layout
- `src/hindi_interp/data.py`, `data_honorific.py`, `data_kinship.py` — build the three minimal-pair
  datasets (`data/*.jsonl`, with `review_*.md` files for native-speaker checks) and check tokenization
  (clean/corrupt prompts aligned token by token).
- `src/hindi_interp/model.py` — Gemma-3 loading, activation caching and patching with PyTorch hooks.
- `src/hindi_interp/experiments.py` — the pipeline: `sanity`, `behaviour`, `lens`, `patching`,
  `probes`, `heads`, `ambig`, `route`, `diagnose`, and for honorifics `honor`, `honor_route`
  (see the module docstring); `sae` runs the Gemma Scope 2 SAE analysis in `sae.py`.
- `src/hindi_interp/plots.py` — report figures from the results directory.
- `scripts/` — Slurm jobs for the GPU server.
- `results/gemma-3-4b-pt{,-honorific,-kinship}/` — all outputs (CSV/JSON) per task;
  `results/EXTENSIONS.md` summarises the honorific and kinship numbers.
- `report/` — the ACL-format report (`report.tex`, figures, bibliography).
- `proposal/` — project proposal and slides.

## Main result
- **Speaker gender** travels name → `मैं` → answer: attention layer 14 (heads L14H1, L14H3) copies it
  onto the pronoun, where one SAE feature fires on every female-speaker prompt, and head L25H6 moves it
  to the answer. Late MLPs 26, 29, 30, 32 write a masculine default. Errors come from female names that
  are also nouns (`कविता`, poem): the model's female-name feature never fires on them.
- **Honorifics** travel addressee → opening quote mark → answer, and the same head L25H6 reads them
  from the quote mark (with different SAE features). The default is formal (`आप`); errors keep the cue
  and steering at layer 26 repairs them.
- **Kinship** is not computed: accuracy is at chance and the model prefers paternal terms.

## Run
```bash
pip install -r requirements.txt
export HF_TOKEN=...            # access to google/gemma-3-4b-pt
python -m hindi_interp.data --tokenizer google/gemma-3-4b-pt
PYTHONPATH=src python -m hindi_interp.experiments sanity behaviour lens patching probes heads ambig route diagnose sae \
    --out results/gemma-3-4b-pt --work /path/to/scratch
python -m hindi_interp.data_honorific && PYTHONPATH=src python -m hindi_interp.experiments \
    sanity behaviour lens patching probes heads diagnose honor honor_route sae \
    --data data/honorific.jsonl --out results/gemma-3-4b-pt-honorific --work /path/to/scratch
python -m hindi_interp.data_kinship && PYTHONPATH=src python -m hindi_interp.experiments behaviour \
    --data data/kinship.jsonl --out results/gemma-3-4b-pt-kinship --work /path/to/scratch
PYTHONPATH=src python -m hindi_interp.plots --res results/gemma-3-4b-pt --fig report/figures
PYTHONPATH=src python -m hindi_interp.plots --task honorific --res results/gemma-3-4b-pt-honorific --fig report/figures/honorific
cd report && xelatex report && bibtex report && xelatex report && xelatex report
```
On turing, `TASK=honorific sbatch --export=ALL scripts/run_gpu.sbatch <steps>` does the same. The
gender run takes about 1.5 h on one L40S (bfloat16), the honorific run about 40 min.
