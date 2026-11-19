# Honorifics and kinship: results for the report

Model: Gemma-3-4B-pt, same methods and settings as the speaker-gender study. Every number below
comes from a file in `results/gemma-3-4b-pt-honorific/` or `results/gemma-3-4b-pt-kinship/`;
the file is named after each claim. Figures are in `report/figures/honorific/`.

## 1. Honorifics (आप vs तुम)

### Data (`data/honorific.jsonl`, `src/hindi_interp/data_honorific.py`)
- The prompt ends just before the second-person pronoun in a quote, e.g.
  `बाद में नेहा ने अपने शिक्षक से पूछा, "क्या` → **आप** (कल आएँगे?).
  The corrupt prompt swaps only the addressee: शिक्षक → दोस्त, so the answer becomes **तुम**.
- 8 addressee pairs (higher status | close or younger), same gender within a pair so अपने/अपनी
  does not change: शिक्षक|दोस्त, अध्यापक|मित्र, डॉक्टर|बेटे, पिताजी|बच्चे, सर|यार, बॉस|भाई, दादी|बेटी,
  टीचर|बहन. Split by pair: 5 train, 1 dev, 2 test.
- 6 speakers (3 female, 3 male names from the gender data), 6 quote openings
  (क्या, कल, अब, आज, इस बार, सुबह से), 3 templates:
  R1 `बाद में {S} ने अपने {A} से पूछा, "`, R2 `फ़ोन पर {S} ने अपने {A} से कहा, "`,
  R3 `बाद में अपने {A} से {S} ने पूछा, "` (addressee first, speaker nearer the answer).
- 1,728 cue items + 72 no-addressee items (`बाद में {S} ने पूछा, "`). All clean/corrupt pairs
  differ at exactly one token. Checked by both authors.

### Behaviour (`behaviour_summary.json`)
- Accuracy (LD > 0) **96.4%** [93.5, 98.7]; R1 94.1%, R2 96.4%, **R3 98.6%**.
- Swapping the addressee flips the answer in **96.2%** [93.0, 98.7] of correct items.
- Top-1 is the right pronoun in only 21.8% (the model often continues the quote with another
  word); free generation contains a pronoun in 29.7% of items.
- No addressee: the model prefers **आप** in 96% of prompts (LD +1.21 [1.03, 1.39]).
  So the default is the polite form, the opposite of the masculine default for gender.
- 63 errors: 41 are familiar addressees answered with आप (बच्चे 10, बहन 10, भाई 7, बेटे 5,
  बेटी 5, दोस्त 3, मित्र 1), 22 are formal addressees answered with तुम (सर 7, अध्यापक 6, टीचर 4,
  पिताजी 3, शिक्षक 2). By opening: कल 22, आज 17, इस बार 15, others ≤ 4.

### Route: addressee → quote mark → answer (`patching_resid_maps.json`; Fig. `resid_positions.pdf`, `resid_map_R1_h01_denoise.pdf`)
Restoring the clean residual stream at one position (12 cells: R1–R3 × 4 openings, up to 40 pairs each):
- **Addressee** position carries the whole effect up to layer 8 (≥ 0.92), then falls: 0.70 at
  layer 9, 0.44 at 14, 0.21 at 16, ≤ 0.07 from layer 21.
- से right after it carries 0.16–0.21 at layers 9–15.
- **Opening quote mark `"`**: 0.13 at layer 14, 0.26 at 16, 0.33–0.35 at 17–20,
  **0.44–0.46 at 21–25**, then drops to 0.26 at 26 and ≤ 0.11 from 27.
- **Answer** position: ≤ 0.08 up to layer 15, 0.23–0.39 at 17–25, **0.62 at 26, 0.86 at 27**,
  0.96–0.97 at 32–33.
- So between layers 16 and 25 the information is split between the quote mark and the answer
  position, and from layer 26 it is at the answer position.
  For speaker gender the parking token was मैं; here it is the quote mark that opens the clause.

### Components at the answer position (`patching_summary.json`; Figs. `attn_mlp.pdf`, `heads.pdf`)
- Attention outputs: layer 26 **0.29**, layer 25 **0.26**, layer 20 0.12, layer 31 0.10.
- MLP outputs: layer 26 **0.34**, layers 20 and 27 0.13.
- Heads: **L25H6 0.25**, **L26H7 0.25**, L31H5 0.10, then ≤ 0.06 (L20H1, L22H3, L20H3).
  Stable across templates: L25H6 0.27 / 0.22 / 0.26 and L26H7 0.27 / 0.22 / 0.26 in R1 / R2 / R3.
- **L25H6 is the same head that carries speaker gender** (0.75 there). For honorifics it shares
  the work with L26H7.

### What the heads attend to (`honor_summary.json`; Fig. `honor_attention.pdf`)
Attention from the answer position, mean over R1–R3:
- **L25H6: ≈ 0.48 on the quote mark**, < 0.02 on the addressee, से or speaker, in every order
  (R1 0.47, R2 0.49, R3 0.48). For speaker gender it attended to मैं (0.69). Same head, same job:
  it reads the feature from the token where it was parked.
- L26H7: ≈ 0.28 on the quote mark, 0.03–0.05 directly on the addressee; it looks at the addressee
  more than at the speaker in 88% of items, also when the speaker is nearer (R3).
- L31H5, L22H3, L20H3: 0.19–0.29 on the quote mark.
- The speaker-gender copying heads L14H1/L14H3 do attend to the addressee (0.07–0.12, more than
  to the speaker in 82% / 50% of items) but have almost no effect at the answer position
  (patching 0.01).

### Who writes onto the quote mark (`route_patching.csv`; second panels of `attn_mlp.pdf`, `heads.pdf`)
Restoring one component's output at the quote mark only, in the corrupt run (same 12 cells):
- Residual stream at the quote mark: 0.13 at layers 14–15, 0.26 at 16, 0.33–0.35 at 17–20,
  0.44–0.45 at 21–25, then ≤ 0.10 from layer 27 (matches the maps above).
- Attention outputs: layer 20 **0.15**, layer 15 **0.13**, layer 16 0.10, layer 13 0.07.
- MLP outputs: **layer 20 0.23**, then ≤ 0.08 (layers 23, 19, 18, 17).
- Heads: L20H1 0.10, L15H6 0.09, L13H5 0.07, L20H3 0.05, L16H4 0.04, L22H5 0.04.
- So several mid-layer heads (13–20) and MLP 20 build the feature on the quote mark; no single
  head dominates, unlike speaker gender, where attention layer 14 (L14H1 0.20, L14H3 0.17) and
  MLP 18 (0.76) did most of the work at मैं.

### Default (`dla_components.csv`, `honor_summary.json` → `default`; Fig. `dla.pdf`)
- On no-addressee prompts the MLPs in layers 17–33 that push hardest towards आप (direct logit
  attribution) are 32 (+0.63), 19 (+0.20), 17 (+0.17) and 25 (+0.16). MLP 27 pushes towards तुम (−1.27).
- Zeroing their output at the answer position: none 96% आप (LD +1.21); MLP 32 only 93% (+1.01);
  **all four 82% (+0.75)**; control MLPs 13–16 100% (+1.36).
- So the formal default is only partly written by these MLPs, unlike the masculine default
  (83% → 8% for gender). MLP 32 pushes both defaults: masculine and formal.

### Failures and steering (`diagnose.json`, `steering.csv`; Fig. `steering.pdf`)
- L* (best probe layer on dev) = 11; first layer where the answer position is used = 26.
- Probes read the addressee's class on the **wrong items** with 100% accuracy at both layers 11
  and 26 (held-out addressees: test accuracy 1.0 from about layer 11, `probes.csv`). Caveat:
  the control task is also at 1.0 there, so the probes show the information is present, not
  that it is encoded more cleanly than arbitrary labels (only 8 addressee pairs).
- Steering the 63 wrong items at **layer 26**: diff-of-means repairs **90%** at α = 0.5 and
  **98–100%** at α ≥ 1; probe direction 100% at all strengths. 20 random directions repair 28%
  on average at α = 0.5 (38% at α = 1), but the best random direction reaches 90–100%: the wrong
  items are close to the boundary.
- At layer 11: diff-of-means 13–57%, probe 14–38%, random mean 9–38%. As for gender, steering
  at the probe-chosen layer barely beats random.
- **H1 for honorifics: supported**: the cue is readable on wrong items and steering at the
  layer where it is used repairs them. This differs from speaker gender, where the cue was lost
  before the answer position.

### Hypotheses, honorifics
- H1 (readout failures): **supported** (see above, with the random-direction caveat).
- H3 (defaults from late MLPs): **partly**: there is a strong formal default; MLPs 17, 19, 25
  and 32 write part of it (96% → 82%).
- H4 analogue (heads find the right person): the output heads read from the quote mark, not
  from the person; the addressee's status is moved onto the quote mark by several heads in
  layers 13–20 and MLP 20. R3 accuracy (98.6%) shows word order does not fool the model.

## 2. Kinship (maternal vs paternal)

### Data (`data/kinship.jsonl`, `src/hindi_interp/data_kinship.py`)
- The prompt ends before the kinship term; the corrupt prompt swaps only the parent word:
  माँ ↔ पिता, मम्मी ↔ पापा, माता ↔ पिता (English: mother ↔ father).
- 6 relations: छोटे भाई (मामा/चाचा), पिता (नाना/दादा), माँ (नानी/दादी), बहन (मौसी/बुआ),
  बहन के पति (मौसा/फूफा), छोटे भाई की पत्नी (मामी/चाची). Two-token terms are scored on their
  first token (मौ vs बु), which already decides the side.
- 5 prompt types: K1 `हम माँ के छोटे भाई को` … कहते हैं; K2 `हिंदी में …को`; K3 a short story
  (`छुट्टियों में हम … के घर गए। हम उन्हें`); K4 English (`In Hindi, my mother's younger brother
  is called`); K5 the same as K1 after two worked examples from other relations (one maternal,
  one paternal). 156 cue items + 6 no-side prompts. Checked by a native speaker; छोटे is kept on
  the mother's side too (still मामा) so the two prompts have the same length.

### Behaviour (`behaviour.csv`, `behaviour_summary.json`)
| Prompt type | n | Accuracy | Flip rate | LD (maternal − paternal), maternal cue | paternal cue |
|---|---|---|---|---|---|
| K1 definition | 36 | 47% | 0% | −1.35 | −1.43 |
| K2 हिंदी में | 36 | 64% | 28% | −0.03 | −0.67 |
| K3 story | 36 | 50% | 6% | −0.20 | −0.21 |
| K4 English | 12 | 50% | 0% | +0.22 | −0.13 |
| K5 two examples | 36 | 50% | 0% | −0.81 | −1.09 |
| **All** | 156 | **52.6%** [48.5, 58.0] | 7.7% of all items | | |

(Flip rate here = share of all items that are answered correctly and whose answer changes when the parent is swapped.)
- The answer barely depends on the parent: the LD moves by 0.0–0.6 between माँ and पिता.
- **Paternal default**: 67% of all cue items prefer the paternal term, whichever parent is named.
  By relation: चाची over मामी 92%, चाचा over मामा 85%, दादा over नाना 81%, दादी over नानी 69%,
  बुआ over मौसी 62%; the exception is मौसा/फूफा, where मौ… is preferred in 88%.
- No side given: अंकल → चाचा (LD −1.19), आंटी → बुआ (−1.75), English aunt → बुआ (−3.16),
  grandfather → दादा (−0.69); English uncle (+2.38) and grandmother (+1.31) go maternal.
  4 of 6 paternal.
- Worked examples in the prompt (K5) do not help: 50%, no flips.

### Scope
Gemma-3-4B does not compute the side of the family, so there is no mechanism to trace:
we report behaviour only. Patching, heads and steering need a cue the model uses. The two-step
relations (H2) are in the data but cannot be tested for an intermediate step. Probes on these
156 items are not informative (no better than the control task), so we do not report them.
This matches Pardasani & Singh: models know the terms but do not produce the right one, with a
paternal advantage in Hindi. H3's "paternal default" is supported at the level of behaviour.

## 3. Across the three tasks
- **Same output head**: L25H6 moves both speaker gender and honorific status to the answer.
  In both it reads from the token that opens the clause where the cue was parked: मैं (gender,
  attention 0.69) and the quote mark (honorifics, 0.48).
- **Defaults**: masculine for gender (83%), formal for honorifics (96%), paternal for kinship
  (67% of cue items, 4/6 no-side prompts). MLP 32 pushes both the masculine and the formal default.
  Late MLPs write almost all of the masculine default but only part of the formal one.
- **Failures**: gender failures lose the cue before the answer (not readout failures);
  honorific failures keep it (readout failures, repaired by steering at layer 26); kinship
  is not computed at all.
- Steering at the probe-chosen layer (10 for gender, 11 for honorifics) barely beats random in
  both; steering at layer 26, where the answer position starts to use the cue, works in both.
