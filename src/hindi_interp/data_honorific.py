"""Honorific minimal pairs: formal आप vs. familiar तुम for the addressee.

Every prompt is a quoted question or remark and ends right before the second-person
pronoun (e.g. `बाद में राहुल ने अपने शिक्षक से पूछा, "क्या` -> " आप" / " तुम"). The only cue is
the addressee, marked by से. Each clean prompt has a corrupt twin in which a higher-status
addressee is swapped for a close or younger one of the same grammatical gender (so अपने/अपनी
does not change), and the two prompts are aligned token by token.

Items use the same fields as data.py so the experiment pipeline runs unchanged:
fem/masc hold the formal/familiar pronoun, speaker_gender is "F" when the addressee calls
for आप and "M" when it calls for तुम, and name_pos is the addressee's position.

Run as a script to write data/honorific.jsonl and data/review_honorific.md:
    python -m hindi_interp.data_honorific --tokenizer <path to tokenizer.json or HF model id>
"""
import argparse
import json
import random
from pathlib import Path

from .data import FEMALE, MALE, load_tokenizer

PRONOUNS = ("आप", "तुम")  # (formal, familiar); single tokens after a space

# (higher-status, close or younger) addressees, same gender within a pair, each one token.
# The possessive अपने/अपनी agrees with the addressee, so pairs never mix genders.
ADDRESSEE_PAIRS = [
    ("शिक्षक", "दोस्त", "m"),
    ("अध्यापक", "मित्र", "m"),
    ("डॉक्टर", "बेटे", "m"),
    ("पिताजी", "बच्चे", "m"),
    ("सर", "यार", "m"),
    ("बॉस", "भाई", "m"),
    ("दादी", "बेटी", "f"),
    ("टीचर", "बहन", "f"),
]
SPLIT_OF_PAIR = {i: ("train" if i < 5 else "dev" if i < 6 else "test") for i in range(len(ADDRESSEE_PAIRS))}
POSS = {"m": "अपने", "f": "अपनी"}

# Speakers: three female and three male names from the gender data. The verbs पूछा/कहा do
# not agree with the speaker, so the speaker's gender changes nothing else in the prompt.
SPEAKERS = [FEMALE[1], FEMALE[4], FEMALE[12], MALE[0], MALE[2], MALE[12]]

# Openings of the quote; the pronoun is the next word, and both pronouns fit.
FRAMES = {
    "h01": "क्या",      # क्या आप कल आएँगे? / क्या तुम कल आओगे?
    "h02": "कल",        # कल आप कहाँ थे? / कल तुम कहाँ थे?
    "h03": "अब",        # अब आप कैसे हैं? / अब तुम कैसे हो?
    "h04": "आज",        # आज आप क्या करेंगे? / आज तुम क्या करोगे?
    "h05": "इस बार",    # इस बार आप कहाँ जा रहे हैं? / इस बार तुम कहाँ जा रहे हो?
    "h06": "सुबह से",   # सुबह से आप कहाँ थे? / सुबह से तुम कहाँ थे?
}
# Natural continuation after the pronoun (formal, familiar), shown only in the review file.
FRAME_TAIL = {
    "h01": ("कल आएँगे?", "कल आओगे?"),
    "h02": ("कहाँ थे?", "कहाँ थे?"),
    "h03": ("कैसे हैं?", "कैसे हो?"),
    "h04": ("क्या करेंगे?", "क्या करोगे?"),
    "h05": ("कहाँ जा रहे हैं?", "कहाँ जा रहे हो?"),
    "h06": ("कहाँ थे?", "कहाँ थे?"),
}

# Feminine forms of the continuations, for female addressees: (formal, familiar) replacements.
FEM_TAIL = {"थे": ("थीं", "थी"), "कैसे": ("कैसी", "कैसी"), "रहे": ("रही", "रही"), "आएँगे": ("आएँगी", "आएँगी"),
            "आओगे": ("आओगी", "आओगी"), "करेंगे": ("करेंगी", "करेंगी"), "करोगे": ("करोगी", "करोगी")}


def tail(fid, formal, gender):
    t = FRAME_TAIL[fid][0 if formal else 1]
    if gender == "f":
        t = " ".join(FEM_TAIL.get(w.rstrip("?"), (w, w))[0 if formal else 1] + ("?" if w.endswith("?") else "")
                     if w.rstrip("?") in FEM_TAIL else w for w in t.split())
    return t


# Names are never sentence-initial, so they keep their single-token form.
# {S} = speaker name, {P} = अपने/अपनी, {A} = addressee, {F} = frame.
CUE_TEMPLATES = {
    "R1": ("asked", 'बाद में {S} ने {P} {A} से पूछा, "{F}'),
    "R2": ("said", 'फ़ोन पर {S} ने {P} {A} से कहा, "{F}'),
    "R3": ("addressee_first", 'बाद में {P} {A} से {S} ने पूछा, "{F}'),
}
# No addressee at all: measures the model's default.
NEUTRAL_TEMPLATES = {
    "N1": ("neutral", 'बाद में {S} ने पूछा, "{F}'),
    "N2": ("neutral", 'फ़ोन पर {S} ने कहा, "{F}'),
}


def build_items():
    items = []
    for tid, (cue_type, tmpl) in CUE_TEMPLATES.items():
        for fid, frame in FRAMES.items():
            for i, (hi, lo, g) in enumerate(ADDRESSEE_PAIRS):
                for spk in SPEAKERS:
                    for formal in (True, False):
                        add, other = (hi, lo) if formal else (lo, hi)
                        correct, contrast = PRONOUNS if formal else PRONOUNS[::-1]
                        items.append(dict(
                            group="cue", template=tid, cue_type=cue_type, frame=fid, pair=i,
                            split=SPLIT_OF_PAIR[i], speaker=add, speaker_gender="F" if formal else "M",
                            addressee=add, status="high" if formal else "low", speaker_name=spk,
                            prompt=tmpl.format(S=spk, P=POSS[g], A=add, F=frame),
                            corrupt_prompt=tmpl.format(S=spk, P=POSS[g], A=other, F=frame),
                            correct=correct, contrast=contrast, fem=PRONOUNS[0], masc=PRONOUNS[1]))
    for tid, (cue_type, tmpl) in NEUTRAL_TEMPLATES.items():
        for fid, frame in FRAMES.items():
            for spk in SPEAKERS:
                items.append(dict(
                    group="neutral", template=tid, cue_type=cue_type, frame=fid, pair=-1, split="all",
                    speaker=None, speaker_gender=None, speaker_name=spk,
                    prompt=tmpl.format(S=spk, F=frame), corrupt_prompt=None,
                    correct=None, contrast=None, fem=PRONOUNS[0], masc=PRONOUNS[1]))
    for k, it in enumerate(items):
        it["id"] = k
    return items


def check_tokenization(items, tokenize):
    for w in [a for hi, lo, _ in ADDRESSEE_PAIRS for a in (hi, lo)] + SPEAKERS:
        assert len(tokenize(" " + w)) == 1, f"{w} is not a single token"
    for it in items:
        toks = tokenize(it["prompt"])
        it["n_tokens"] = len(toks)
        for w in PRONOUNS:
            assert len(tokenize(it["prompt"] + " " + w)) == len(toks) + 1, f"{w} splits after {it['prompt']}"
        if it["corrupt_prompt"]:
            ctoks = tokenize(it["corrupt_prompt"])
            assert len(ctoks) == len(toks), f"length mismatch: {it['prompt']} / {it['corrupt_prompt']}"
            diff = [p for p, (a, b) in enumerate(zip(toks, ctoks)) if a != b]
            assert len(diff) == 1, f"clean/corrupt differ at {diff}: {it['prompt']}"
            it["name_pos"] = diff[0]
    return items


def write_review(items, path):
    rng = random.Random(0)
    lines = ["# Honorific data: please check\n",
             "Mark anything unnatural, ambiguous or wrong. The model must choose the pronoun in [brackets] "
             "(bold in the samples); the prompt stops before it. Both of you should agree on every "
             "addressee; we drop the ones you disagree on.\n",
             "## Addressee pairs (आप | तुम, same gender per row)\n"]
    lines += [f"{i+1}. {hi} | {lo}  ({g}, {SPLIT_OF_PAIR[i]})" for i, (hi, lo, g) in enumerate(ADDRESSEE_PAIRS)]
    lines.append("\nQuestion: would you address each left-hand person with आप and each right-hand one "
                 "with तुम? Mark any you would not (e.g. दादी, भाई, बहन vary by family).\n")
    lines.append("## Speakers\n")
    lines.append(", ".join(SPEAKERS) + "\n")
    lines.append("## Quote openings (आप / तुम continuation)\n")
    lines += [f"- {fid}: \"{fr} … [आप / तुम] {FRAME_TAIL[fid][0]} / {FRAME_TAIL[fid][1]}"
              f"  (to a woman: {tail(fid, True, 'f')} / {tail(fid, False, 'f')})" for fid, fr in FRAMES.items()]
    lines.append("\n## Context templates\n")
    for d in (CUE_TEMPLATES, NEUTRAL_TEMPLATES):
        lines += [f"- {tid} ({ct}): `{t}`" for tid, (ct, t) in d.items()]
    lines.append("\n## Random sample of full items (prompt → correct answer)\n")
    for grp, k in (("cue", 24), ("neutral", 4)):
        pool = [it for it in items if it["group"] == grp]
        for it in rng.sample(pool, k):
            if it["correct"]:
                g = ADDRESSEE_PAIRS[it["pair"]][2]
                ans = f"**{it['correct']}** {tail(it['frame'], it['correct'] == PRONOUNS[0], g)}"
            else:
                ans = "**आप/तुम** (no addressee)"
            lines.append(f"- [{it['template']}] {it['prompt']} → {ans}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--out", default="data/honorific.jsonl")
    args = ap.parse_args()
    items = check_tokenization(build_items(), load_tokenizer(args.tokenizer))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    write_review(items, Path(args.out).with_name("review_honorific.md"))
    counts = {}
    for it in items:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    print(f"wrote {len(items)} items to {args.out}: {counts}")


if __name__ == "__main__":
    main()
