"""Speaker-gender minimal pairs for Hindi first-person verbs.

Every prompt ends right before a gendered verb form (e.g. "... मैं कल बाज़ार" ->
" गई" / " गया"). The only cue to the speaker's gender is a name in the context.
Each clean prompt has a corrupt twin in which the speaker's name is swapped for
a name of the other gender with the same token count, so the two prompts are
aligned token by token.

Run as a script to write data/speaker_gender.jsonl and data/review_speaker_gender.md:
    python -m hindi_interp.data --tokenizer <path to tokenizer.json or HF model id>
"""
import argparse
import json
import random
from pathlib import Path

# 20 female / 20 male names, each a single Gemma-3 token when preceded by a space.
# Unisex names (सुमन, किरण) and names that are masculine common nouns (सपना) are excluded.
FEMALE = ["सीमा", "नेहा", "पूजा", "कविता", "प्रिया", "सुनीता", "दीपिका", "कमला", "सुधा", "प्रियंका",
          "अर्चना", "सुषमा", "निशा", "मीरा", "हेमा", "दिव्या", "अदिति", "शिल्पा", "गौरी", "रेखा"]
MALE = ["राहुल", "अमित", "रोहित", "सुनील", "अनिल", "राजेश", "संजय", "मनोज", "अजय", "सुरेश",
        "रमेश", "अर्जुन", "नितिन", "सौरभ", "राकेश", "पंकज", "सचिन", "तरुण", "महेश", "दिनेश"]

# Name pairs (FEMALE[i], MALE[i]) are split by index, so test names are never seen in training.
SPLIT_OF_PAIR = {i: ("train" if i < 12 else "dev" if i < 16 else "test") for i in range(20)}

# Verb pairs: (feminine form, masculine form). All are single tokens with a leading space.
GAYA = ("गई", "गया")
THA = ("थी", "था")
RAHA = ("रही", "रहा")
JATA = ("जाती", "जाता")
AYA = ("आई", "आया")

# First-person sentence frames; the gendered verb is the next token.
FRAMES = {
    "f01": ("मैं कल बाज़ार", GAYA),
    "f02": ("मैं पिछले साल दिल्ली", GAYA),
    "f03": ("मैं आज सुबह जल्दी उठ", GAYA),
    "f04": ("मैं कल रात बहुत थक", GAYA),
    "f05": ("मैं कल बीमार", THA),
    "f06": ("मैं बचपन में बहुत शरारती", THA),
    "f07": ("मैं उस दिन बहुत खुश", THA),
    "f08": ("मैं अभी खाना बना", RAHA),
    "f09": ("मैं आजकल हिंदी सीख", RAHA),
    "f10": ("मैं रोज़ सुबह पार्क", JATA),
    "f11": ("मैं हर रविवार मंदिर", JATA),
    "f12": ("मैं कल रात देर से घर", AYA),
}

# Context templates. {N} = speaker name, {F} = frame. The name is never sentence-initial,
# so it always gets the single-token "▁name" form.
CUE_TEMPLATES = {
    "T1": ("quoted", 'फ़ोन पर {N} ने कहा, "{F}'),
    "T2": ("quoted", 'बाद में {N} ने बताया, "{F}'),
    "T3": ("intro", "नमस्ते, मेरा नाम {N} है। {F}"),
    "T4": ("diary", "यह {N} की डायरी है। {F}"),
    "T5": ("letter", "प्रिय दोस्त, मैं {N} हूँ। {F}"),
}
# No cue to gender at all: measures the model's default (H3).
NEUTRAL_TEMPLATES = {
    "N1": ("neutral", 'फ़ोन पर उसने कहा, "{F}'),
    "N2": ("neutral", 'बाद में किसी ने बताया, "{F}'),
    "N3": ("neutral", "प्रिय दोस्त, {F}"),
}
# Two names of opposite gender; only the speaker ({S}) decides the verb (H4).
# D1 puts the distractor ({D}) nearer to the answer, D2 puts the speaker nearer.
DISTRACTOR_TEMPLATES = {
    "D1": ("distractor_near", 'फ़ोन पर {S} ने {D} से कहा, "{F}'),
    "D2": ("speaker_near", 'फ़ोन पर {D} से {S} ने कहा, "{F}'),
}


def _answer(frame_verbs, gender):
    fem, masc = frame_verbs
    return (fem, masc) if gender == "F" else (masc, fem)


def build_items():
    """Return a list of item dicts (clean prompt, corrupt prompt, answers, metadata)."""
    items = []
    for tid, (cue_type, tmpl) in CUE_TEMPLATES.items():
        for fid, (frame, verbs) in FRAMES.items():
            for i in range(20):
                for gender in ("F", "M"):
                    spk, other = (FEMALE[i], MALE[i]) if gender == "F" else (MALE[i], FEMALE[i])
                    correct, contrast = _answer(verbs, gender)
                    items.append(dict(
                        group="cue", template=tid, cue_type=cue_type, frame=fid, pair=i,
                        split=SPLIT_OF_PAIR[i], speaker=spk, speaker_gender=gender,
                        prompt=tmpl.format(N=spk, F=frame),
                        corrupt_prompt=tmpl.format(N=other, F=frame),
                        correct=correct, contrast=contrast, fem=verbs[0], masc=verbs[1]))
    for tid, (cue_type, tmpl) in DISTRACTOR_TEMPLATES.items():
        for fid, (frame, verbs) in FRAMES.items():
            for i in range(20):
                j = (i + 7) % 20  # distractor comes from a different pair, same split family
                for gender in ("F", "M"):
                    spk, other = (FEMALE[i], MALE[i]) if gender == "F" else (MALE[i], FEMALE[i])
                    dis = MALE[j] if gender == "F" else FEMALE[j]  # opposite gender to speaker
                    correct, contrast = _answer(verbs, gender)
                    items.append(dict(
                        group="distractor", template=tid, cue_type=cue_type, frame=fid, pair=i,
                        split=SPLIT_OF_PAIR[i], speaker=spk, speaker_gender=gender, distractor=dis,
                        prompt=tmpl.format(S=spk, D=dis, F=frame),
                        corrupt_prompt=tmpl.format(S=other, D=dis, F=frame),
                        correct=correct, contrast=contrast, fem=verbs[0], masc=verbs[1]))
    for tid, (cue_type, tmpl) in NEUTRAL_TEMPLATES.items():
        for fid, (frame, verbs) in FRAMES.items():
            items.append(dict(
                group="neutral", template=tid, cue_type=cue_type, frame=fid, pair=-1, split="all",
                speaker=None, speaker_gender=None, prompt=tmpl.format(F=frame), corrupt_prompt=None,
                correct=None, contrast=None, fem=verbs[0], masc=verbs[1]))
    for k, it in enumerate(items):
        it["id"] = k
    return items


def load_tokenizer(path_or_id):
    if path_or_id.endswith(".json"):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(path_or_id)
        return lambda s: tok.encode(s, add_special_tokens=False).tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_id)
    return lambda s: tok.tokenize(s)


def check_tokenization(items, tokenize):
    """Assert the properties the causal analyses rely on; annotate token positions."""
    for w in FEMALE + MALE:
        assert len(tokenize(" " + w)) == 1, f"name {w} is not a single token"
    for _, verbs in FRAMES.values():
        for v in verbs:
            assert len(tokenize(" " + v)) == 1, f"verb {v} is not a single token"
    for it in items:
        toks = tokenize(it["prompt"])
        it["n_tokens"] = len(toks)
        if it["corrupt_prompt"] is not None:
            ctoks = tokenize(it["corrupt_prompt"])
            assert len(ctoks) == len(toks), f"length mismatch: {it['prompt']} / {it['corrupt_prompt']}"
            diff = [p for p, (a, b) in enumerate(zip(toks, ctoks)) if a != b]
            assert len(diff) == 1, f"clean/corrupt differ at {diff}: {it['prompt']}"
            it["name_pos"] = diff[0]  # index into tokens without BOS
        if it.get("distractor"):
            it["distractor_pos"] = toks.index("▁" + it["distractor"])
    return items


def write_review(items, path):
    rng = random.Random(0)
    lines = ["# Speaker-gender data: please check\n",
             "Mark anything unnatural, ambiguous or wrong. The model must predict the word in [brackets].\n",
             "## Names (female | male, paired by row)\n"]
    lines += [f"{i+1}. {f} | {m}  ({SPLIT_OF_PAIR[i]})" for i, (f, m) in enumerate(zip(FEMALE, MALE))]
    lines.append("\n## Sentence frames (feminine / masculine continuation)\n")
    lines += [f"- {fid}: {fr} … [{v[0]} / {v[1]}]" for fid, (fr, v) in FRAMES.items()]
    lines.append("\n## Context templates\n")
    for d in (CUE_TEMPLATES, NEUTRAL_TEMPLATES, DISTRACTOR_TEMPLATES):
        lines += [f"- {tid} ({ct}): `{t}`" for tid, (ct, t) in d.items()]
    lines.append("\n## Random sample of full items (prompt → correct answer)\n")
    for grp, k in (("cue", 20), ("distractor", 8), ("neutral", 4)):
        pool = [it for it in items if it["group"] == grp]
        for it in rng.sample(pool, k):
            ans = it["correct"] or f"{it['fem']}/{it['masc']} (no cue)"
            lines.append(f"- [{it['template']}] {it['prompt']} → **{ans}**")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_items(path="data/speaker_gender.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--out", default="data/speaker_gender.jsonl")
    args = ap.parse_args()
    items = check_tokenization(build_items(), load_tokenizer(args.tokenizer))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    write_review(items, Path(args.out).with_name("review_speaker_gender.md"))
    counts = {}
    for it in items:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    print(f"wrote {len(items)} items to {args.out}: {counts}")


if __name__ == "__main__":
    main()
