"""Kinship minimal pairs: maternal vs. paternal side (मामा vs. चाचा).

Every prompt ends right before a kinship term (e.g. "हम माँ के छोटे भाई को" -> " मामा" /
" चाचा"). The only cue is the parent word: the corrupt twin swaps माँ for पिता (or मम्मी
for पापा, माता for पिता), so the two prompts are aligned token by token. Terms that take two
tokens (मौसी = मौ+सी, बुआ = बु+आ) are scored on their first token, which already decides
the side.

Items use the same fields as data.py so the experiment pipeline runs unchanged:
fem/masc hold the first token of the maternal/paternal term, speaker_gender is "F" for a
maternal cue and "M" for a paternal one, and name_pos is the parent word's position.

Run as a script to write data/kinship.jsonl and data/review_kinship.md:
    python -m hindi_interp.data_kinship --tokenizer <path to tokenizer.json or HF model id>
"""
import argparse
import json
import random
from pathlib import Path

from .data import load_tokenizer

# (maternal, paternal) parent words, each one token. Index = cue pair.
PARENT_PAIRS = [("माँ", "पिता"), ("मम्मी", "पापा"), ("माता", "पिता")]

# Relation of the parent -> (Hindi phrase, postposition after the parent, English phrase,
# maternal term, paternal term). k05 and k06 are two-step relations.
RELATIONS = {
    "k01": ("छोटे भाई", "के", "younger brother", "मामा", "चाचा"),
    "k02": ("पिता", "के", "father", "नाना", "दादा"),
    "k03": ("माँ", "की", "mother", "नानी", "दादी"),
    "k04": ("बहन", "की", "sister", "मौसी", "बुआ"),
    "k05": ("बहन के पति", "की", "sister's husband", "मौसा", "फूफा"),
    "k06": ("छोटे भाई की पत्नी", "के", "younger brother's wife", "मामी", "चाची"),
}
SPLIT_OF_RELATION = {"k01": "train", "k02": "train", "k04": "train", "k03": "dev", "k05": "test", "k06": "test"}

# {P} = parent, {k} = के/की, {R} = relation. Hindi templates use all parent pairs;
# the English one only mother/father.
CUE_TEMPLATES = {
    "K1": ("definition", "हम {P} {k} {R} को"),                                      # … मामा कहते हैं।
    "K2": ("definition", "हिंदी में {P} {k} {R} को"),                                # … मामा कहते हैं।
    "K3": ("story", "छुट्टियों में हम {P} {k} {R} के घर गए। हम उन्हें"),             # … मामा बुलाते हैं।
    "K4": ("english", "In Hindi, my {P}'s {R} is called"),
}
ENGLISH_PARENTS = ("mother", "father")
# Few-shot: two worked examples from other relations (one maternal, one paternal) before the
# query. The examples are the same in the clean and corrupt prompt, so only the parent differs.
FEWSHOT = "K5"
CUE_TEMPLATES[FEWSHOT] = ("fewshot", "{D} हम {P} {k} {R} को")


def demos(rid):
    rids = list(RELATIONS)
    a, b = rids[(rids.index(rid) + 1) % len(rids)], rids[(rids.index(rid) + 2) % len(rids)]
    (ra, ka, _, ma, _), (rb, kb, _, _, pb) = RELATIONS[a], RELATIONS[b]
    return f"हम माँ {ka} {ra} को {ma} कहते हैं। हम पिता {kb} {rb} को {pb} कहते हैं।"
# No side given: measures the model's default (paternal?). अंकल/आंटी do not mark the side.
NEUTRAL_PROMPTS = {
    "N1": ("k01", "मेरे अंकल को हिंदी में हम"),
    "N2": ("k04", "मेरी आंटी को हिंदी में हम"),
    "N3": ("k01", "In Hindi, my uncle is called"),
    "N4": ("k04", "In Hindi, my aunt is called"),
    "N5": ("k02", "In Hindi, my grandfather is called"),
    "N6": ("k03", "In Hindi, my grandmother is called"),
}
TAIL = {"K1": "कहते हैं।", "K2": "कहते हैं।", "K3": "बुलाते हैं।", "K4": "", "K5": "कहते हैं।"}


def build_items():
    items = []
    for tid, (cue_type, tmpl) in CUE_TEMPLATES.items():
        pairs = [ENGLISH_PARENTS] if tid == "K4" else PARENT_PAIRS
        for rid, (rel, post, rel_en, mat, pat) in RELATIONS.items():
            for i, (pm, pp) in enumerate(pairs):
                for maternal in (True, False):
                    par, other = (pm, pp) if maternal else (pp, pm)
                    fmt = (lambda p: tmpl.format(P=p, R=rel_en)) if tid == "K4" else \
                          (lambda p: tmpl.format(P=p, k=post, R=rel, D=demos(rid)))
                    items.append(dict(
                        group="cue", template=tid, cue_type=cue_type, frame=rid, pair=i,
                        split=SPLIT_OF_RELATION[rid], speaker=par, speaker_gender="F" if maternal else "M",
                        side="maternal" if maternal else "paternal",
                        prompt=fmt(par), corrupt_prompt=fmt(other),
                        correct_word=mat if maternal else pat, contrast_word=pat if maternal else mat,
                        fem_word=mat, masc_word=pat))
    for tid, (rid, prompt) in NEUTRAL_PROMPTS.items():
        _, _, _, mat, pat = RELATIONS[rid]
        items.append(dict(
            group="neutral", template=tid, cue_type="neutral", frame=rid, pair=-1, split="all",
            speaker=None, speaker_gender=None, prompt=prompt, corrupt_prompt=None,
            correct_word=None, contrast_word=None, fem_word=mat, masc_word=pat))
    for k, it in enumerate(items):
        it["id"] = k
    return items


def check_tokenization(items, tokenize):
    """Add first-token answers (fem/masc/correct/contrast) and token positions."""
    first = {}
    for _, _, _, mat, pat in RELATIONS.values():
        for w in (mat, pat):
            first[w] = tokenize(" " + w)[0].lstrip("▁")
    for _, _, _, mat, pat in RELATIONS.values():
        assert first[mat] != first[pat], f"{mat} and {pat} share their first token"
    for it in items:
        it["fem"], it["masc"] = first[it["fem_word"]], first[it["masc_word"]]
        it["correct"] = first[it["correct_word"]] if it["correct_word"] else None
        it["contrast"] = first[it["contrast_word"]] if it["contrast_word"] else None
        toks = tokenize(it["prompt"])
        it["n_tokens"] = len(toks)
        for w in (it["fem_word"], it["masc_word"]):
            assert tokenize(it["prompt"] + " " + w)[len(toks)].lstrip("▁") == first[w], (it["prompt"], w)
        if it["corrupt_prompt"]:
            ctoks = tokenize(it["corrupt_prompt"])
            assert len(ctoks) == len(toks), f"length mismatch: {it['prompt']} / {it['corrupt_prompt']}"
            diff = [p for p, (a, b) in enumerate(zip(toks, ctoks)) if a != b]
            assert len(diff) == 1, f"clean/corrupt differ at {diff}: {it['prompt']}"
            it["name_pos"] = diff[0]
    return items


def write_review(items, path):
    rng = random.Random(0)
    lines = ["# Kinship data: please check\n",
             "Mark anything unnatural, ambiguous or wrong. The model must predict the term in [brackets] "
             "(bold in the samples); the prompt stops before it.\n",
             "## Parent words (maternal | paternal)\n"]
    lines += [f"{i+1}. {m} | {p}" for i, (m, p) in enumerate(PARENT_PAIRS)]
    lines.append("\n## Relations (maternal term / paternal term)\n")
    for rid, (rel, post, rel_en, mat, pat) in RELATIONS.items():
        lines.append(f"- {rid}: माँ {post} {rel} → [{mat}]  |  पिता {post} {rel} → [{pat}]  ({rel_en}, {SPLIT_OF_RELATION[rid]})")
    lines.append("\nQuestions: are these the terms you use? (e.g. पिता के बड़े भाई = ताऊ / बड़े पापा, so we "
                 "say छोटे भाई for चाचा.) Is k05/k06 (two-step) natural?\n")
    lines.append("## Context templates\n")
    lines += [f"- {tid} ({ct}): `{t}` … {TAIL[tid]}" for tid, (ct, t) in CUE_TEMPLATES.items()]
    lines.append("\n## No-side prompts (which side does the model default to?)\n")
    lines += [f"- {tid}: `{p}` → [{RELATIONS[rid][3]} / {RELATIONS[rid][4]}]" for tid, (rid, p) in NEUTRAL_PROMPTS.items()]
    lines.append("\n## Random sample of full items (prompt → correct answer)\n")
    pool = [it for it in items if it["group"] == "cue"]
    for it in rng.sample(pool, 24):
        lines.append(f"- [{it['template']}] {it['prompt']} → **{it['correct_word']}** {TAIL[it['template']]}".rstrip())
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    ap.add_argument("--out", default="data/kinship.jsonl")
    args = ap.parse_args()
    items = check_tokenization(build_items(), load_tokenizer(args.tokenizer))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    write_review(items, Path(args.out).with_name("review_kinship.md"))
    counts = {}
    for it in items:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
    print(f"wrote {len(items)} items to {args.out}: {counts}")


if __name__ == "__main__":
    main()
