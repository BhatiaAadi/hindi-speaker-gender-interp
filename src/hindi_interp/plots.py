"""Figures for the report, made from the CSV/JSON files in a results directory.

    python -m hindi_interp.plots --res results/gemma-3-4b-pt --fig report/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = dict(F="#c2185b", M="#1565c0", attn="#ef6c00", mlp="#5e35b1", grey="#9e9e9e", dark="#424242")
W1 = 3.1  # single ACL column width in inches


def load(res, name):
    p = res / name
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


def fig_behaviour(res, fig):
    df = load(res, "behaviour.csv")
    d = df[df.group != "neutral"].copy()
    order = ["T1", "T2", "T3", "T4", "T5", "D1", "D2"]
    labels = ["T1\nquote", "T2\nquote", "T3\nintro", "T4\ndiary", "T5\nletter", "D1\ndist.\nnear",
              "D2\nspkr.\nnear"]
    fig_, ax = plt.subplots(figsize=(W1, 2.0))
    x = np.arange(len(order))
    for k, (g, off) in enumerate((("F", -0.2), ("M", 0.2))):
        acc = [d[(d.template == t) & (d.gender == g)].correct.mean() for t in order]
        ax.bar(x + off, acc, 0.38, color=COL[g], label="female speaker" if g == "F" else "male speaker")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("accuracy (LD > 0)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "behaviour.pdf")
    plt.close(fig_)


# Matplotlib cannot shape Devanagari, so map labels are transliterated.
TRANSLIT = {"<bos>": "<bos>", "फ़": "pho", "ोन": "n", "▁पर": "par", "▁ने": "ne", "▁कहा": "kahā", ",": ",",
            '▁"': '"', "मैं": "maiṃ", "▁मैं": "maiṃ", "▁कल": "kal", "▁बा": "bā", "ज़ार": "zār",
            "नमस्ते": "namaste", "▁मेरा": "merā", "▁नाम": "nām", "▁है": "hai", "।": "|", "▁बीमार": "bīmār",
            "यह": "yah", "▁की": "kī", "▁डाय": "ḍāy", "री": "rī", "▁अभी": "abhī", "▁खाना": "khānā",
            "▁बना": "banā", "प्रिय": "priy", "▁दोस्त": "dost", "▁हूँ": "hū̃", "▁रोज़": "roz", "▁सुबह": "subah",
            "▁पार्क": "pārk", "▁बताया": "batāyā", "▁बाद": "bād", "▁में": "meṃ"}


def translit(tokens, name_pos):
    return ["NAME" if i == name_pos else TRANSLIT.get(t, t.replace("▁", "")) for i, t in enumerate(tokens)]


def pron_index(tokens):
    return max(i for i, t in enumerate(tokens) if t.lstrip("▁") == "मैं")


def fig_resid_map(res, fig, key="T1_f01", direction="denoise"):
    maps = load(res, "patching_resid_maps.json")
    if not maps:
        return
    key = key if key in maps else sorted(maps)[0]
    mp = maps[key]
    a = np.array(mp[direction])
    eff = (a / mp["n"] - mp["corrupt"] / mp["n"]) / (mp["clean"] / mp["n"] - mp["corrupt"] / mp["n"])
    if direction == "noise":
        eff = 1 - eff
    toks = translit(mp["tokens"], mp["name_pos"])
    fig_, ax = plt.subplots(figsize=(W1, 2.3))
    im = ax.imshow(eff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(toks)), toks, rotation=60, fontsize=6, ha="right", rotation_mode="anchor")
    ax.set_ylabel("layer")
    fig_.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="patching effect")
    fig_.tight_layout()
    fig_.savefig(fig / f"resid_map_{key}_{direction}.pdf")
    plt.close(fig_)


def _dev_font():
    from matplotlib import font_manager
    for name in ("Noto Sans Devanagari", "Kohinoor Devanagari", "Devanagari MT", "Lohit Devanagari"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            return name
    return None


def fig_resid_positions(res, fig):
    """Patching effect at the name position vs the last position, per layer, averaged over cells."""
    maps = load(res, "patching_resid_maps.json")
    rows = []
    for key, mp in maps.items():
        a = np.array(mp["denoise"])
        eff = (a - mp["corrupt"]) / (mp["clean"] - mp["corrupt"])
        rows.append((eff[:, mp["name_pos"]], eff[:, pron_index(mp["tokens"])], eff[:, -1]))
    if not rows:
        return
    name, pron, last = (np.mean([r[i] for r in rows], 0) for i in range(3))
    fig_, ax = plt.subplots(figsize=(W1, 1.9))
    ax.plot(name, color=COL["attn"], label="name")
    ax.plot(pron, color=COL["mlp"], label="maiṃ")
    ax.plot(last, color=COL["dark"], label="answer")
    ax.set_xlabel("layer")
    ax.set_ylabel("patching effect")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "resid_positions.pdf")
    plt.close(fig_)
    return dict(name=name.tolist(), last=last.tolist())


def fig_attn_mlp(res, fig):
    """Attention vs MLP outputs per layer, patched at the answer position and at maiṃ. Restoring and
    replacing give identical pooled effects (every pair appears in both orientations), so we show one."""
    s = load(res, "patching_summary.json")["ALL"]
    panels = [("answer position", s["attn_last_den"], s["mlp_last_den"])]
    rp = res / "route_patching.csv"
    if rp.exists():
        d = pd.read_csv(rp)
        d = d[d.group == "cue"]
        panels.append(("pronoun maiṃ", d[d.kind == "attn"].sort_values("layer").effect.values,
                       d[d.kind == "mlp"].sort_values("layer").effect.values))
    fig_, axs = plt.subplots(1, len(panels), figsize=(W1, 1.6), sharey=True, squeeze=False)
    for ax, (title, a, m_) in zip(axs[0], panels):
        x = np.arange(len(a))
        ax.bar(x - 0.2, a, 0.4, color=COL["attn"], label="attention")
        ax.bar(x + 0.2, m_, 0.4, color=COL["mlp"], label="MLP")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("layer")
    axs[0][0].set_ylabel("patching effect")
    axs[0][-1].legend(frameon=False, loc="upper left")
    fig_.tight_layout()
    fig_.savefig(fig / "attn_mlp.pdf")
    plt.close(fig_)


def fig_heads(res, fig):
    s = load(res, "patching_summary.json")["ALL"]
    panels = [("heads at answer position", np.array(s["head_den"]))]
    rp = res / "route_patching.csv"
    if rp.exists():
        d = pd.read_csv(rp)
        d = d[(d.group == "cue") & (d.kind == "head")]
        a = np.zeros((d.layer.max() + 1, d["head"].max() + 1))
        a[d.layer, d["head"]] = d.effect
        panels.append(("heads at maiṃ", a))
    fig_, axs = plt.subplots(1, len(panels), figsize=(W1, 1.7), sharey=True, squeeze=False)
    for ax, (title, a) in zip(axs[0], panels):
        im = ax.imshow(a.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("layer")
    axs[0][0].set_ylabel("head")
    fig_.colorbar(im, ax=axs[0].tolist(), fraction=0.04, pad=0.02, label="patching effect")
    fig_.savefig(fig / "heads.pdf", bbox_inches="tight")
    plt.close(fig_)


def fig_lens(res, fig):
    df = load(res, "logit_lens.csv")
    cols = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
    fig_, ax = plt.subplots(figsize=(W1, 2.1))
    cue = df[(df.group == "cue") & df.ok.astype(bool)]
    for g in ("F", "M"):
        sub = cue[cue.gender == g][cols]
        ax.plot(sub.mean().values, color=COL[g], label=f"{'female' if g == 'F' else 'male'} speaker (correct − contrast)")
    neu = df[df.group == "neutral"][cols]
    ax.plot(neu.mean().values, color=COL["grey"], label="no cue (fem − masc)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("logit-lens LD")
    ax.set_xlim(8, len(cols) - 1)  # earlier layers are not aligned with the output space
    ax.set_ylim(-2.5, 4.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "logit_lens.pdf")
    plt.close(fig_)


def fig_dla(res, fig):
    df = load(res, "dla_components.csv")
    L = len([c for c in df.columns if c.startswith("mlp")])
    fig_, ax = plt.subplots(figsize=(W1, 2.1))
    cue = df[(df.group == "cue") & df.ok.astype(bool)]
    neu = df[df.group == "neutral"]
    x = np.arange(L)
    ax.bar(x - 0.2, [cue[f"attn{l}"].mean() for l in x], 0.4, color=COL["attn"], label="attn (cue, correct dir.)")
    ax.bar(x + 0.2, [cue[f"mlp{l}"].mean() for l in x], 0.4, color=COL["mlp"], label="MLP (cue, correct dir.)")
    ax.plot(x, [neu[f"mlp{l}"].mean() for l in x], color=COL["dark"], marker=".", ms=3, lw=0.8,
            label="MLP (no cue, fem − masc)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("direct effect on LD")
    ax.set_xlim(14.5, L - 0.5)  # all components below layer 15 are ~0
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "dla.pdf")
    plt.close(fig_)


def fig_probes(res, fig):
    df = load(res, "probes.csv")
    if (res / "probes_pronoun.csv").exists():
        df = pd.concat([df, pd.read_csv(res / "probes_pronoun.csv")])
    fig_, ax = plt.subplots(figsize=(W1, 2.0))
    for pos, c, lab in (("name", COL["attn"], "speaker name"), ("pronoun", COL["mlp"], "pronoun maiṃ"),
                        ("last", COL["dark"], "answer position")):
        d = df[df.position == pos]
        if len(d):
            ax.plot(d.layer, d.acc_test, color=c, label=lab)
    d = df[df.position == "last"]
    ax.plot(d.layer, d.acc_control, color=COL["grey"], ls="--", label="control task (seen names)")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy, held-out names")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "probes.pdf")
    plt.close(fig_)


def fig_steering(res, fig):
    df = load(res, "steering.csv")
    if "layer" not in df:
        df["layer"] = load(res, "probe_layer.json")["best_layer_last"]
    layers = sorted(df.layer.unique())
    fig_, axs = plt.subplots(1, len(layers), figsize=(W1, 1.7), sharey=True, squeeze=False)
    for ax, L in zip(axs[0], layers):
        d0 = df[df.layer == L]
        rnd = d0[d0.direction.str.startswith("random")].groupby("alpha").repair_rate
        ax.fill_between(rnd.mean().index, rnd.min().values, rnd.max().values, color=COL["grey"], alpha=0.3,
                        lw=0, label="20 random dir. (range)")
        ax.plot(rnd.mean().index, rnd.mean().values, color=COL["grey"])
        for name, c in (("diff_means", COL["F"]), ("probe", COL["M"])):
            d = d0[d0.direction == name]
            ax.plot(d.alpha, d.repair_rate, color=c, marker="o", ms=3,
                    label="difference of means" if name == "diff_means" else "probe direction")
        ax.set_xscale("log", base=2)
        ax.set_title(f"layer {L}")
        ax.set_xlabel("strength (× class sep.)")
    axs[0][0].set_ylabel("repair rate")
    axs[0][0].legend(frameon=False, loc="upper left")
    fig_.tight_layout()
    fig_.savefig(fig / "steering.pdf")
    plt.close(fig_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/gemma-3-4b-pt")
    ap.add_argument("--fig", default="report/figures")
    args = ap.parse_args()
    res, fig = Path(args.res), Path(args.fig)
    fig.mkdir(parents=True, exist_ok=True)
    for fn in (fig_behaviour, fig_resid_map, fig_resid_positions, fig_attn_mlp, fig_heads, fig_lens, fig_dla,
               fig_probes, fig_steering):
        try:
            fn(res, fig)
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e.filename)


if __name__ == "__main__":
    main()
