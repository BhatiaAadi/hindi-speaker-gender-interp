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
COL = dict(F="#c2185b", M="#1565c0", attn="#ef6c00", mlp="#2e7d32", grey="#9e9e9e", dark="#424242")
W1 = 3.1  # single ACL column width in inches


def load(res, name):
    p = res / name
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


def fig_behaviour(res, fig):
    df = load(res, "behaviour.csv")
    d = df[df.group != "neutral"].copy()
    order = ["T1", "T2", "T3", "T4", "T5", "D1", "D2"]
    labels = ["quoted\n(T1)", "quoted\n(T2)", "intro\n(T3)", "diary\n(T4)", "letter\n(T5)",
              "distr.\nnear (D1)", "speaker\nnear (D2)"]
    fig_, ax = plt.subplots(figsize=(W1, 1.8))
    x = np.arange(len(order))
    for k, (g, off) in enumerate((("F", -0.2), ("M", 0.2))):
        acc = [d[(d.template == t) & (d.gender == g)].correct.mean() for t in order]
        ax.bar(x + off, acc, 0.38, color=COL[g], label="female speaker" if g == "F" else "male speaker")
    ax.axhline(0.5, color=COL["grey"], lw=0.8, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy (LD > 0)")
    ax.legend(loc="lower left", ncol=2, frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "behaviour.pdf")
    plt.close(fig_)


def fig_resid_map(res, fig, key=None, direction="denoise"):
    maps = load(res, "patching_resid_maps.json")
    if not maps:
        return
    key = key if key in maps else sorted(maps)[0]
    mp = maps[key]
    a = np.array(mp[direction])
    eff = (a / mp["n"] - mp["corrupt"] / mp["n"]) / (mp["clean"] / mp["n"] - mp["corrupt"] / mp["n"])
    if direction == "noise":
        eff = 1 - eff
    toks = [t.replace("▁", "_") for t in mp["tokens"]]
    fig_, ax = plt.subplots(figsize=(W1, 2.4))
    im = ax.imshow(eff, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(toks)), toks, rotation=70, fontsize=6, fontname=_dev_font())
    ax.set_ylabel("layer")
    ax.set_title(f"residual patching ({'restore clean' if direction == 'denoise' else 'insert corrupt'}), {key}")
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
        rows.append((eff[:, mp["name_pos"]], eff[:, -1]))
    if not rows:
        return
    name = np.mean([r[0] for r in rows], 0)
    last = np.mean([r[1] for r in rows], 0)
    fig_, ax = plt.subplots(figsize=(W1, 1.7))
    ax.plot(name, color=COL["F"], label="speaker-name position")
    ax.plot(last, color=COL["dark"], label="answer (last) position")
    ax.set_xlabel("layer")
    ax.set_ylabel("patching effect")
    ax.legend(frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "resid_positions.pdf")
    plt.close(fig_)
    return dict(name=name.tolist(), last=last.tolist())


def fig_attn_mlp(res, fig):
    s = load(res, "patching_summary.json")["ALL"]
    fig_, axs = plt.subplots(1, 2, figsize=(2 * W1, 1.7), sharey=True)
    for ax, pos in zip(axs, ("last", "all")):
        for kind in ("attn", "mlp"):
            ax.plot(s[f"{kind}_{pos}_den"], color=COL[kind], label=f"{kind} (restore)")
            ax.plot(s[f"{kind}_{pos}_noi"], color=COL[kind], ls="--", label=f"{kind} (replace)")
        ax.set_title(f"{'answer position only' if pos == 'last' else 'all positions'}")
        ax.set_xlabel("layer")
    axs[0].set_ylabel("patching effect")
    axs[1].legend(frameon=False, ncol=2)
    fig_.tight_layout()
    fig_.savefig(fig / "attn_mlp.pdf")
    plt.close(fig_)


def fig_heads(res, fig):
    s = load(res, "patching_summary.json")["ALL"]
    a = np.array(s["head_den"])
    fig_, ax = plt.subplots(figsize=(W1, 2.2))
    im = ax.imshow(a.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-np.abs(a).max(), vmax=np.abs(a).max())
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    fig_.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="patching effect (restore)")
    fig_.tight_layout()
    fig_.savefig(fig / "heads.pdf")
    plt.close(fig_)


def fig_lens(res, fig):
    df = load(res, "logit_lens.csv")
    cols = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
    fig_, ax = plt.subplots(figsize=(W1, 1.7))
    cue = df[(df.group == "cue") & df.ok.astype(bool)]
    for g in ("F", "M"):
        sub = cue[cue.gender == g][cols]
        ax.plot(sub.mean().values, color=COL[g], label=f"{'female' if g == 'F' else 'male'} speaker (correct − contrast)")
    neu = df[df.group == "neutral"][cols]
    ax.plot(neu.mean().values, color=COL["grey"], label="no cue (fem − masc)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("logit-lens LD")
    ax.legend(frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "logit_lens.pdf")
    plt.close(fig_)


def fig_dla(res, fig):
    df = load(res, "dla_components.csv")
    L = len([c for c in df.columns if c.startswith("mlp")])
    fig_, ax = plt.subplots(figsize=(W1, 1.7))
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
    ax.legend(frameon=False)
    fig_.tight_layout()
    fig_.savefig(fig / "dla.pdf")
    plt.close(fig_)


def fig_probes(res, fig):
    df = load(res, "probes.csv")
    fig_, ax = plt.subplots(figsize=(W1, 1.8))
    for pos, ls in (("last", "-"), ("name", ":")):
        d = df[df.position == pos]
        ax.plot(d.layer, d.acc_test, color=COL["dark"], ls=ls, label=f"probe, held-out names ({pos} pos.)")
    d = df[df.position == "last"]
    ax.plot(d.layer, d.acc_control, color=COL["grey"], label="control task (last pos.)")
    ax.axhline(0.5, color=COL["grey"], lw=0.6, ls="--")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, loc="lower right")
    fig_.tight_layout()
    fig_.savefig(fig / "probes.pdf")
    plt.close(fig_)


def fig_steering(res, fig):
    df = load(res, "steering.csv")
    fig_, ax = plt.subplots(figsize=(W1, 1.7))
    rnd = df[df.direction.str.startswith("random")].groupby("alpha").repair_rate
    ax.fill_between(rnd.mean().index, rnd.min().values, rnd.max().values, color=COL["grey"], alpha=0.3,
                    label="20 random directions (range)")
    ax.plot(rnd.mean().index, rnd.mean().values, color=COL["grey"])
    for name, c in (("diff_means", COL["F"]), ("probe", COL["M"])):
        d = df[df.direction == name]
        ax.plot(d.alpha, d.repair_rate, color=c, marker="o", ms=3,
                label="difference of means" if name == "diff_means" else "probe direction")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("steering strength (× natural class separation)")
    ax.set_ylabel("repair rate")
    ax.legend(frameon=False)
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
