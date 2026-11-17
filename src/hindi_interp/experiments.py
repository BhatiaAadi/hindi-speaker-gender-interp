"""Speaker-gender experiments on Gemma-3-4B (pipeline steps 1-7).

    python -m hindi_interp.experiments <step> [--model ...] [--out results]

Steps (run in this order; later steps read earlier outputs):
    sanity      residual-sum and DLA-sum identities on a few prompts
    behaviour   LD / accuracy / top-1 / free generation / flip rate        (RQ1, RQ2, H3)
    lens        logit lens and direct logit attribution (layers, heads)    (RQ3, H3)
    patching    activation patching: residual (layer x position),
                attention / MLP per layer, single heads; both directions    (RQ3, H4)
    probes      linear probes per layer + control task + baselines         (RQ3)
    diagnose    cue readability on wrong items + steering repair           (RQ4, H1)
    heads       candidate speaker heads: attention, distractor test,
                and late-MLP ablation for the masculine default            (H3, H4)
    ambig       probe readability on female names that are also nouns      (H1)
    route       follow-up: the cue's route name -> मैं -> answer; probes, attention and
                patching at मैं; zero-ablation of late MLPs                  (RQ3, H1, H3, H4)
"""
import argparse
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_items
from .model import Model, patch_input, patch_output, set_positions

OUT = Path("results")


# ======================================================================= utils
def cells(items):
    """Group items by (template, frame): every item in a cell has the same token length."""
    g = defaultdict(list)
    for it in items:
        g[(it["template"], it["frame"])].append(it)
    return g


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def cluster_boot(df, value, cluster=("template", "frame"), stat=np.mean, n=1000, seed=0):
    """95% bootstrap CI resampling whole (template, frame) cells, not single items."""
    rng = np.random.default_rng(seed)
    groups = [g[value].to_numpy() for _, g in df.groupby(list(cluster))]
    est = stat(np.concatenate(groups))
    boots = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        boots.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(est), lo=float(lo), hi=float(hi), n=int(sum(len(g) for g in groups)))


def ids_for(m, items, a_key, b_key):
    a = torch.tensor([m.tok_id(it[a_key]) for it in items], device=m.device)
    b = torch.tensor([m.tok_id(it[b_key]) for it in items], device=m.device)
    return a, b


def ld_from_logits(logits, a, b):
    return (logits.gather(1, a[:, None]) - logits.gather(1, b[:, None]))[:, 0]


# ===================================================================== sanity
def step_sanity(m, items, out):
    """Check that (1) components add up to the residual and (2) DLA adds up to the final LD."""
    cue = next(iter(cells([it for it in items if it["group"] == "cue"]).values()))[:8]
    ids = m.encode([it["prompt"] for it in cue])
    c = m.run_with_cache(ids)
    a, b = ids_for(m, cue, "correct", "contrast")
    res = {}
    # (1) resid[l+1] == resid[l] + attn[l] + mlp[l]
    errs = []
    for l in range(m.n_layers):
        dev = c["resid"][l + 1].device
        rec = c["resid"][l].float().to(dev) + c["attn"][l].float().to(dev) + c["mlp"][l].float().to(dev)
        errs.append(((rec - c["resid"][l + 1].float()).norm() / c["resid"][l + 1].float().norm()).item())
    res["resid_identity_max_rel_err"] = max(errs)
    # (2) logits from our own final readout match HF
    ld_hf = ld_from_logits(c["logits"], a, b)
    ld_lens = m.logit_lens(c["resid"][m.n_layers][:, -1], a, b)
    res["final_readout_max_abs_err"] = (ld_hf - ld_lens).abs().max().item()
    # (3) DLA: embed + sum(attn) + sum(mlp) projected on u equals the LD
    u = m.dla_direction(c["resid"][m.n_layers][:, -1], a, b)
    parts = [c["resid"][0][:, -1].float()] + [c["attn"][l][:, -1].float() for l in range(m.n_layers)] \
        + [c["mlp"][l][:, -1].float() for l in range(m.n_layers)]
    parts = [p.to(u.device) for p in parts]
    dla_total = sum((p * u).sum(-1) for p in parts)
    res["dla_sum_max_abs_err"] = (dla_total - ld_hf).abs().max().item()
    # (4) per-head DLA sums to the attention-block DLA
    l = m.n_layers // 2
    head = head_dla(m, c, l, u)
    res["head_sum_vs_attn_max_abs_err"] = (head.sum(1) - (c["attn"][l][:, -1].float().to(u.device) * u).sum(-1)).abs().max().item()
    res["ld_values"] = ld_hf.tolist()
    print(json.dumps(res, indent=2))
    save_json(res, out / "sanity.json")


def head_dla(m, c, l, u):
    """Direct contribution of each head at layer l (last position) to LD: [batch, n_heads].

    attn_out = post_attention_layernorm(sum_h z_h W_O,h). With the norm's RMS factor frozen,
    each head's share is linear: (z_h W_O,h) * inv_rms * (1 + w) * u.
    """
    layer = m.layers[l]
    z = c["z"][l][:, -1].float()                                # [b, H, hd]
    out_device, u = u.device, u.to(z.device)
    W_O = layer.self_attn.o_proj.weight.float()                 # [d, H*hd]
    W_O = W_O.view(m.d_model, m.n_heads, m.head_dim)            # [d, H, hd]
    per_head = torch.einsum("bhk,dhk->bhd", z, W_O)              # [b, H, d]
    o = per_head.sum(1)
    norm = layer.post_attention_layernorm
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))
    inv = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)    # [b, 1]
    scale = (inv * (1.0 + norm.weight.float()))[:, None, :]      # [b, 1, d]
    return (per_head * scale * u[:, None, :]).sum(-1).to(out_device)  # [b, H]


# ================================================================== behaviour
def first_verb(text, fem, masc):
    """First of the two gendered forms that appears as a word in generated text (or None)."""
    for w in text.replace('"', " ").replace(",", " ").replace("।", " ").split():
        if w in (fem, masc):
            return w
    return None


def step_behaviour(m, items, out, gen_tokens=6):
    rows = []
    for (tid, fid), cell in cells(items).items():
        ids = m.encode([it["prompt"] for it in cell])
        logits = m.logits(ids)
        f_ids, m_ids = ids_for(m, cell, "fem", "masc")
        ld_fem = ld_from_logits(logits, f_ids, m_ids)  # logit(fem) - logit(masc)
        top1 = logits.argmax(-1)
        gen = m.hf.generate(input_ids=ids, max_new_tokens=gen_tokens, do_sample=False,
                            attention_mask=torch.ones_like(ids))[:, ids.shape[1]:]
        gen_txt = m.tok.batch_decode(gen, skip_special_tokens=True)
        if cell[0]["corrupt_prompt"] is not None:
            c_logits = m.logits(m.encode([it["corrupt_prompt"] for it in cell]))
            c_ld_fem = ld_from_logits(c_logits, f_ids, m_ids)
        else:
            c_ld_fem = torch.full_like(ld_fem, float("nan"))
        lp = torch.log_softmax(logits, -1)
        for k, it in enumerate(cell):
            sign = 1.0 if it["speaker_gender"] == "F" else -1.0
            row = dict(id=it["id"], group=it["group"], template=tid, cue_type=it["cue_type"], frame=fid,
                       pair=it["pair"], split=it["split"], gender=it["speaker_gender"],
                       ld_fem=ld_fem[k].item(), ld_fem_corrupt=c_ld_fem[k].item(),
                       lp_fem=lp[k, f_ids[k]].item(), lp_masc=lp[k, m_ids[k]].item(),
                       top1=m.tok.decode(top1[k]), gen=gen_txt[k])
            if it["speaker_gender"] is not None:
                row["ld"] = sign * row["ld_fem"]                    # correct - contrast
                row["ld_corrupt"] = sign * row["ld_fem_corrupt"]    # same orientation as clean
                row["correct"] = row["ld"] > 0
                row["top1_correct"] = top1[k].item() == (f_ids[k] if sign > 0 else m_ids[k]).item()
                first = first_verb(gen_txt[k], it["fem"], it["masc"])
                row["gen_has_verb"] = first is not None
                row["gen_correct"] = first == it["correct"]
                row["flipped"] = bool(row["ld"] > 0 and row["ld_corrupt"] < 0)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "behaviour.csv", index=False)
    summarise_behaviour(df, out)
    return df


def summarise_behaviour(df, out):
    s = {}
    cue, dis, neu = (df[df.group == g] for g in ("cue", "distractor", "neutral"))
    for name, d in (("cue", cue), ("distractor", dis)):
        d = d.assign(acc=d.correct.astype(float), top1=d.top1_correct.astype(float),
                     genc=d.gen_correct.astype(float), genv=d.gen_has_verb.astype(float))
        s[name] = {k: cluster_boot(d, k) for k in ("acc", "top1", "genc", "genv", "ld")}
        s[name]["by_gender"] = {g: {k: cluster_boot(d[d.gender == g], k) for k in ("acc", "top1", "genc", "ld")}
                                for g in ("F", "M")}
        s[name]["by_template"] = {t: cluster_boot(d[d.template == t], "acc") for t in sorted(d.template.unique())}
        cc = d[d.correct]
        cc = cc.assign(flip=cc.flipped.astype(float))
        s[name]["flip_rate"] = {"all": cluster_boot(cc, "flip"),
                                "F_to_M": cluster_boot(cc[cc.gender == "F"], "flip"),
                                "M_to_F": cluster_boot(cc[cc.gender == "M"], "flip")}
    s["neutral"] = {"ld_fem": cluster_boot(neu, "ld_fem"),
                    "share_masc_preferred": float((neu.ld_fem < 0).mean()),
                    "by_template": {t: float(g.ld_fem.mean()) for t, g in neu.groupby("template")}}
    by_frame = cue.groupby("frame").apply(lambda g: pd.Series(dict(
        acc=g.correct.mean(), acc_F=g[g.gender == "F"].correct.mean(), acc_M=g[g.gender == "M"].correct.mean())))
    by_frame.to_csv(out / "behaviour_by_frame.csv")
    save_json(s, out / "behaviour_summary.json")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("acc", "top1", "genc", "flip_rate", "ld_fem")}
                      for k, v in s.items()}, indent=1, ensure_ascii=False))


def load_beh(out):
    return pd.read_csv(out / "behaviour.csv")


# ======================================================================= lens
def step_lens(m, items, out):
    beh = load_beh(out).set_index("id")
    lens_rows, comp_rows, head_rows = [], [], []
    for (tid, fid), cell in cells(items).items():
        grp = cell[0]["group"]
        ids = m.encode([it["prompt"] for it in cell])
        c = m.run_with_cache(ids)
        f_ids, m_ids = ids_for(m, cell, "fem", "masc")
        resid_final = c["resid"][m.n_layers][:, -1]
        u = m.dla_direction(resid_final, f_ids, m_ids)  # DLA towards fem - masc
        lens = torch.stack([m.logit_lens(c["resid"][l][:, -1], f_ids, m_ids)
                            for l in range(m.n_layers + 1)], 1)  # [b, L+1]
        comps = [("embed", -1, c["resid"][0][:, -1])] + \
                [("attn", l, c["attn"][l][:, -1]) for l in range(m.n_layers)] + \
                [("mlp", l, c["mlp"][l][:, -1]) for l in range(m.n_layers)]
        comp_vals = torch.stack([(x.float().to(u.device) * u).sum(-1) for _, _, x in comps], 1)  # [b, C]
        heads = torch.stack([head_dla(m, c, l, u) for l in range(m.n_layers)], 1)  # [b, L, H]
        for k, it in enumerate(cell):
            sign = 1.0 if it["speaker_gender"] != "M" else -1.0   # orient to correct (neutral: fem)
            ok = bool(beh.loc[it["id"], "correct"]) if it["speaker_gender"] else True
            meta = dict(id=it["id"], group=grp, template=tid, frame=fid, gender=it["speaker_gender"], ok=ok)
            lens_rows.append({**meta, **{f"L{l}": sign * lens[k, l].item() for l in range(m.n_layers + 1)}})
            comp_rows.append({**meta, **{f"{n}{'' if l < 0 else l}": sign * comp_vals[k, j].item()
                                         for j, (n, l, _) in enumerate(comps)}})
            head_rows.append({**meta, **{f"L{l}H{h}": sign * heads[k, l, h].item()
                                         for l in range(m.n_layers) for h in range(m.n_heads)}})
    pd.DataFrame(lens_rows).to_csv(out / "logit_lens.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(out / "dla_components.csv", index=False)
    pd.DataFrame(head_rows).to_csv(out / "dla_heads.csv", index=False)
    print("lens/DLA written")


# =================================================================== patching
def patch_pairs(beh, items, group="cue"):
    """Items whose clean prompt is answered correctly and whose corrupt prompt flips it."""
    ok = set(beh[(beh.group == group) & beh.flipped.astype(bool)].id)
    return [it for it in items if it["id"] in ok]


def step_patching(m, items, out, max_per_cell=40, map_frames=("f01", "f05", "f08", "f10")):
    beh = load_beh(out)
    pairs = patch_pairs(beh, items, "cue")
    L, H = m.n_layers, m.n_heads
    acc = defaultdict(lambda: defaultdict(float))  # (kind, template) -> key -> summed LD
    resid_maps = {}
    t0 = time.time()
    for (tid, fid), cell in sorted(cells(pairs).items()):
        cell = cell[:max_per_cell]
        n = len(cell)
        clean_ids = m.encode([it["prompt"] for it in cell])
        corr_ids = m.encode([it["corrupt_prompt"] for it in cell])
        a, b = ids_for(m, cell, "correct", "contrast")
        cc = m.run_with_cache(clean_ids)
        rc = m.run_with_cache(corr_ids)
        ld_clean = ld_from_logits(cc["logits"], a, b).sum().item()
        ld_corr = ld_from_logits(rc["logits"], a, b).sum().item()
        S = clean_ids.shape[1]
        name_pos = cell[0]["name_pos"] + 1  # +1 for BOS
        key_base = (tid,)
        acc[key_base]["n"] += n
        acc[key_base]["clean"] += ld_clean
        acc[key_base]["corrupt"] += ld_corr

        def run(hooks, ids):
            return ld_from_logits(m.logits(ids, hooks), a, b).sum().item()

        # Residual stream, every layer x position, both directions (subset of frames: costly).
        if fid in map_frames:
            rmap = np.zeros((2, L, S))
            for l in range(L):
                for p in range(S):
                    rmap[0, l, p] = run([(m.layers[l], "pre", patch_input(set_positions(cc["resid"][l], [p])))], corr_ids)
                    rmap[1, l, p] = run([(m.layers[l], "pre", patch_input(set_positions(rc["resid"][l], [p])))], clean_ids)
            resid_maps[f"{tid}_{fid}"] = dict(denoise=rmap[0].tolist(), noise=rmap[1].tolist(), n=n,
                                             clean=ld_clean, corrupt=ld_corr, name_pos=name_pos, seq_len=S,
                                             tokens=m.tok.convert_ids_to_tokens(clean_ids[0].tolist()))
        # Attention and MLP outputs per layer: at the last position and at all positions.
        for l in range(L):
            for kind, mod in (("attn", m.layers[l].post_attention_layernorm),
                              ("mlp", m.layers[l].post_feedforward_layernorm)):
                for pos_name, pos in (("last", [S - 1]), ("all", None)):
                    acc[key_base][f"{kind}_{pos_name}_den_{l}"] += run(
                        [(mod, "fwd", patch_output(set_positions(cc[kind][l], pos)))], corr_ids)
                    acc[key_base][f"{kind}_{pos_name}_noi_{l}"] += run(
                        [(mod, "fwd", patch_output(set_positions(rc[kind][l], pos)))], clean_ids)
        # Single heads at the last position (input of o_proj, one head slice).
        for l in range(L):
            o_proj = m.layers[l].self_attn.o_proj
            for h in range(H):
                sl = slice(h * m.head_dim, (h + 1) * m.head_dim)
                for direction, src, ids in (("den", cc, corr_ids), ("noi", rc, clean_ids)):
                    zsrc = src["z"][l][:, -1, h]  # [b, hd]

                    def fn(old, zsrc=zsrc, sl=sl):
                        new = old.clone()
                        new[:, -1, sl] = zsrc.to(old.device, old.dtype)
                        return new
                    acc[key_base][f"head_{direction}_{l}_{h}"] += run([(o_proj, "pre", patch_input(fn))], ids)
        print(f"{tid} {fid}: n={n} S={S} t={time.time() - t0:.0f}s", flush=True)
    # Normalised effects per template and pooled.
    summary = {}
    for (tid,), d in list(acc.items()) + [(("ALL",), _pool(acc))]:
        clean, corr = d["clean"], d["corrupt"]
        eff = lambda v, direction: ((v - corr) / (clean - corr)) if direction == "den" else ((clean - v) / (clean - corr))
        s = dict(n=d["n"], ld_clean=clean / d["n"], ld_corrupt=corr / d["n"])
        for kind in ("attn", "mlp"):
            for pos_name in ("last", "all"):
                for direction in ("den", "noi"):
                    s[f"{kind}_{pos_name}_{direction}"] = [eff(d[f"{kind}_{pos_name}_{direction}_{l}"], direction)
                                                          for l in range(L)]
        for direction in ("den", "noi"):
            s[f"head_{direction}"] = [[eff(d[f"head_{direction}_{l}_{h}"], direction) for h in range(H)]
                                      for l in range(L)]
        summary[tid] = s
    save_json(summary, out / "patching_summary.json")
    save_json(resid_maps, out / "patching_resid_maps.json")
    top = sorted(((summary["ALL"]["head_den"][l][h], l, h) for l in range(L) for h in range(H)), reverse=True)[:10]
    print("top heads (denoising effect, layer, head):", [(round(e, 3), l, h) for e, l, h in top])


def _pool(acc):
    tot = defaultdict(float)
    for d in acc.values():
        for k, v in d.items():
            tot[k] += v
    return tot


# ===================================================================== probes
def collect_acts(m, items, positions=("last", "name")):
    """Residual stream (all layers) at the answer position and at the speaker-name position."""
    X = {p: [] for p in positions}
    meta = []
    for _, cell in sorted(cells(items).items()):
        ids = m.encode([it["prompt"] for it in cell])
        c = m.run_with_cache(ids, want=("resid",))
        R = torch.stack(c["resid"], 1)  # [b, L+1, S, d]
        for k, it in enumerate(cell):
            if "last" in positions:
                X["last"].append(R[k, :, -1].float().cpu().numpy())
            if "name" in positions:
                X["name"].append(R[k, :, it["name_pos"] + 1].float().cpu().numpy())
            meta.append(it)
    return {p: np.stack(v) for p, v in X.items()}, meta


class Probe:
    """Standardised L2 logistic regression (sklearn's objective with C), fit full-batch on GPU."""

    def __init__(self, X, y, C=0.1, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.as_tensor(np.asarray(X, np.float32), device=device)
        y = torch.as_tensor(np.asarray(y, np.float32), device=device)
        self.mu, self.sd = X.mean(0), X.std(0, unbiased=False).clamp_min(1e-6)
        Xs = (X - self.mu) / self.sd
        w = torch.zeros(X.shape[1] + 1, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn="strong_wolfe", tolerance_grad=1e-7)

        def closure():
            opt.zero_grad()
            z = Xs @ w[:-1] + w[-1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="sum") \
                + 0.5 / C * (w[:-1] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach()

    def predict(self, X):
        X = torch.as_tensor(np.asarray(X, np.float32), device=self.w.device)
        return (((X - self.mu) / self.sd) @ self.w[:-1] + self.w[-1] > 0).int().cpu().numpy()

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())

    def direction(self):
        """Probe normal in raw activation space."""
        return (self.w[:-1] / self.sd).cpu().numpy()


def fit_probe(Xtr, ytr, Xte, yte, C=0.1):
    clf = Probe(Xtr, ytr, C)
    return clf, clf.score(Xte, yte)


def step_probes(m, items, out, work):
    cue = [it for it in items if it["group"] == "cue"]
    X, meta = collect_acts(m, cue)
    np.savez_compressed(work / "acts_cue.npz", last=X["last"], name=X["name"],
                        ids=np.array([it["id"] for it in meta]))
    y = np.array([it["speaker_gender"] == "F" for it in meta], int)
    split = np.array([it["split"] for it in meta])
    names = np.array([it["speaker"] for it in meta])
    rng = np.random.default_rng(0)
    ctrl_of = {n: int(rng.integers(0, 2)) for n in sorted(set(names))}   # Hewitt & Liang control task
    y_ctrl = np.array([ctrl_of[n] for n in names])
    tr, dv, te = split == "train", split == "dev", split == "test"
    # Within-name split (same names, different items) for the control-task comparison.
    perm = rng.permutation(np.where(tr)[0])
    wtr, wte = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
    rows = []
    for pos in ("last", "name"):
        for l in range(m.n_layers + 1):
            Xl = X[pos][:, l]
            _, acc_dev = fit_probe(Xl[tr], y[tr], Xl[dv], y[dv])
            _, acc_test = fit_probe(Xl[tr], y[tr], Xl[te], y[te])
            _, acc_within = fit_probe(Xl[wtr], y[wtr], Xl[wte], y[wte])
            _, acc_ctrl = fit_probe(Xl[wtr], y_ctrl[wtr], Xl[wte], y_ctrl[wte])
            rows.append(dict(position=pos, layer=l, acc_dev=acc_dev, acc_test=acc_test,
                             acc_within=acc_within, acc_control=acc_ctrl,
                             selectivity=acc_within - acc_ctrl, majority=max(y[te].mean(), 1 - y[te].mean())))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "probes.csv", index=False)
    last = df[df.position == "last"]
    best = int(last.sort_values(["acc_dev", "layer"], ascending=[False, True]).iloc[0].layer)
    save_json(dict(best_layer_last=best), out / "probe_layer.json")
    print("layer chosen on dev:", best)


# =================================================================== diagnose
def use_layer(out, thresh=0.5):
    """First layer at which patching the last-position residual restores > thresh of the effect."""
    maps = json.loads((out / "patching_resid_maps.json").read_text())
    eff = np.mean([[(v[-1] - mp["corrupt"]) / (mp["clean"] - mp["corrupt"]) for v in mp["denoise"]]
                   for mp in maps.values()], 0)
    return int(np.argmax(eff > thresh))


def step_diagnose(m, items, out, work, n_random=20, alphas=(0.5, 1, 2, 4, 8)):
    """Readability and steering at L* (best probe layer on dev) and at the first layer where the
    answer position is causally used (from residual patching)."""
    beh = load_beh(out).set_index("id")
    L_star = json.loads((out / "probe_layer.json").read_text())["best_layer_last"]
    L_use = use_layer(out)
    acts = np.load(work / "acts_cue.npz")
    id2row = {int(i): k for k, i in enumerate(acts["ids"])}
    byid = {it["id"]: it for it in items}
    tr_ids = [i for i in acts["ids"] if byid[int(i)]["split"] == "train"]
    ytr = np.array([byid[int(i)]["speaker_gender"] == "F" for i in tr_ids], int)

    # Wrong items: cue items and distractor items the model gets wrong on the clean prompt.
    wrong = [byid[i] for i in beh.index if beh.loc[i, "group"] in ("cue", "distractor")
             and not bool(beh.loc[i, "correct"])]
    right = [byid[i] for i in beh.index if beh.loc[i, "group"] in ("cue", "distractor")
             and bool(beh.loc[i, "correct"])]
    rng = np.random.default_rng(0)
    right_sample = [right[i] for i in rng.choice(len(right), min(400, len(right)), replace=False)]
    res = dict(L_star=L_star, L_use=L_use, n_wrong=len(wrong), n_right=len(right),
               wrong_by_group=pd.Series([it["template"] for it in wrong]).value_counts().to_dict())
    if not wrong:
        save_json(res, out / "diagnose.json")
        print("no wrong items")
        return
    Xw_all, meta_w = collect_acts(m, wrong, positions=("last",))
    Xr_all, meta_r = collect_acts(m, right_sample, positions=("last",))
    yw = np.array([it["speaker_gender"] == "F" for it in meta_w], int)   # same order as the activations
    yr = np.array([it["speaker_gender"] == "F" for it in meta_r], int)

    rows = []
    for L in sorted({L_star, L_use}):
        Xtr = acts["last"][[id2row[int(i)] for i in tr_ids], L].astype(np.float32)
        clf, _ = fit_probe(Xtr, ytr, Xtr, ytr)
        dom = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        natural = float(np.linalg.norm(dom))             # class-mean separation along the direction
        v_probe = clf.direction()                        # back to raw activation space
        dirs = {"diff_means": dom / natural, "probe": v_probe / np.linalg.norm(v_probe)}
        rs = np.random.default_rng(1)
        for r in range(n_random):
            v = rs.standard_normal(m.d_model)
            dirs[f"random{r}"] = v / np.linalg.norm(v)

        # (a) Is the speaker's gender readable at the answer position on wrong items?
        res[f"L{L}"] = dict(natural_norm=natural,
                            probe_acc_on_wrong=float((clf.predict(Xw_all["last"][:, L]) == yw).mean()),
                            probe_acc_on_right=float((clf.predict(Xr_all["last"][:, L]) == yr).mean()))

        # (b) Steering at layer L, last position, towards the speaker's gender.
        def steer_ld(sub, v, alpha):
            out_ld, out_top = [], []
            for _, cell in sorted(cells(sub).items()):
                ids = m.encode([it["prompt"] for it in cell])
                a, b = ids_for(m, cell, "correct", "contrast")
                sign = torch.tensor([1.0 if it["speaker_gender"] == "F" else -1.0 for it in cell], device=m.device)
                vec = torch.tensor(v, device=m.device, dtype=torch.float32)

                def fn(old, sign=sign, vec=vec):
                    new = old.clone()
                    new[:, -1] += (alpha * natural * sign[:, None] * vec[None]).to(old.device, old.dtype)
                    return new
                lg = m.logits(ids, [(m.layers[L], "pre", patch_input(fn))])
                out_ld += ld_from_logits(lg, a, b).tolist()
                out_top += (lg.argmax(-1) == a).tolist()
            return np.array(out_ld), np.array(out_top)

        for name, v in dirs.items():
            for alpha in alphas:
                ldw, topw = steer_ld(wrong, v, alpha)
                ldr, topr = steer_ld(right_sample, v, alpha)
                rows.append(dict(layer=L, direction=name, alpha=alpha, repair_rate=float((ldw > 0).mean()),
                                 repair_top1=float(topw.mean()), right_stay_right=float((ldr > 0).mean()),
                                 right_top1=float(topr.mean())))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "steering.csv", index=False)
    for L, d in df.groupby("layer"):
        rnd = d[d.direction.str.startswith("random")].groupby("alpha").repair_rate.agg(["mean", "max"])
        res[f"L{L}"]["random_repair"] = rnd.to_dict()
        res[f"L{L}"]["repair"] = d[~d.direction.str.startswith("random")].to_dict(orient="records")
    save_json(res, out / "diagnose.json")
    print(json.dumps(res, indent=1)[:4000])


# ================================================================== ambiguity
# Female names that are also common nouns (poem, line, border, worship). Found in the error
# analysis after the behaviour step, not chosen in advance.
AMBIGUOUS = ("कविता", "रेखा", "सीमा", "पूजा")


def step_ambig(m, items, out, work):
    """Is the speaker's gender linearly readable on the noun-like names, per template?
    Probes are trained on train-split pairs that do not contain an ambiguous name."""
    acts = np.load(work / "acts_cue.npz")
    beh = load_beh(out).set_index("id")
    # Same layer at both positions: the best name-position layer on dev is the embedding, which
    # cannot depend on the context the name appears in.
    L_star = json.loads((out / "probe_layer.json").read_text())["best_layer_last"]
    byid = {it["id"]: it for it in items}
    meta = [byid[int(i)] for i in acts["ids"]]
    y = np.array([it["speaker_gender"] == "F" for it in meta], int)
    amb_pairs = {it["pair"] for it in meta if it["speaker"] in AMBIGUOUS}
    train = np.array([it["split"] == "train" and it["pair"] not in amb_pairs for it in meta])
    fem = np.array([it["speaker_gender"] == "F" for it in meta])
    amb = np.array([it["speaker"] in AMBIGUOUS for it in meta])
    tmpl = np.array([it["template"] for it in meta])
    model_ok = np.array([bool(beh.loc[it["id"], "correct"]) for it in meta])
    rows = []
    for pos, L in (("last", L_star), ("name", L_star)):
        X = acts[pos][:, L]
        clf = Probe(X[train], y[train])
        pred = clf.predict(X)
        for t in sorted(set(tmpl)):
            for grp, mask in (("ambiguous", fem & amb), ("other_female", fem & ~amb & ~train),
                              ("male", ~fem & ~train)):
                sel = mask & (tmpl == t)
                if sel.any():
                    rows.append(dict(position=pos, layer=L, template=t, names=grp, n=int(sel.sum()),
                                     model_acc=float(model_ok[sel].mean()),
                                     probe_acc=float((pred[sel] == y[sel]).mean()),
                                     probe_acc_model_wrong=float((pred[sel & ~model_ok] == y[sel & ~model_ok]).mean())
                                     if (sel & ~model_ok).any() else None))
    df = pd.DataFrame(rows)
    df.to_csv(out / "ambiguity.csv", index=False)
    print(df.to_string())


# ====================================================================== route
def pron_pos(m, prompt):
    """Index (with BOS) of the frame's first-person pronoun: the last मैं token of the prompt."""
    toks = m.tok.convert_ids_to_tokens(m.tok(prompt).input_ids)
    return max(i for i, t in enumerate(toks) if t.lstrip("▁") == "मैं")


def step_route(m, items, out, work, max_per_cell=40, map_frames=("f01", "f05", "f08", "f10"),
               lens_layers=(10, 15, 20, 25, 26, 30)):
    """Follow-up to the residual patching maps, which show the cue moving name -> मैं -> answer.
    (a) probes at the मैं position; (b) where the cue is lost for the noun-like names;
    (c) attention to मैं and to the names; (d) components and heads that write the cue into मैं;
    (e) zero / resample ablations of the late MLPs for the masculine default."""
    beh = load_beh(out).set_index("id")
    L, H = m.n_layers, m.n_heads
    byid = {it["id"]: it for it in items}
    cue = [it for it in items if it["group"] == "cue"]

    # (a) residual stream at मैं, every layer, and probes trained on train names.
    X, meta = [], []
    for _, cell in sorted(cells(cue).items()):
        p = pron_pos(m, cell[0]["prompt"])
        c = m.run_with_cache(m.encode([it["prompt"] for it in cell]), want=("resid",))
        X.append(torch.stack(c["resid"], 1)[:, :, p].float().cpu().numpy())
        meta += cell
    X = np.concatenate(X)
    y = np.array([it["speaker_gender"] == "F" for it in meta], int)
    split = np.array([it["split"] for it in meta])
    tr, dv, te = split == "train", split == "dev", split == "test"
    rows = []
    for l in range(L + 1):
        rows.append(dict(position="pronoun", layer=l, acc_dev=fit_probe(X[tr, l], y[tr], X[dv, l], y[dv])[1],
                         acc_test=fit_probe(X[tr, l], y[tr], X[te, l], y[te])[1]))
    pd.DataFrame(rows).to_csv(out / "probes_pronoun.csv", index=False)
    print("pronoun probes:", [round(r["acc_test"], 3) for r in rows], flush=True)

    # (b) Noun-like names: is the speaker's gender readable at the name, at मैं, at the answer?
    acts = np.load(work / "acts_cue.npz")
    row_of = {int(i): k for k, i in enumerate(acts["ids"])}
    order = [row_of[it["id"]] for it in meta]              # align saved activations with `meta`
    by_pos = {"name": acts["name"][order], "pronoun": X, "last": acts["last"][order]}
    amb_pairs = {it["pair"] for it in meta if it["speaker"] in AMBIGUOUS}
    train = np.array([it["split"] == "train" and it["pair"] not in amb_pairs for it in meta])
    amb = np.array([it["speaker"] in AMBIGUOUS for it in meta])
    fem = y == 1
    t4 = np.array([it["template"] == "T4" for it in meta])
    groups = {"T4_ambiguous": amb & t4, "other_ambiguous": amb & ~t4, "T4_other_female": fem & ~amb & ~train & t4}
    rows = []
    for pos, Xp in by_pos.items():
        for l in [l for l in lens_layers if l <= L]:
            clf = Probe(Xp[train, l], y[train])
            pred = clf.predict(Xp[:, l])
            for g, sel in groups.items():
                rows.append(dict(position=pos, layer=l, names=g, n=int(sel.sum()),
                                 probe_acc=float((pred[sel] == y[sel]).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(out / "route_ambiguity.csv", index=False)
    print(df.pivot_table(index=["names", "position"], columns="layer", values="probe_acc").round(2).to_string())

    # (c) Attention from the answer position and from मैं, per layer and head.
    att = defaultdict(lambda: np.zeros((L, H)))
    cnt = defaultdict(int)
    for (tid, fid), cell in sorted(cells([it for it in items if it["group"] in ("cue", "distractor")]).items()):
        ids = m.encode([it["prompt"] for it in cell])
        c = m.run_with_cache(ids, want=(), attn_patterns=True)
        p, s = pron_pos(m, cell[0]["prompt"]), cell[0]["name_pos"] + 1
        d = cell[0]["distractor_pos"] + 1 if "distractor_pos" in cell[0] else None
        grp = "cue" if tid.startswith("T") else tid
        pat = torch.stack(c["pattern"], 0)                  # [L, b, H, q, k]
        pairs_qk = [("last", -1, "pron", p), ("last", -1, "speaker", s), ("pron", p, "speaker", s)]
        if d is not None:
            pairs_qk += [("last", -1, "distractor", d), ("pron", p, "distractor", d)]
        for qn, qi, kn, ki in pairs_qk:
            att[f"{grp}:{qn}->{kn}"] += pat[:, :, :, qi, ki].sum(1).cpu().numpy()
        cnt[grp] += len(cell)
    att_mean = {k: (v / cnt[k.split(":")[0]]).tolist() for k, v in att.items()}
    save_json(att_mean, out / "route_attention.json")

    # (d) Patching at the मैं position: attention / MLP output per layer and single heads.
    eff_rows = []
    for group in ("cue", "distractor"):
        pairs = [it for it in patch_pairs(beh.reset_index(), items, group) if it["frame"] in map_frames]
        tot = defaultdict(float)
        for (tid, fid), cell in sorted(cells(pairs).items()):
            cell = cell[:max_per_cell]
            clean_ids = m.encode([it["prompt"] for it in cell])
            corr_ids = m.encode([it["corrupt_prompt"] for it in cell])
            a, b = ids_for(m, cell, "correct", "contrast")
            p = pron_pos(m, cell[0]["prompt"])
            cc = m.run_with_cache(clean_ids)
            tot["clean"] += ld_from_logits(cc["logits"], a, b).sum().item()
            tot["corrupt"] += ld_from_logits(m.logits(corr_ids), a, b).sum().item()

            def run(hooks):
                return ld_from_logits(m.logits(corr_ids, hooks), a, b).sum().item()
            for l in range(L):
                tot[f"attn_{l}"] += run([(m.layers[l].post_attention_layernorm, "fwd",
                                          patch_output(set_positions(cc["attn"][l], [p])))])
                tot[f"mlp_{l}"] += run([(m.layers[l].post_feedforward_layernorm, "fwd",
                                         patch_output(set_positions(cc["mlp"][l], [p])))])
                tot[f"resid_{l}"] += run([(m.layers[l], "pre", patch_input(set_positions(cc["resid"][l], [p])))])
                for h in range(H):
                    sl = slice(h * m.head_dim, (h + 1) * m.head_dim)

                    def fn(old, zsrc=cc["z"][l][:, p, h], sl=sl):
                        new = old.clone()
                        new[:, p, sl] = zsrc.to(old.device, old.dtype)
                        return new
                    tot[f"head_{l}_{h}"] += run([(m.layers[l].self_attn.o_proj, "pre", patch_input(fn))])
            print(f"route patch {group} {tid} {fid}", flush=True)
        norm = lambda v: (v - tot["corrupt"]) / (tot["clean"] - tot["corrupt"])
        for l in range(L):
            for kind in ("attn", "mlp", "resid"):
                eff_rows.append(dict(group=group, kind=kind, layer=l, head=-1, effect=norm(tot[f"{kind}_{l}"])))
            for h in range(H):
                eff_rows.append(dict(group=group, kind="head", layer=l, head=h, effect=norm(tot[f"head_{l}_{h}"])))
    pe = pd.DataFrame(eff_rows)
    pe.to_csv(out / "route_patching.csv", index=False)
    top = pe[(pe.kind == "head") & (pe.group == "cue")].nlargest(8, "effect")
    print("heads writing into मैं:\n", top.to_string(), flush=True)

    # (e) Masculine default on no-cue prompts: zero-ablate late MLPs at the answer position.
    sets = {"32": [32], "26,29,30": [26, 29, 30], "26,29,30,32": [26, 29, 30, 32],
            "control 19-22": [19, 20, 21, 22]}
    neutral = [it for it in items if it["group"] == "neutral"]
    res = {}
    sets = {k: v for k, v in sets.items() if max(v) < L}
    for name, mlps in [("none", [])] + list(sets.items()):
        lds = []
        for _, cell in sorted(cells(neutral).items()):
            ids = m.encode([it["prompt"] for it in cell])
            f_ids, m_ids = ids_for(m, cell, "fem", "masc")
            hooks = [(m.layers[l].post_feedforward_layernorm, "fwd",
                      patch_output(lambda old: torch.cat([old[:, :-1], torch.zeros_like(old[:, -1:])], 1)))
                     for l in mlps]
            lds += ld_from_logits(m.logits(ids, hooks), f_ids, m_ids).tolist()
        res[name] = dict(ld_fem=float(np.mean(lds)), share_masc=float(np.mean(np.array(lds) < 0)))
        print("zero-ablate", name, res[name], flush=True)
    save_json(res, out / "route_default_ablation.json")


# ====================================================================== heads
def step_heads(m, items, out, top_k=6):
    beh = load_beh(out).set_index("id")
    patch = json.loads((out / "patching_summary.json").read_text())["ALL"]
    eff = np.array(patch["head_den"])
    flat = sorted(((eff[l, h], l, h) for l in range(m.n_layers) for h in range(m.n_heads)), reverse=True)
    top = [(int(l), int(h)) for _, l, h in flat[:top_k]]
    rows = []
    for (tid, fid), cell in sorted(cells([it for it in items if it["group"] in ("cue", "distractor")]).items()):
        ids = m.encode([it["prompt"] for it in cell])
        c = m.run_with_cache(ids, want=("resid", "z"), attn_patterns=True)
        f_ids, m_ids = ids_for(m, cell, "fem", "masc")
        u = m.dla_direction(c["resid"][m.n_layers][:, -1], f_ids, m_ids)
        dla = {l: head_dla(m, c, l, u) for l in {l for l, _ in top}}
        for k, it in enumerate(cell):
            sign = 1.0 if it["speaker_gender"] == "F" else -1.0
            sp = it["prompt"]
            row = dict(id=it["id"], group=it["group"], template=tid, frame=fid,
                       gender=it["speaker_gender"], correct=bool(beh.loc[it["id"], "correct"]))
            s_pos = it["name_pos"] + 1
            d_pos = it.get("distractor_pos")
            for l, h in top:
                pat = c["pattern"][l][k, h, -1]  # attention from the answer position
                row[f"L{l}H{h}_att_speaker"] = pat[s_pos].item()
                if d_pos is not None:
                    row[f"L{l}H{h}_att_distractor"] = pat[d_pos + 1].item()
                row[f"L{l}H{h}_dla"] = sign * dla[l][k, h].item()
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "heads.csv", index=False)
    s = dict(top_heads=top, top_effects=[float(e) for e, _, _ in flat[:top_k]])
    for l, h in top:
        key = f"L{l}H{h}"
        d = df[df.group == "distractor"]
        s[key] = dict(
            att_speaker_cue=float(df[df.group == "cue"][f"{key}_att_speaker"].mean()),
            dla_cue=float(df[df.group == "cue"][f"{key}_dla"].mean()),
            att_speaker_D1=float(d[d.template == "D1"][f"{key}_att_speaker"].mean()),
            att_distractor_D1=float(d[d.template == "D1"][f"{key}_att_distractor"].mean()),
            att_speaker_D2=float(d[d.template == "D2"][f"{key}_att_speaker"].mean()),
            att_distractor_D2=float(d[d.template == "D2"][f"{key}_att_distractor"].mean()),
            share_more_on_speaker_D1=float((d[d.template == "D1"][f"{key}_att_speaker"] >
                                            d[d.template == "D1"][f"{key}_att_distractor"]).mean()))
    s["mlp_default_ablation"] = mlp_default_ablation(m, items, out)
    save_json(s, out / "heads_summary.json")
    print(json.dumps(s, indent=1))


def mlp_default_ablation(m, items, out, k=4):
    """H3: replace the late MLPs that push hardest towards the masculine form (on neutral
    prompts) with their mean output over gender-balanced cue prompts of the same frame."""
    comp = pd.read_csv(out / "dla_components.csv")
    neu = comp[comp.group == "neutral"]
    late = [l for l in range(m.n_layers // 2, m.n_layers)]
    means = {l: neu[f"mlp{l}"].mean() for l in late}                     # oriented fem - masc
    pick = sorted(late, key=lambda l: means[l])[:k]                       # most masculine-pushing
    neutral = [it for it in items if it["group"] == "neutral"]
    cue = [it for it in items if it["group"] == "cue" and it["template"] == "T1"]
    res = dict(mlps=pick, dla=[float(means[l]) for l in pick])
    before, after = [], []
    for (tid, fid), cell in sorted(cells(neutral).items()):
        ref = [it for it in cue if it["frame"] == fid]
        rc = m.run_with_cache(m.encode([it["prompt"] for it in ref]), want=("mlp",))
        mean_out = {l: rc["mlp"][l][:, -1].float().mean(0) for l in pick}
        ids = m.encode([it["prompt"] for it in cell])
        f_ids, m_ids = ids_for(m, cell, "fem", "masc")
        before += ld_from_logits(m.logits(ids), f_ids, m_ids).tolist()
        hooks = []
        for l in pick:
            def fn(old, mo=mean_out[l]):
                new = old.clone()
                new[:, -1] = mo.to(old.device, old.dtype)
                return new
            hooks.append((m.layers[l].post_feedforward_layernorm, "fwd", patch_output(fn)))
        after += ld_from_logits(m.logits(ids, hooks), f_ids, m_ids).tolist()
    res.update(ld_fem_before=float(np.mean(before)), ld_fem_after=float(np.mean(after)),
               share_masc_before=float(np.mean(np.array(before) < 0)),
               share_masc_after=float(np.mean(np.array(after) < 0)))
    return res


# ======================================================================= main
STEPS = dict(sanity=step_sanity, behaviour=step_behaviour, lens=step_lens, patching=step_patching,
             probes=step_probes, diagnose=step_diagnose, heads=step_heads, ambig=step_ambig,
             route=step_route)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=list(STEPS))
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--data", default="data/speaker_gender.jsonl")
    ap.add_argument("--out", default="results/gemma-3-4b-pt")
    ap.add_argument("--work", default="/tmp/hindi_interp_work", help="large intermediate files")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0, help="debug: keep only this many items per group")
    ap.add_argument("--multi-gpu", action="store_true", help="split the model over all visible GPUs")
    args = ap.parse_args()
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    items = load_items(args.data)
    if args.limit:
        keep, cnt = [], defaultdict(int)
        for it in items:  # keep whole cells so batches stay rectangular
            key = (it["group"], it["template"], it["frame"])
            if it["frame"] in ("f01", "f05") and cnt[key] < args.limit:
                keep.append(it)
                cnt[key] += 1
        items = keep
    m = Model(args.model, dtype=getattr(torch, args.dtype), multi_gpu=args.multi_gpu)
    print(f"loaded {args.model}: L={m.n_layers} H={m.n_heads} d={m.d_model} on {m.device}", flush=True)
    failed = []
    for s in args.steps:
        t = time.time()
        fn = STEPS[s]
        try:  # keep going so one broken step does not cost the GPU time of the others
            if s in ("probes", "diagnose", "ambig", "route"):
                fn(m, items, out, work)
            else:
                fn(m, items, out)
            print(f"== {s} done in {time.time() - t:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(s)
            print(f"== {s} FAILED after {time.time() - t:.0f}s", flush=True)
    if failed:
        raise SystemExit(f"failed steps: {failed}")


if __name__ == "__main__":
    main()
