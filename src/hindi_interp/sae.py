"""Sparse-autoencoder analysis with Gemma Scope 2 residual-stream SAEs (google/gemma-scope-2-4b-pt).

At three sites per task, the cue word, the token where the residual maps show the cue parked
in the middle layers, and the answer position:
    speaker gender: name (layer 9), मैं (layer 20), answer (layer 26)
    honorifics:     addressee (layer 9), quote mark " (layer 22), answer (layer 26)
("layer L" = output of decoder layer L, the SAE's hook point, = our resid[L + 1].)

(0) fidelity: replace the activation by its SAE reconstruction; how much LD survives?
(1) features: which features differ most between the two classes (selected on training
    names/addressees), how well each one alone separates held-out items, and which output
    tokens its decoder direction promotes;
(2) causal: in the corrupt run, patch only the top-k features' difference (clean - corrupt)
    into the site, against random active features, all features, and the full residual;
    and zero the top-k features in the clean run;
(3) across tasks (honorific run): do the speaker-gender features at the answer position
    overlap with the honorific ones?
(4) speaker-gender errors: is the top female feature present at the name, at मैं and at the
    answer position on the wrong (mostly noun-like) names?
Writes sae_summary.json and sae_features.csv in the results directory.
"""
import json

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

from .experiments import AMBIGUOUS, cells, ids_for, ld_from_logits, load_beh, patch_pairs, save_json
from .model import patch_input

REPO = "google/gemma-scope-2-4b-pt"
SITES = {
    "speaker_gender": [("name", 9), ("pronoun", 20), ("answer", 26)],
    "honorific": [("addressee", 9), ("quote", 22), ("answer", 26)],
}
KS = (1, 2, 5, 10, 20, 50)


class SAE:
    def __init__(self, layer, device, width="16k", l0="big"):
        path = hf_hub_download(REPO, f"resid_post_all/layer_{layer}_width_{width}_l0_{l0}/params.safetensors")
        p = {k.lower(): v.float().to(device) for k, v in load_file(path).items()}
        self.W_enc, self.W_dec = p["w_enc"], p["w_dec"]
        if self.W_enc.shape[0] != self.W_dec.shape[1]:  # stored as [n, d]
            self.W_enc = self.W_enc.T
        self.b_enc, self.b_dec, self.thr = p["b_enc"], p["b_dec"], p["threshold"]
        self.n = self.W_dec.shape[0]

    def encode(self, x):
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.thr)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec


def site_pos(m, it, site):
    """Index (with BOS) of the site's token in the item's prompt; -1 for the answer position."""
    if site == "answer":
        return -1
    if site in ("name", "addressee"):
        return it["name_pos"] + 1
    toks = m.tok.convert_ids_to_tokens(m.tok(it["prompt"]).input_ids)
    if site == "pronoun":
        return max(i for i, t in enumerate(toks) if t.lstrip("▁") == "मैं")
    return toks.index('▁"')


def add_at(pos, delta):
    """Pre-hook fn adding delta [b, d] to the residual stream at one position."""
    def fn(old):
        new = old.clone()
        new[:, pos] += delta.to(old.device, old.dtype)
        return new
    return fn


@torch.no_grad()
def collect(m, items, layer, site, sae):
    """SAE activations [N, n] (CPU, float32) at the site, plus raw activations' FVU."""
    acts, meta, err, var = [], [], 0.0, 0.0
    for _, cell in sorted(cells(items).items()):
        c = m.run_with_cache(m.encode([it["prompt"] for it in cell]), want=("resid",))
        p = site_pos(m, cell[0], site)
        x = c["resid"][layer + 1][:, p].float()
        a = sae.encode(x)
        err += (sae.decode(a) - x).pow(2).sum().item()
        var += (x - x.mean(0)).pow(2).sum().item()
        acts.append(a.cpu())
        meta += cell
    return torch.cat(acts), meta, 1 - err / var


def rank_features(A, y, sel):
    """Mean activation difference (class 1 - class 0) on the selection set, per feature."""
    A1, A0 = A[sel & (y == 1)], A[sel & (y == 0)]
    return (A1.mean(0) - A0.mean(0)).numpy()


@torch.no_grad()
def fidelity(m, items, layer, site, sae, per_cell=20):
    """LD with the site replaced by its SAE reconstruction, relative to the clean LD."""
    ld, ld_rec, ok_rec = [], [], []
    for _, cell in sorted(cells(items).items()):
        cell = cell[:per_cell]
        ids = m.encode([it["prompt"] for it in cell])
        a, b = ids_for(m, cell, "correct", "contrast")
        c = m.run_with_cache(ids, want=("resid",))
        p = site_pos(m, cell[0], site)
        x = c["resid"][layer + 1][:, p].float()
        delta = sae.decode(sae.encode(x)) - x
        ld += ld_from_logits(c["logits"], a, b).tolist()
        r = ld_from_logits(m.logits(ids, [(m.layers[layer + 1], "pre", patch_input(add_at(p, delta)))]), a, b)
        ld_rec += r.tolist()
        ok_rec += (r > 0).tolist()
    return dict(ld_clean=float(np.mean(ld)), ld_recon=float(np.mean(ld_rec)),
                ld_kept=float(np.mean(ld_rec) / np.mean(ld)), acc_recon=float(np.mean(ok_rec)))


@torch.no_grad()
def feature_patching(m, pairs, layer, site, sae, top, rng, per_cell=20, n_random=10):
    """Pooled normalised effect of patching feature differences (clean - corrupt) into the corrupt
    run at the site, and of zeroing the top features in the clean run."""
    tot = {}
    kept, still_ok = {k: [] for k in KS}, {k: [] for k in KS}

    def acc(key, v):
        tot[key] = tot.get(key, 0.0) + v
    for _, cell in sorted(cells(pairs).items()):
        cell = cell[:per_cell]
        clean = m.encode([it["prompt"] for it in cell])
        corr = m.encode([it["corrupt_prompt"] for it in cell])
        a, b = ids_for(m, cell, "correct", "contrast")
        p = site_pos(m, cell[0], site)
        cc = m.run_with_cache(clean, want=("resid",))
        cr = m.run_with_cache(corr, want=("resid",))
        xc, xr = cc["resid"][layer + 1][:, p].float(), cr["resid"][layer + 1][:, p].float()
        ac, ar = sae.encode(xc), sae.encode(xr)
        ld_c = ld_from_logits(cc["logits"], a, b)
        acc("clean", ld_c.sum().item())
        acc("corrupt", ld_from_logits(cr["logits"], a, b).sum().item())

        def run(ids, delta):
            hook = [(m.layers[layer + 1], "pre", patch_input(add_at(p, delta)))]
            return ld_from_logits(m.logits(ids, hook), a, b)
        acc("full", run(corr, xc - xr).sum().item())
        acc("all_features", run(corr, (ac - ar) @ sae.W_dec).sum().item())
        active = torch.nonzero(((ac > 0) | (ar > 0)).any(0)).flatten().cpu().numpy()
        pool = np.setdiff1d(active, top[:max(KS)])
        for k in KS:
            S = torch.as_tensor(top[:k], device=ac.device)
            acc(f"top{k}", run(corr, (ac - ar)[:, S] @ sae.W_dec[S]).sum().item())
            ld_abl = run(clean, -ac[:, S] @ sae.W_dec[S])
            kept[k] += (ld_abl / ld_c).tolist()
            still_ok[k] += (ld_abl > 0).tolist()
            if k in (5, 20) and len(pool) >= k:
                for _ in range(n_random):
                    R = torch.as_tensor(rng.choice(pool, k, replace=False), device=ac.device)
                    acc(f"random{k}", run(corr, (ac - ar)[:, R] @ sae.W_dec[R]).sum().item() / n_random)
    norm = lambda v: (v - tot["corrupt"]) / (tot["clean"] - tot["corrupt"])
    res = {key: float(norm(v)) for key, v in tot.items() if key not in ("clean", "corrupt")}
    res["ablate_top"] = {k: dict(ld_kept=float(np.median(kept[k])), still_correct=float(np.mean(still_ok[k])))
                         for k in KS}
    res["n_pairs"] = int(sum(min(len(c), per_cell) for c in cells(pairs).values()))
    return res


def promoted_tokens(m, sae, f, n=8):
    d = sae.W_dec[f] * (1.0 + m.final_norm.weight.float())
    logits = m.W_U.float() @ d
    return [m.tok.decode([int(i)]) for i in torch.topk(logits, n).indices]


@torch.no_grad()
def run_sae(m, items, out, work, n_top=10):
    task = "honorific" if "addressee" in items[0] else "speaker_gender"
    beh = load_beh(out).set_index("id")
    cue = [it for it in items if it["group"] == "cue"]
    pairs = patch_pairs(beh.reset_index(), items, "cue")
    held_pairs = [it for it in pairs if it["split"] != "train"]
    rng = np.random.default_rng(0)
    summary, feat_rows = dict(task=task, sites={}), []
    saes = {}
    for site, layer in SITES[task]:
        sae = saes[layer] = SAE(layer, m.device)
        A, meta, fve = collect(m, cue, layer, site, sae)
        y = np.array([it["speaker_gender"] == "F" for it in meta], int)
        ok = np.array([bool(beh.loc[it["id"], "correct"]) for it in meta])
        amb = np.array([it.get("speaker") in AMBIGUOUS for it in meta]) if task == "speaker_gender" else np.zeros(len(meta), bool)
        amb_pairs = {it["pair"] for it in meta if it.get("speaker") in AMBIGUOUS} if task == "speaker_gender" else set()
        train = np.array([it["split"] == "train" and it["pair"] not in amb_pairs for it in meta])
        sel, held = train & ok, ~train & ok & ~amb
        diff = rank_features(A, y, torch.as_tensor(sel))
        order = np.argsort(-np.abs(diff))
        top = order[:max(KS)]
        l0 = float((A > 0).sum(1).float().mean())
        s = dict(layer=layer, l0=l0, frac_variance_explained=fve, fidelity=fidelity(m, cue, layer, site, sae))
        for rank, f in enumerate(order[:2 * n_top]):
            a_held = A[held, f].numpy()
            auc = roc_auc_score(y[held], a_held) if diff[f] > 0 else roc_auc_score(1 - y[held], a_held)
            feat_rows.append(dict(task=task, site=site, layer=layer, rank=rank, feature=int(f),
                                  diff=float(diff[f]), favours="F" if diff[f] > 0 else "M",
                                  mean_F=float(A[held & (y == 1), f].mean()), mean_M=float(A[held & (y == 0), f].mean()),
                                  fire_F=float((A[held & (y == 1), f] > 0).float().mean()),
                                  fire_M=float((A[held & (y == 0), f] > 0).float().mean()),
                                  auc_held=float(auc), tokens=" | ".join(promoted_tokens(m, sae, int(f)))))
        s["patching"] = feature_patching(m, held_pairs, layer, site, sae, top, rng)
        s["top_features"] = [int(f) for f in top[:20]]
        if task == "speaker_gender":  # (4) errors: top female-favouring feature on wrong vs right items
            fem = [f for f in order if diff[f] > 0][:5]
            groups = dict(right_F_non_ambiguous=ok & (y == 1) & ~amb, right_F_ambiguous=ok & (y == 1) & amb,
                          wrong_F=~ok & (y == 1), wrong_F_ambiguous=~ok & (y == 1) & amb, right_M=ok & (y == 0))
            s["errors"] = {g: dict(n=int(mask.sum()),
                                   top1_mean=float(A[mask, fem[0]].mean()) if mask.any() else None,
                                   top1_fire=float((A[mask, fem[0]] > 0).float().mean()) if mask.any() else None,
                                   top5_sum=float(A[mask][:, fem].sum(1).mean()) if mask.any() else None)
                           for g, mask in groups.items()}
            s["errors"]["female_features"] = [int(f) for f in fem]
        summary["sites"][site] = s
        print(site, json.dumps({k: v for k, v in s.items() if k != "top_features"}, ensure_ascii=False), flush=True)
    if task == "honorific":  # (3) overlap with the speaker-gender features at the answer position
        g = out.parent / "gemma-3-4b-pt" / "sae_summary.json"
        if g.exists():
            gs = json.loads(g.read_text())["sites"]["answer"]["top_features"]
            hs = summary["sites"]["answer"]["top_features"]
            W = saes[26].W_dec
            Wg, Wh = W[gs[:5]], W[hs[:5]]
            cos = torch.nn.functional.normalize(Wg, dim=1) @ torch.nn.functional.normalize(Wh, dim=1).T
            summary["cross_task_answer"] = dict(
                gender_top=gs[:20], honorific_top=hs[:20], overlap_top20=len(set(gs[:20]) & set(hs[:20])),
                overlap_top5=len(set(gs[:5]) & set(hs[:5])), max_abs_cos_top5=float(cos.abs().max()),
                cos_top5=cos.cpu().numpy().round(3).tolist())
    pd.DataFrame(feat_rows).to_csv(out / "sae_features.csv", index=False)
    save_json(summary, out / "sae_summary.json")
