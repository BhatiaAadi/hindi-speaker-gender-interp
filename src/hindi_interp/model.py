"""Gemma-3 loading, caching and interventions with plain PyTorch hooks.

Gemma-3 decoder layer (per layer l):
    a = post_attention_layernorm(self_attn(input_layernorm(x)))      # "attn_out[l]"
    x = x + a
    m = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))  # "mlp_out[l]"
    x = x + m
so resid_pre[l+1] = resid_pre[l] + attn_out[l] + mlp_out[l], and the final logits are
lm_head(norm(resid_post[L-1])). We read and patch at exactly these points.
"""
import contextlib

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, name="google/gemma-3-4b-pt", device=None, dtype=torch.bfloat16, multi_gpu=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        if multi_gpu:  # e.g. fp32 split over several 11 GB cards
            self.hf = AutoModelForCausalLM.from_pretrained(
                name, dtype=dtype, attn_implementation="eager", device_map="auto").eval()
            self.device = "cuda:0"
        else:
            self.hf = AutoModelForCausalLM.from_pretrained(
                name, dtype=dtype, attn_implementation="eager").to(self.device).eval()
        lm = self.hf.model
        self.text = lm.language_model if hasattr(lm, "language_model") else lm
        self.layers = self.text.layers
        self.final_norm = self.text.norm
        self.W_U = self.hf.get_output_embeddings().weight  # [vocab, d_model]
        cfg = getattr(self.hf.config, "text_config", self.hf.config)
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.hidden_size
        assert getattr(cfg, "final_logit_softcapping", None) is None, "DLA assumes no logit softcap"

    # ------------------------------------------------------------------ tokens
    def tok_id(self, word):
        """Id of ' word' as one token (all our answers are single tokens)."""
        ids = self.tok(" " + word, add_special_tokens=False).input_ids
        assert len(ids) == 1, (word, ids)
        return ids[0]

    def encode(self, prompts):
        """Tokenize same-length prompts (with BOS) into one [batch, seq] tensor."""
        enc = [self.tok(p).input_ids for p in prompts]
        assert len({len(e) for e in enc}) == 1, "batch prompts must have equal token length"
        return torch.tensor(enc, device=self.device)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def logits(self, input_ids, hooks=()):
        """Last-position logits [batch, vocab] (float32), with optional hooks attached."""
        with self.hooked(hooks):
            out = self.hf(input_ids=input_ids, use_cache=False)
        return out.logits[:, -1].float()

    @torch.no_grad()
    def run_with_cache(self, input_ids, want=("resid", "attn", "mlp", "z"), attn_patterns=False):
        """Run once and record per-layer activations (all positions).

        Returns dict with lists over layers:
          resid[l]   input to layer l (l = 0..L, resid[L] = final residual before norm)
          attn[l]    attn_out[l]       mlp[l]  mlp_out[l]
          z[l]       per-head attention outputs before o_proj, [b, s, n_heads, head_dim]
          pattern[l] attention probabilities [b, heads, q, k] (if attn_patterns)
        plus 'logits' at the last position.
        """
        cache = {k: [None] * self.n_layers for k in ("attn", "mlp", "z")}
        cache["resid"] = [None] * (self.n_layers + 1)
        hooks = []
        for l, layer in enumerate(self.layers):
            if "resid" in want:
                hooks.append((layer, "pre", _store(cache["resid"], l, from_input=True)))
            if "attn" in want:
                hooks.append((layer.post_attention_layernorm, "fwd", _store(cache["attn"], l)))
            if "mlp" in want:
                hooks.append((layer.post_feedforward_layernorm, "fwd", _store(cache["mlp"], l)))
            if "z" in want:
                hooks.append((layer.self_attn.o_proj, "pre", _store(cache["z"], l, from_input=True,
                                                                    reshape=(self.n_heads, self.head_dim))))
        if "resid" in want:
            hooks.append((self.final_norm, "pre", _store(cache["resid"], self.n_layers, from_input=True)))
        with self.hooked(hooks):
            out = self.hf(input_ids=input_ids, use_cache=False, output_attentions=attn_patterns)
        cache["logits"] = out.logits[:, -1].float()
        if attn_patterns:
            cache["pattern"] = [a.float() for a in out.attentions]
        return cache

    @contextlib.contextmanager
    def hooked(self, hooks):
        handles = []
        try:
            for module, kind, fn in hooks:
                if kind == "pre":
                    handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
                else:
                    handles.append(module.register_forward_hook(fn))
            yield
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------ readouts
    def final_scale(self, resid_final):
        """Per-example RMSNorm factor of the final norm: rsqrt(mean(x^2)+eps) * (1 + w)."""
        x = resid_final.float()
        eps = getattr(self.final_norm, "eps", getattr(self.final_norm, "variance_epsilon", 1e-6))
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return inv, (1.0 + self.final_norm.weight.float())

    def logit_lens(self, resid, ids_a, ids_b):
        """LD = logit(a) - logit(b) after applying the final norm + unembedding to `resid` [b, d]."""
        normed = self.final_norm(resid.to(self.W_U.device, self.W_U.dtype)).float()
        d = (self.W_U[ids_a].float() - self.W_U[ids_b].float())  # [b, d]
        return (normed * d).sum(-1)

    def dla_direction(self, resid_final, ids_a, ids_b):
        """Vector u such that LD = <component, u> for any component of the final residual
        (final-norm scale frozen at its value on this run)."""
        inv, w = self.final_scale(resid_final)
        d = self.W_U[ids_a].float() - self.W_U[ids_b].float()
        return inv * w * d  # [b, d]


# ---------------------------------------------------------------- hook makers
def _first(args, kwargs):
    return args[0] if args else kwargs.get("hidden_states", next(iter(kwargs.values())))


def _store(slot, idx, from_input=False, reshape=None):
    def fn(module, *a):
        if from_input:
            args, kwargs = a
            x = _first(args, kwargs)
        else:
            out = a[-1]
            x = out[0] if isinstance(out, tuple) else out
        x = x.detach()
        if reshape is not None:
            x = x.view(*x.shape[:2], *reshape)
        slot[idx] = x
    return fn


def patch_input(new_value_fn):
    """Forward pre-hook that replaces the module's first input with new_value_fn(old)."""
    def fn(module, args, kwargs):
        if args:
            return (new_value_fn(args[0]),) + tuple(args[1:]), kwargs
        key = "hidden_states" if "hidden_states" in kwargs else next(iter(kwargs))
        kwargs = dict(kwargs)
        kwargs[key] = new_value_fn(kwargs[key])
        return args, kwargs
    return fn


def patch_output(new_value_fn):
    """Forward hook that replaces the module's output with new_value_fn(old)."""
    def fn(module, args, out):
        if isinstance(out, tuple):
            return (new_value_fn(out[0]),) + tuple(out[1:])
        return new_value_fn(out)
    return fn


def set_positions(src, positions=None):
    """Return fn(old) that copies `src` into `old` at the given positions (all if None)."""
    def fn(old):
        new = old.clone()
        if positions is None:
            new[:] = src.to(old.device, old.dtype)
        else:
            new[:, positions] = src[:, positions].to(old.device, old.dtype)
        return new
    return fn


def log_softmax_pair(logits, ids_a, ids_b):
    lp = F.log_softmax(logits.float(), -1)
    return lp.gather(1, ids_a[:, None])[:, 0], lp.gather(1, ids_b[:, None])[:, 0]
