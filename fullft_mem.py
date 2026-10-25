#!/usr/bin/env python3
"""(FULLFT) Memory engineering for FULL fine-tuning of Gemma-4-31B-it (+ the residual VBLL head) on p8 (8 x H100 80 GB) at 2k-32k context.

Why the SCREEN4 memsmoke (screen4_memsmoke.py) OOMed at 8k / 32k. Gemma-4-31B's 10 full-attention layers have head_dim 512 and GQA
(32 query / 4 KV heads). transformers' sdpa path passes enable_gqa=True with no mask there, which neither SDPA's flash kernel (head_dim
<= 256) nor its memory-efficient kernel (no GQA) accepts, so PyTorch falls back to the MATH kernel, which materialises fp32 scores
[B, 32, L, L]: exactly the 128.00 GiB allocation the 32k log shows (32 x 32768^2 x 4 B), and 8 GiB per full layer at 8k. The sliding
layers also got a materialised [B, 1, L, L] window mask (1 GiB bool at 32k).

What this module provides (each piece is optional; tests in test_fullft.py check each against the plain path on a tiny Gemma-4):
  attention "jev_mem"   registered in transformers' AttentionInterface (and a mask function that builds NOTHING). Never materialises
                        [L, L]: full layers with one document per row -> one causal SDPA call; sliding-window layers and packed rows
                        (documents found from position_ids exactly as transformers' find_packed_sequence_indices) -> query chunks of
                        `chunk` rows, each against only the keys it can see (window / its documents), with a [chunk, keys] mask. K/V
                        are expanded to the query heads (repeat_kv) so the memory-efficient kernel applies; on CUDA SDPA is
                        restricted to the efficient / flash kernels (the MATH fallback raises instead of silently allocating L^2).
  offload checkpointing per-decoder-layer activation checkpointing whose saved LAYER INPUT goes to pinned host memory on a side stream
                        (non-reentrant checkpoint saves its inputs through _NoopSaveInputs BEFORE entering its own hook, so an outer
                        saved_tensors_hooks sees exactly those); restored on the backward's recompute. Rows shorter than
                        `min_tokens` are not offloaded. At 32k this moves 60 x 352 MB = 19.7 GiB off each GPU.
  MLP sequence chunking  each layer's MLP runs in `chunk` token slices under nested checkpoints (finer-grained AC): the backward
                        holds one slice's 4 x [chunk, 21504] intermediates instead of the whole row's (5.3 GiB at 32k). Costs one
                        extra MLP forward per long row; off below `min_tokens`.
  packing               first-fit-decreasing rows of documents with per-document positions (block-diagonal causal attention).
  LowMemAdamW           AdamW whose m / v are stored in bf16 with STOCHASTIC rounding (2 + 2 B / param instead of 4 + 4: -14.3 GiB
                        per GPU at 31B / 8). FIDELITY CAVEAT: not bit-identical to fp32 AdamW; unbiased rounding of the moments, the
                        update itself in fp32 (tests bound the trajectory difference on a tiny model only). 8-bit states need torchao /
                        bitsandbytes, which the node image does not have.
"""
from __future__ import annotations

import math
import os
import types

import torch

VERSION = "fullft_mem v5 (2026-10-14; attention policies C / B; masked calls get materialised K / V, never a stride-0 view)"
# v5 (2026-10-14): l32replay_v2 (12B, FP32, FP64 truth per call) showed torch's memory-efficient SDPA kernel WRONG when it gets an
# explicit mask AND broadcast (stride-0) K / V -- repeat_kv's view when there is ONE KV head (12B global layers): the last query row of
# calls whose query length is 1 mod 32 off by 17-35 (HF's own masked path, and our PACKED path); the same call with contiguous K / V,
# the math kernel, and the no-mask is_causal call (our unpacked path) are right. torch 2.13 and 2.9.1 alike. v5: every call that
# passes a mask gets materialised K / V (once per layer call, before the chunk loop; and a last guard in _sdpa). The no-mask single
# call keeps the view (proven right; no copy). JEV_KV_MATERIALIZE=0 restores v4 behaviour (diagnostics / positive controls only).
# v3 (bc4613da, 2026-10-14) padded every fused call whose query length is 1 mod 32 with a zero query row. Its premise (a fused-kernel
# 1-mod-32 tail bug) was refuted by sdv1_kernelprobe, and the pad contaminated every diagnostic run on it (fullft17-21). v3 is WITHDRAWN;
# v4 is v2's attention, byte-for-byte, with only this header changed. repeat_kv still returns a broadcast (stride-0) view when there is
# one KV head (12B global layers): that is the live layout the l32replay diagnostic tests, so it is deliberately unchanged here.
ATTN_NAME = "jev_mem"
REF_NAME = "jev_dense_ref"                           # the explicit dense-mask REFERENCE (tests / tiny models only; refuses long rows)
# ATTENTION POLICY (stock_decision_v1 Stage 1). The model config carries it (config.jev_attention_policy), so training, feature
# extraction, export (config.json) and scoring read the same value. A config without it is the legacy causal policy C (fullft <= v12).
#   C  causal:                 global  same_document AND key <= query
#                              local   same_document AND key <= query AND query - key < W
#   B  global-bidirectional:   global  same_document                      (every key of the query's own document)
#                              local   identical to C (causal, window W)
# The layer TYPE decides global vs local (sliding_window is None <=> a full-attention layer; checked against module.is_sliding), never
# the row length: a local layer whose window covers the row (W >= L) stays causal in both policies.
POLICIES = ("C", "B")
POLICY_ATTR = "jev_attention_policy"
REF_MAX_TOKENS = 4096
# kernel / mask bookkeeping (the canary reports it): calls per (layer kind, policy, path), the largest mask built, CUDA eligibility
KERNEL_LOG = {"calls": {}, "max_mask_elems": 0, "eligible": {}, "kv_materialized": 0}
KV_MATERIALIZE = os.environ.get("JEV_KV_MATERIALIZE", "1") != "0"


def has_stride0(*ts):
    """True when any tensor is a broadcast view (a stride-0 dim of size > 1)"""
    return any(t is not None and any(st == 0 and n > 1 for st, n in zip(t.stride(), t.shape)) for t in ts)


def materialize_kv(k, v):
    """(v5) contiguous copies of broadcast K / V (for masked SDPA calls); counted in KERNEL_LOG"""
    if KV_MATERIALIZE and has_stride0(k, v):
        KERNEL_LOG["kv_materialized"] = KERNEL_LOG.get("kv_materialized", 0) + 1
        return k.contiguous(), v.contiguous()
    return k, v
DEFAULT_CHUNK = int(os.environ.get("JEV_ATTN_CHUNK", "2048"))
# "kernels" (default): SDPA restricted to the memory-efficient / flash kernels. "chunked_math": the FALLBACK if the memory-efficient
# kernel refuses head_dim 512 on some torch build: every query chunk (the full layers too) runs the math kernel inside its own
# non-reentrant checkpoint, so only one chunk's [H, chunk, keys] fp32 scores live at a time (forward and backward); slower.
ATTN_IMPL = os.environ.get("JEV_ATTN_IMPL", "kernels")


# ----------------------------------------------------------------------------------------------------------------------- attention
_DOC_CACHE = {}


def doc_ids(position_ids, B, L, device):
    """[B, L] document index per token (transformers' find_packed_sequence_indices rule: a new document wherever the position does not
    advance by exactly 1), or None when every row is ONE document starting at 0 (the plain causal case). Cached per position_ids tensor
    (every layer of a forward passes the same one), so the host sync happens once per forward, not once per layer."""
    if position_ids is None:
        return None
    key = (position_ids.data_ptr(), tuple(position_ids.shape), position_ids._version, B, L, str(device))
    hit = _DOC_CACHE.get("k")
    if hit is not None and hit[0] == key and hit[1] is position_ids:
        return hit[2]
    out = _doc_ids(position_ids, B, L, device)
    _DOC_CACHE["k"] = (key, position_ids, out)
    return out


def _doc_ids(position_ids, B, L, device):
    pos = position_ids
    if pos.dim() == 1:
        pos = pos[None]
    if pos.shape[-1] != L:
        raise RuntimeError(f"{ATTN_NAME}: position_ids length {pos.shape[-1]} != query length {L}")
    pos = pos.to(device).expand(B, L)
    first = pos[:, :1] - 1
    d = (torch.diff(pos, prepend=first, dim=-1) != 1).cumsum(-1)
    if bool((d == d[:, :1]).all()):
        return None
    return d


def policy_of(config):
    """the attention policy a model config carries ("C" when absent: the legacy causal path)"""
    pol = getattr(config, POLICY_ATTR, None) if config is not None else None
    if pol is None:
        return "C"
    if pol not in POLICIES:
        raise RuntimeError(f"{ATTN_NAME}: unknown attention policy {pol!r} (expected one of {POLICIES})")
    return pol


def set_policy(config, policy):
    """write the policy onto a (text) config; every attention module of a model built from it reads it"""
    if policy not in POLICIES:
        raise ValueError(f"attention policy {policy!r} not in {POLICIES}")
    setattr(config, POLICY_ATTR, policy)
    return policy


def _layer_kind(module, sliding_window):
    """'global' (full attention) or 'local' (sliding window), from the layer TYPE; refuses an inconsistent module"""
    is_global = sliding_window is None
    ms = getattr(module, "is_sliding", None)
    if ms is not None and bool(ms) == is_global:
        raise RuntimeError(f"{ATTN_NAME}: layer type (is_sliding={ms}) disagrees with sliding_window={sliding_window}")
    return "global" if is_global else "local"


def _log(kind, policy, path, mask):
    k = (kind, policy, path)
    KERNEL_LOG["calls"][k] = KERNEL_LOG["calls"].get(k, 0) + 1
    if mask is not None:
        KERNEL_LOG["max_mask_elems"] = max(KERNEL_LOG["max_mask_elems"], int(mask.numel()))


def _doc_ends(doc, L):
    """[B, L] -> for each token, 1 + the last index of its document"""
    idx = torch.arange(L, device=doc.device).expand_as(doc)
    last = torch.diff(doc, append=doc[:, -1:] + 1, dim=-1) != 0
    val = torch.where(last, idx + 1, torch.full_like(idx, L))
    return val.flip(-1).cummin(-1).values.flip(-1)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _eligible(q, k, v, mask, causal):
    """CUDA: which fused SDPA kernels accept this call (recorded once per signature; the canary reports it)"""
    sig = (tuple(q.shape[1:]), tuple(k.shape[2:]), str(q.dtype), mask is not None, bool(causal))
    if sig in KERNEL_LOG["eligible"]:
        return
    out = {}
    try:
        from torch.backends.cuda import SDPAParams, can_use_efficient_attention, can_use_flash_attention
        try:
            prm = SDPAParams(q, k, v, mask, 0.0, bool(causal), False)
        except TypeError:
            prm = SDPAParams(q, k, v, mask, 0.0, bool(causal))
        out = {"flash": bool(can_use_flash_attention(prm, False)), "efficient": bool(can_use_efficient_attention(prm, False))}
    except Exception as e:      # noqa: BLE001
        out = {"probe_error": f"{type(e).__name__}: {str(e)[:120]}"}
    KERNEL_LOG["eligible"][sig] = out


def allowed_backends():
    """the SDPA kernels jev_mem may use on CUDA ($JEV_SDPA_BACKENDS, default efficient,flash). The math kernel (it materialises
    [H, L, L] scores) is refused unless JEV_ALLOW_SDPA_MATH=1: no SILENT math fallback; an ineligible call raises instead."""
    allowed = [x for x in os.environ.get("JEV_SDPA_BACKENDS", "efficient,flash").split(",") if x]
    if "math" in allowed and os.environ.get("JEV_ALLOW_SDPA_MATH") != "1":
        raise RuntimeError(f"{ATTN_NAME}: the SDPA math kernel is not an allowed fallback (it materialises [H, L, L]); "
                           "JEV_ALLOW_SDPA_MATH=1 is the explicit, recorded override")
    return allowed


def _sdpa(q, k, v, mask, causal, scale):
    if mask is not None:                             # (v5) last guard: no masked call ever sees a stride-0 K / V
        k, v = materialize_kv(k, v)
    if ATTN_IMPL == "chunked_math":
        from torch.utils.checkpoint import checkpoint
        f = torch.nn.functional.scaled_dot_product_attention
        if q.device.type == "cuda":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def f(q_, k_, v_, m_, _f=f):                 # noqa: E306
                with sdpa_kernel([SDPBackend.MATH]):
                    return _f(q_, k_, v_, attn_mask=m_, is_causal=causal, scale=scale)
            return checkpoint(f, q, k, v, mask, use_reentrant=False) if torch.is_grad_enabled() else f(q, k, v, mask)
        return checkpoint(lambda q_, k_, v_, m_: f(q_, k_, v_, attn_mask=m_, is_causal=causal, scale=scale), q, k, v, mask,
                          use_reentrant=False) if torch.is_grad_enabled() else f(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
    if q.device.type == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        allowed = allowed_backends()
        _eligible(q, k, v, mask, causal)
        bk = {"efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
              "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}
        with sdpa_kernel([bk[a] for a in allowed if a]):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)


def jev_mem_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None, softcap=None,
                      **kwargs):
    """transformers attention interface: query [B, Hq, L, D], key / value [B, Hkv, L, D] -> ([B, L, Hq, D], None). Documents from
    position_ids; the model config's attention POLICY (module docstring): local layers causal + window (key j visible from query i iff
    j <= i and j > i - W, transformers' sliding_window_overlay); global layers causal (C) or every key of the document (B). No KV cache
    (training and full-prompt scoring only); shared-KV layers get the full-length K / V and the same document mask."""
    if attention_mask is not None:
        raise RuntimeError(f"{ATTN_NAME}: got a materialised attention mask; this implementation derives masks from position_ids "
                           "(pass attention_mask=None / no padding)")
    if dropout:
        raise RuntimeError(f"{ATTN_NAME}: attention dropout is not supported")
    if softcap is not None:
        raise RuntimeError(f"{ATTN_NAME}: attention logit softcapping is not supported")
    B, H, L, D = query.shape
    if key.shape[2] != L:
        raise RuntimeError(f"{ATTN_NAME}: key length {key.shape[2]} != query length {L} (no KV cache support)")
    policy = policy_of(getattr(module, "config", None))
    kind = _layer_kind(module, sliding_window)
    bidir = policy == "B" and kind == "global"
    scale = scaling if scaling is not None else D ** -0.5
    n_rep = H // key.shape[1]
    k, v = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    W = int(sliding_window) if kind == "local" else None
    if W is not None and W >= L:
        W = None                                     # a LOCAL layer whose window covers the row: plain causal (never bidirectional)
    doc = doc_ids(kwargs.get("position_ids"), B, L, query.device)
    if doc is None and W is None and ATTN_IMPL != "chunked_math":
        _log(kind, policy, "single_bidir" if bidir else "single_causal", None)
        out = _sdpa(query, k, v, None, (L > 1) and not bidir, scale)
        return out.transpose(1, 2).contiguous(), None
    k, v = materialize_kv(k, v)                      # (v5) every call below passes a mask: K / V once, before the chunk slices
    C = max(16, int(kwargs.get("jev_chunk") or DEFAULT_CHUNK))
    if doc is not None:                              # first index of each token's document (min over rows), on the host: one sync
        idx = torch.arange(L, device=query.device).expand(B, L)
        start = torch.where(torch.diff(doc, prepend=doc[:, :1] - 1, dim=-1) != 0, idx, torch.zeros_like(idx)).cummax(-1).values
        start = start.min(0).values.tolist()
        if bidir:                                    # B: 1 + the last index of each token's document (max over rows)
            end = _doc_ends(doc, L).max(0).values.tolist()
    outs = []
    for s in range(0, L, C):
        e = min(L, s + C)
        if bidir:                                    # keys = the documents the chunk's queries belong to (past AND future)
            k0, k1 = (int(start[s]), int(end[e - 1])) if doc is not None else (0, L)
            m = (doc[:, s:e, None] == doc[:, None, k0:k1])[:, None] if doc is not None else None
        else:
            k0, k1 = (0 if W is None else max(0, s - W + 1)), e
            if doc is not None:
                k0 = max(k0, int(start[s]))
            qi = torch.arange(s, e, device=query.device)[:, None]
            ki = torch.arange(k0, k1, device=query.device)[None, :]
            m = ki <= qi
            if W is not None:
                m = m & (ki > qi - W)
            if doc is not None:
                m = m[None] & (doc[:, s:e, None] == doc[:, None, k0:k1])
                m = m[:, None]
            else:
                m = m[None, None]
        _log(kind, policy, "chunked_bidir" if bidir else "chunked_causal", m)
        outs.append(_sdpa(query[:, :, s:e], k[:, :, k0:k1], v[:, :, k0:k1], m, False, scale))
    out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    return out.transpose(1, 2).contiguous(), None


def dense_mask(doc, B, L, kind, policy, W, device):
    """REFERENCE: the explicit [B, L, L] boolean mask of the contract (ROADMAP Stage 1), built without any shortcut (the window applies
    whatever L is). doc [B, L] or None (one document per row)."""
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    same = (doc[:, :, None] == doc[:, None, :]) if doc is not None else torch.ones(B, L, L, dtype=torch.bool, device=device)
    if policy == "B" and kind == "global":
        return same
    m = same & (j <= i)[None]
    if kind == "local":
        m = m & ((i - j) < W)[None]
    return m


def dense_reference_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, sliding_window=None,
                              softcap=None, **kwargs):
    """REFERENCE attention (tests): explicit dense mask + explicit softmax, the query dtype throughout. Refuses rows longer than
    REF_MAX_TOKENS (it materialises [B, H, L, L])."""
    if attention_mask is not None or dropout or softcap is not None:
        raise RuntimeError(f"{REF_NAME}: masks / dropout / softcap are not part of the reference contract")
    B, H, L, D = query.shape
    if L > REF_MAX_TOKENS:
        raise RuntimeError(f"{REF_NAME}: reference only (L={L} > {REF_MAX_TOKENS})")
    policy = policy_of(getattr(module, "config", None))
    kind = _layer_kind(module, sliding_window)
    pos = kwargs.get("position_ids")
    doc = _doc_ids(pos, B, L, query.device) if pos is not None else None
    m = dense_mask(doc, B, L, kind, policy, int(sliding_window) if kind == "local" else None, query.device)
    n_rep = H // key.shape[1]
    k, v = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    sc = (query @ k.transpose(-1, -2)) * (scaling if scaling is not None else D ** -0.5)
    sc = sc.masked_fill(~m[:, None], float("-inf"))
    out = torch.softmax(sc, dim=-1) @ v
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, attention_mask=None, **kwargs):
    """mask interface for jev_mem: build nothing; refuse a padding mask (padded batches are not supported)."""
    if attention_mask is not None and not bool(torch.as_tensor(attention_mask).bool().all()):
        raise RuntimeError(f"{ATTN_NAME}: padded batches are not supported (pack documents with position_ids instead)")
    return None


def register():
    """make attn_implementation='jev_mem' available to transformers models (idempotent)."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register(ATTN_NAME, jev_mem_attention)
    AttentionMaskInterface.register(ATTN_NAME, _no_mask)
    AttentionInterface.register(REF_NAME, dense_reference_attention)
    AttentionMaskInterface.register(REF_NAME, _no_mask)
    return ATTN_NAME


def require_policy(config, policy):
    """loaders (training, feature extraction, export reload, scoring): the config must run through jev_mem (or the reference) with
    exactly this policy; a B checkpoint run by stock sdpa / eager attention would silently be causal"""
    impl = getattr(config, "_attn_implementation", None)
    if policy_of(config) != policy:
        raise RuntimeError(f"attention policy {policy_of(config)} != required {policy}")
    if policy != "C" and impl not in (ATTN_NAME, REF_NAME):
        raise RuntimeError(f"attention policy {policy} requires attn_implementation {ATTN_NAME} (got {impl})")
    return policy


# --------------------------------------------------------------------------------------------------- offloaded layer checkpointing
class _HostCopy:
    __slots__ = ("buf", "event", "device")

    def __init__(self, buf, event, device):
        self.buf, self.event, self.device = buf, event, device

    def restore(self):
        if self.event is not None:
            self.event.synchronize()
            return self.buf.to(self.device, non_blocking=True)
        return self.buf.clone()


_STREAM = {}
STATS = {"offloaded": 0, "offloaded_bytes": 0, "restored": 0}


def _to_host(t):
    if t.device.type == "cuda":
        st = _STREAM.get(t.device)
        if st is None:
            st = _STREAM[t.device] = torch.cuda.Stream(device=t.device)
        buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        st.wait_stream(torch.cuda.current_stream(t.device))
        with torch.cuda.stream(st):
            buf.copy_(t, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(st)
        t.record_stream(st)
        hc = _HostCopy(buf, ev, t.device)
    else:                                           # CPU (tests): a separate copy, so a test sees the round trip
        hc = _HostCopy(t.detach().clone(), None, t.device)
    STATS["offloaded"] += 1
    STATS["offloaded_bytes"] += t.numel() * t.element_size()
    return hc


def make_checkpoint_fn(offload=False, min_tokens=0, force_cpu=False):
    """a `_gradient_checkpointing_func` for transformers' GradientCheckpointingLayer: non-reentrant checkpoint of the whole layer; with
    `offload`, the layer's hidden-state input (args[0]) is moved to pinned host memory while the row has >= min_tokens tokens."""
    from torch.utils.checkpoint import checkpoint

    def ckpt(fn, *args):
        x = args[0] if args else None
        if not offload or not isinstance(x, torch.Tensor) or x.dim() < 2 or x.shape[-2] < min_tokens \
                or (x.device.type != "cuda" and not force_cpu):
            return checkpoint(fn, *args, use_reentrant=False)
        target = (x.data_ptr(), tuple(x.shape), x.dtype)

        def pack(t):
            if t.data_ptr() == target[0] and tuple(t.shape) == target[1] and t.dtype == target[2]:
                return _to_host(t)
            return t

        def unpack(p):
            if isinstance(p, _HostCopy):
                STATS["restored"] += 1
                return p.restore()
            return p

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return checkpoint(fn, *args, use_reentrant=False)
    return ckpt


def enable_checkpointing(text, offload=False, min_tokens=0, force_cpu=False):
    """activation checkpointing on every decoder layer of a Gemma-4 text stack (transformers' own switch, then our function)."""
    text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fn = make_checkpoint_fn(offload, min_tokens, force_cpu)
    n = 0
    for layer in text.layers:
        layer._gradient_checkpointing_func = fn
        n += 1
    return n


# ---------------------------------------------------------------------------------------------------------- MLP sequence chunking
def chunk_mlps(text, chunk=4096, min_tokens=8192):
    """Each decoder layer's MLP (token-wise) runs in `chunk`-token slices under nested non-reentrant checkpoints when grad is on and the
    row has >= min_tokens tokens. Same parameters / names (the bound forward is replaced in place). -> number of MLPs patched."""
    from torch.utils.checkpoint import checkpoint
    n = 0
    for layer in text.layers:
        mlp = layer.mlp
        if getattr(mlp, "_jev_chunked", False):
            continue
        orig = mlp.forward

        def fwd(self, x, _orig=orig):
            T = x.shape[-2]
            if not torch.is_grad_enabled() or T < min_tokens or T <= chunk:
                return _orig(x)
            outs = [checkpoint(_orig, x[..., s:min(T, s + chunk), :], use_reentrant=False) for s in range(0, T, chunk)]
            return torch.cat(outs, dim=-2)
        mlp.forward = types.MethodType(fwd, mlp)
        mlp._jev_chunked = True
        n += 1
    return n


# ----------------------------------------------------------------------------------------------------------------------- packing
def pack_rows(lengths, budget):
    """first-fit-decreasing: indices -> rows whose length sums are <= budget (an item longer than budget gets a row of its own).
    Deterministic (ties by index). -> list of lists of indices, each row in ascending index order."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    rows, room = [], []
    for i in order:
        n = lengths[i]
        for r in range(len(rows)):
            if room[r] >= n:
                rows[r].append(i)
                room[r] -= n
                break
        else:
            rows.append([i])
            room.append(budget - n)
    return [sorted(r) for r in rows]


def packed_batch(token_lists, pad_to=0, pad_id=0):
    """one packed row: input_ids [1, T], position_ids [1, T] restarting at 0 per document, last index of each document [n]. `pad_to`:
    pad with ONE extra document (positions from 0; never read) to a multiple of pad_to tokens."""
    flat, pos, last, off = [], [], [], 0
    for ids in token_lists:
        flat.extend(ids)
        pos.extend(range(len(ids)))
        off += len(ids)
        last.append(off - 1)
    extra = (-len(flat)) % pad_to if pad_to else 0
    if extra:
        flat.extend([pad_id] * extra)
        pos.extend(range(extra))
    return (torch.tensor([flat], dtype=torch.long), torch.tensor([pos], dtype=torch.long), torch.tensor(last, dtype=torch.long))


# ------------------------------------------------------------------------------------------------ AdamW with bf16 moments (SR)
def _bf16_sr(x, gen=None):
    """fp32 -> bf16 with stochastic rounding (unbiased): add uniform 16-bit noise to the dropped mantissa bits, then truncate."""
    xi = x.contiguous().view(torch.int32)
    r = torch.randint(0, 1 << 16, xi.shape, device=x.device, dtype=torch.int32, generator=gen)
    y = ((xi + r) & -65536).view(torch.float32)
    y = torch.where(torch.isfinite(x), y, x)
    return y.to(torch.bfloat16)


class LowMemAdamW(torch.optim.Optimizer):
    """AdamW (decoupled weight decay, bias correction as torch.optim.AdamW) with exp_avg / exp_avg_sq stored in bf16 via stochastic
    rounding; params (fp32 masters) and the update are fp32. Works on plain tensors and DTensors (operates on the local shard).
    FIDELITY CAVEAT (module docstring). `chunk`: elements processed at a time (bounds the fp32 temporaries)."""

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, seed=0, chunk=1 << 24):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.seed, self.chunk = int(seed), int(chunk)
        self._gen = {}

    def _g(self, dev):
        g = self._gen.get(dev)
        if g is None:
            g = self._gen[dev] = torch.Generator(device=dev).manual_seed(self.seed)
        return g

    @staticmethod
    def _local(t):
        return t.to_local() if hasattr(t, "to_local") else t

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                pl, gl = self._local(p), self._local(p.grad)
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros(pl.shape, dtype=torch.bfloat16, device=pl.device)
                    st["exp_avg_sq"] = torch.zeros(pl.shape, dtype=torch.bfloat16, device=pl.device)
                st["step"] += 1
                t = st["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                pf, gf = pl.view(-1), gl.reshape(-1)
                mf, vf = st["exp_avg"].view(-1), st["exp_avg_sq"].view(-1)
                gen = self._g(pl.device)
                for s in range(0, pf.numel(), self.chunk):
                    e = min(pf.numel(), s + self.chunk)
                    g = gf[s:e].float()
                    m = mf[s:e].float().mul_(b1).add_(g, alpha=1 - b1)
                    v = vf[s:e].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                    mf[s:e] = _bf16_sr(m, gen)
                    vf[s:e] = _bf16_sr(v, gen)
                    x = pf[s:e]
                    xf = x.float()
                    if group["weight_decay"]:
                        xf.mul_(1 - group["lr"] * group["weight_decay"])
                    denom = (v / bc2).sqrt_().add_(group["eps"])
                    xf.addcdiv_(m, denom, value=-group["lr"] / bc1)
                    x.copy_(xf)
        return loss


class StateEvictor:
    """moves an optimizer's moment tensors (exp_avg, exp_avg_sq; local shards of DTensors) to host buffers and frees their device
    storage (the tensor objects stay: storage resized to 0, as FSDP2 does for unsharded parameters); restore() re-allocates and
    copies back, so the optimizer runs the same kernel on the same values: bit-identical to never evicting. Host buffers are
    allocated once (pinned on CUDA) and reused."""
    KEYS = ("exp_avg", "exp_avg_sq")

    def __init__(self, opt, pinned=False):
        self.opt, self.pinned, self.host, self.out = opt, pinned, {}, []
        self.stats = {"evictions": 0, "evicted_bytes": 0, "evict_s": 0.0, "restore_s": 0.0}

    def _tensors(self):
        for p, st in self.opt.state.items():
            for k in self.KEYS:
                t = st.get(k)
                if isinstance(t, torch.Tensor):
                    yield (id(p), k), (t.to_local() if hasattr(t, "to_local") else t)

    def evict(self):
        if self.out:
            return False
        import time
        t0, todo = time.time(), []
        for key, t in self._tensors():
            if t.untyped_storage().size() == 0:
                continue
            h = self.host.get(key)
            if h is None:
                h = self.host[key] = torch.empty(t.shape, dtype=t.dtype, pin_memory=self.pinned and t.is_cuda)
            h.copy_(t, non_blocking=t.is_cuda)
            todo.append((t, h))
        if not todo:
            return False
        if todo[0][0].is_cuda:
            torch.cuda.current_stream(todo[0][0].device).synchronize()
        for t, h in todo:
            self.stats["evicted_bytes"] += t.untyped_storage().size()
            t.untyped_storage().resize_(0)
        self.out = todo
        self.stats["evictions"] += 1
        self.stats["evict_s"] += time.time() - t0
        return True

    def restore(self):
        if not self.out:
            return False
        import time
        t0 = time.time()
        for t, h in self.out:
            t.untyped_storage().resize_(t.numel() * t.element_size())
            t.copy_(h, non_blocking=t.is_cuda)
        if self.out[0][0].is_cuda:
            torch.cuda.current_stream(self.out[0][0].device).synchronize()
        self.out = []
        self.stats["restore_s"] += time.time() - t0
        return True


def memory_plan_summary(plan):
    return {k: plan[k] for k in sorted(plan)}
