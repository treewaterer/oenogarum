#!/usr/bin/env python3
"""(DIKIT, 2026-10-15) The stock_decision_v1 31B full fine-tune exports (attention policy B) + their Latent01-post head (l01post_di) as a
Decision Index engine, for the kit's runner (commit 9eb2dbe2):

    python -m decision_index run --engine dikit_engine:DikitEngine --option model=<export dir> --rows ... --out ...

= garas_vbll_engine.GarasVBLLEngine (f2a988c0, UNCHANGED: it reads the export's jev_attention_policy, refuses anything but attn='jev_mem'
for policy B, loads the model with fullft_mem's jev_mem attention -- the trainer's own attention code -- and packs each request's
prompts in one row with per-prompt positions, so the global layers see exactly their own prompt, bidirectionally) with two pins:

  * the prompt renderer is garas_engine.py 101a6f9f (noul_defs=True = renderer R2), the renderer every stock_decision_v1 manifest was
    built with (manifest headers: render "r2", renderer_sha256 0fab3c66); 4af46b6e (R1) would drop the yes / no definitions;
  * the Latent01-post head (vbll_core.L01PostHead) runs on the CPU in fp64: its code builds per-kind tensors on the CPU (kinds,
    r_scale, dm_noul / var_noul) and mixes them with the hidden state's device, so on CUDA it raises a device mismatch; on the CPU it is
    sdv1_head's own arithmetic (the dev predictions are computed there). The backbone, the native letter logits and everything else
    stay on the GPU. Other head schemas are served as the base engine serves them.

require_schema (default jev-l01post-head-v1) refuses an export whose serving head is anything else, so a shard can never silently serve
an older head. provenance adds the md5s of every engine file and the head's identity.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch                                                   # noqa: E402

import vbll_core                                               # noqa: E402
from garas_vbll_engine import GarasVBLLEngine                  # noqa: E402

VERSION = "dikit_engine v1 (2026-10-15)"
L01_SCHEMA = "jev-l01post-head-v1"


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class CpuHead:
    """a head served on the CPU: inputs moved to the CPU, outputs returned on the CPU (the engine moves them there anyway)"""

    def __init__(self, head):
        self.inner = head.cpu()
        self.heads = getattr(head, "heads", [])
        self.needs_kinds = bool(getattr(head, "needs_kinds", False))
        self.needs_yes_no = bool(getattr(head, "needs_yes_no", False))

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def predict(self, h, ell, ks, *args, **kw):
        kw = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kw.items()}
        return self.inner.predict(h.cpu(), ell.cpu(), ks.cpu(), *args, **kw)


class DikitEngine(GarasVBLLEngine):
    name = "sd31-dikit"

    def __init__(self, model="", device="cuda", require_schema=L01_SCHEMA, head_device="auto", **options):
        super().__init__(model=model, device=device, **options)
        meta = self.head_meta
        if require_schema and meta.get("schema") != require_schema:
            raise SystemExit(f"{model}: the serving head is {meta.get('schema')!r} ({meta.get('mode') or meta.get('head_rule')}), "
                             f"not {require_schema!r}")
        hd = head_device
        if hd == "auto":
            hd = "cpu" if isinstance(self.head, vbll_core.L01PostHead) else str(device)
        if hd == "cpu" and not isinstance(self.head, CpuHead):
            self.head = CpuHead(self.head)
        import garas_core
        import garas_engine
        import fullft_mem
        prov = {"dikit_engine": VERSION, "dikit_engine_md5": _md5(__file__), "garas_engine_md5": _md5(garas_engine.__file__),
                "garas_core_md5": _md5(garas_core.__file__), "fullft_mem_md5": _md5(fullft_mem.__file__), "head_device": hd,
                "head_schema": meta.get("schema"), "head_mode": meta.get("mode") or meta.get("head_rule"),
                "head_variant": meta.get("variant"), "head_constants": meta.get("constants")}
        if isinstance(getattr(self, "provenance", None), dict):
            self.provenance.update(prov)
        else:
            self.provenance = prov
