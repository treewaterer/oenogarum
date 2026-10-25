#!/usr/bin/env python3
"""Garas full fine-tune + ACTIVE residual VBLL head as a Decision Index engine (the kit's `Engine` interface), for the kit's runner
(github.com/apolinario/decision-index):

    python -m decision_index run --engine garas_vbll_engine:GarasVBLLEngine --option model=<export dir or repo> --out runs/x

What the served answer is: the head's POSTERIOR PREDICTIVE E_q softmax(tau ell + phi(h)^T beta) from ONE forward per prompt, not the
backbone's letter softmax. The engine refuses to start without the head files (vbll_head.safetensors + vbll_head.json, sha256-checked)
and the exported config's `jev_requires_head` marker, so a release cannot silently drop the head again (the released Garas adapter
dropped its auxiliary VBLL sidecar).

Requests, prompts, option orders, K > 52 rounds and answer assembly are the released Garas engine's (garas_engine.py / garas_core.py,
imported unchanged). The read-out differs: the final hidden state h at each prompt's last token (the model's inner stack), the native
letter logits in fp32 from h (as in training: fullft_train.py), then the head. Attention: 'jev_mem' (default; memory-efficient, no
[L, L] tensors, prompts packed in one row with per-prompt positions) or 'sdpa' with the released engine's right-padded batches.
raw (second return value) carries, per question key, the native probabilities and the head's epistemic / aleatoric terms, for audit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import garas_core as core                                      # noqa: E402  (released file, unchanged)
import vbll_core                                               # noqa: E402
from garas_engine import GarasEngine, check_state_shown        # noqa: E402  (released file + the 2026-10-20 no-truncation check)
from decision_index.engines.base import Engine, Unsupported    # noqa: E402  (the kit must be on sys.path)


class GarasVBLLEngine(GarasEngine):
    name = "garas-vbll"
    latency = ("CUDA-synchronized in-process request wall time: conversion, one prompt per question (packed or right-padded batches; "
               "K > 52 round 2 after round 1), one forward per prompt, the residual VBLL head's posterior predictive (fixed draws); "
               "excludes model loading.")

    def __init__(self, model="", device="cuda", max_len=32768, tokens_per_batch=32768, attn="jev_mem", samples=0, _backend=None,
                 dtype="bfloat16", ordinal=False, **options):
        Engine.__init__(self, **options)
        self.max_len, self.tokens_per_batch = int(max_len), int(tokens_per_batch)
        # (2026-10-15) the Latent01 head's kinds: the request's question type per prompt (choice / noul). ORDINAL PATH (our own API only,
        # off by default, never on the board): with ordinal=True a choice question carrying "ordinal": true is rendered in its criteria
        # order (no shuffle) and served as a score row (the score-interval integral)
        self.ordinal = str(ordinal).lower() in ("1", "true", "yes")
        self._kind, self._ord_keys = {}, set()
        if _backend is not None:                 # tests: (tok, chat, forward(texts) -> (h [B, H], ell [B, >= k]), sync, head, meta)
            self.tok, self.chat, self._forward, self._sync, self.head, self.head_meta = _backend
            self.samples = int(samples) or int(self.head_meta.get("samples", 64))
            self.provenance = {"kind": "test backend"}
            return
        self._load_vbll(model, device, attn, samples, dtype)

    def _load_vbll(self, model, device, attn, samples, dtype="bfloat16"):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        mdir = model
        if not os.path.isdir(mdir):
            from huggingface_hub import snapshot_download
            mdir = snapshot_download(model)
        cfg_raw = json.load(open(os.path.join(mdir, "config.json")))
        if "jev_requires_head" not in cfg_raw:
            raise SystemExit(f"{mdir}: config.json has no jev_requires_head marker (not a Garas full-FT + VBLL export)")
        tcr = cfg_raw["text_config"] if isinstance(cfg_raw.get("text_config"), dict) else cfg_raw
        policy = tcr.get("jev_attention_policy", "C")       # (stock_decision_v1) the export's attention policy: C causal, B bidirectional
        if policy != "C" and attn != "jev_mem":             # global layers; only jev_mem applies it (stock sdpa would serve it causally)
            raise SystemExit(f"{mdir}: attention policy {policy} needs attn='jev_mem' (got {attn!r}: it would serve the model causally)")
        self.head, self.head_meta = vbll_core.load_head(mdir)    # (2026-10-15) the fixedE / L-loc schema or the residual VBLL
        if len(self.head.heads) != 1:
            raise SystemExit("the export must carry exactly the selected head")
        mcfg = AutoConfig.from_pretrained(mdir)
        _, model_cls = core.model_loader(mcfg, "auto")
        self.chat = core.chat_preset(mcfg.model_type, False)
        tok = AutoTokenizer.from_pretrained(mdir)
        tok.padding_side = "right"
        ids = [tok(c, add_special_tokens=False)["input_ids"] for c in core.LABELS]
        if any(len(x) != 1 for x in ids):
            raise SystemExit("a letter is not a single token")
        label_ids = [x[0] for x in ids]
        if label_ids != list(self.head_meta["label_ids"]):
            raise SystemExit("the head was trained with other letter tokens")
        kw = {}
        if attn == "jev_mem":
            import fullft_mem as fm                              # shipped beside this file
            fm.register()
            kw["attn_implementation"] = fm.ATTN_NAME
        if dtype not in ("bfloat16", "float32"):          # float32: the export / scorer parity test (same path, no quantisation)
            raise SystemExit(f"dtype {dtype!r}: bfloat16 (serving) or float32 (parity tests)")
        m = model_cls.from_pretrained(mdir, dtype=getattr(torch, dtype), device_map=device, **kw).eval()
        if attn == "jev_mem":
            fm.require_policy(m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config, policy)
        self.tok, self.model, self.device = tok, m, device
        self.head = self.head.to(device)
        lab = torch.tensor(label_ids, device=device)
        tcfg = mcfg.get_text_config() if hasattr(mcfg, "get_text_config") else mcfg
        cap = float(getattr(tcfg, "final_logit_softcapping", 0) or 0)
        mode = "pack" if attn == "jev_mem" else "pad"
        readout = self.head_meta.get("letter_readout")     # (2026-10-15) the readout the head was trained on (absent = stock)
        vbll_core.readout_spec(readout)

        @torch.no_grad()
        def forward(texts):
            keep = []
            z = core.exact_letter_logits(m, tok, texts, lab, device, mode, head="fp32", keep_h=keep)
            h = torch.cat(keep, 0)
            E = m.get_output_embeddings().weight[lab]
            ell = vbll_core.letter_logits_fp32(h, E, cap)
            if not torch.equal(ell, z):
                raise RuntimeError("letter readout mismatch")
            if readout and readout != "stock":           # the stock path is checked above against the scorer; then the trained readout
                ell = vbll_core.letter_readout_fp32(h, E, cap, readout)
            return h, ell

        self._forward = forward
        self._sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
        self.samples = int(samples) or int(self.head_meta.get("samples", 64))
        import transformers
        self.provenance = {"kind": "in-process", "model": model, "dtype": dtype, "attn": attn, "readout": "vbll posterior predictive",
                           "head_sha256": self.head_meta["tensor_sha256"], "head": self.head_meta["heads"][0],
                           "samples": self.samples, "transformers": transformers.__version__, "torch": torch.__version__,
                           "attention_policy": policy, "temperature": float(self.head_meta.get("temperature", 1.0)),
                           "letter_readout": readout or "stock",
                           "engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
                           "vbll_core_md5": hashlib.md5(open(vbll_core.__file__, "rb").read()).hexdigest()}

    def run_prompts(self, items):
        import torch
        out = {}
        if not hasattr(self, "last_raw"):
            self.last_raw = {}
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for _, _, t in items]
        i = 0
        while i < len(items):
            j, mx = i, 0
            while j < len(items) and max(mx, lens[j]) * (j - i + 1) <= max(self.tokens_per_batch, lens[i]):
                mx = max(mx, lens[j])
                j += 1
            h, ell = self._forward([t for _, _, t in items[i:j]])
            ks = torch.tensor([len(o) for _, o, _ in items[i:j]], device=ell.device)
            kmax = int(ks.max())
            kw = {}
            if getattr(self.head, "needs_kinds", False):    # (2026-10-15) the Latent01 head: the request's question type per prompt
                kw["kinds"] = torch.tensor([vbll_core.KIND_CODE.get(self._kind.get(qid, "choice"), 0) for qid, _, _ in items[i:j]], dtype=torch.long)
            if getattr(self.head, "needs_yes_no", False):   # (2026-10-15) the L-loc head: the Yes / No positions of the RENDERED prompts
                kw["yes_no"] = vbll_core.yes_no_tensor([t for _, _, t in items[i:j]], [len(o) for _, o, _ in items[i:j]]).to(ell.device)
            p, epi, alea = self.head.predict(h, ell[:, :kmax], ks, self.samples, int(self.head_meta.get("draw_seed", 0)),
                                             self.head_meta.get("draw_grid_size"), int(self.head_meta.get("draw_slot", 0)),
                                             temperature=float(self.head_meta.get("temperature", 1.0)), **kw)
            p = p[0].double().cpu().numpy()
            nat = torch.softmax(ell[:, :kmax].double().masked_fill(
                torch.arange(kmax, device=ell.device)[None] >= ks[:, None], float("-inf")), -1).cpu().numpy()
            for b, (qid, order, _) in enumerate(items[i:j]):
                k = len(order)
                pb = p[b, :k] / p[b, :k].sum()
                q = np.zeros(k)
                q[np.asarray(order)] = pb
                nq = np.zeros(k)
                nq[np.asarray(order)] = nat[b, :k]
                out[qid] = q.tolist()
                self.last_raw[qid] = {"native": nq.tolist(), "epistemic": float(epi[0, b]), "aleatoric": float(alea[0, b])}
                if "yes_no" in kw:
                    self.last_raw[qid]["yes_no"] = bool(kw["yes_no"][b, 0] >= 0)
                if "kinds" in kw:
                    self.last_raw[qid]["kind"] = self._kind.get(qid, "choice")
            i = j
        return out

    def prompt(self, u, qid):
        """(2026-10-15) the released prompt, recording the question type for the Latent01 head; the optional ordinal path renders an
        ordinal choice in its criteria order"""
        if qid.split("|", 1)[-1] in self._ord_keys:
            order = list(range(len(u["option_keys"])))
            text = core.build_prompt(self.tok, u, order, self.max_len, chat=self.chat, noul_defs=True)
            check_state_shown(u["state"], text, qid.split("|", 1)[1], self.max_len)   # (2026-10-20) leading whitespace is trimmed
            self._kind[qid] = "score"
            return order, text
        order, text = super().prompt(u, qid)
        self._kind[qid] = u["qtype"] if u["qtype"] in ("choice", "noul") else "choice"
        return order, text

    def __call__(self, state, questions):
        self.last_raw = {}
        self._kind = {}
        self._ord_keys = {k for k, q in questions.items() if self.ordinal and q.get("type") == "choice" and q.get("ordinal") is True}
        resp, _ = super().__call__(state, questions)
        return resp, {"vbll": getattr(self, "last_raw", {})}
