#!/usr/bin/env python3
"""lenavg_bayes_engine v1 (MACKAY, 2026-10-23; pre-registered in AGENT_NOTES 2026-10-23 "lenavg_bayes"): lenavg_engine.LenAvgEngine (the
L <= 939 rule, dikit42 recipe for choices, unchanged) with the Bayesian head KEPT ACTIVE on averaged yes / no (noul) questions (variant V4):

  p_len   = lenavg's noul output: base + swapped orders' native probabilities, averaged, then p^(1/T_NOUL) renormalised (T_NOUL 0.93738)
  ell_avg = log(averaged native probability before T_NOUL), in the BASE prompt's letter order
  p_on    = the head's own l01 readout on (h_base, ell_avg), trained constants (beta * dm_noul shift, infl * var_noul posterior variance)
  p_off   = the same readout with beta = 0 and infl = 0            (h_base = the served, yes-first prompt's hidden state)
  Delta   = logodds_A(p_on) - logodds_A(p_off)
  p_V4    = the binary distribution with logodds_A = logodds_A(p_len) + Delta, mapped back to options
No extra temperature. Identity: head inactive (beta = 0, infl = 0 or var = 0) -> Delta = 0 -> p_V4 = p_len. Unaveraged requests (L > L*),
K > 52 chunks and choices: exactly lenavg (choices: the served head averaged over k = 4 orders, T_HEAD). Optional per-question activity log
(env LENAVG_BAYES_ACTIVITY = a file path): kind, route, |Delta|, |p_V4 - p_len| (averaged noul) or |head - native| (unaveraged), head MI.

    python -m decision_index run --engine lenavg_bayes_engine:LenAvgBayesEngine --option model=<export> ...
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lenavg_engine as le                                    # noqa: E402
import oavg_engine as oe                                      # noqa: E402
import vbll_core                                              # noqa: E402

VERSION = "lenavg_bayes_engine v1 (2026-10-23)"


def _logodds(p0, p1):
    return math.log(max(p0, 1e-300)) - math.log(max(p1, 1e-300))


class LenAvgBayesEngine(le.LenAvgEngine):
    name = "sd31-dikit-lenavg-bayes"

    def __init__(self, model="", **options):
        super().__init__(model=model, **options)
        self._stamp()

    def _stamp(self):
        self.provenance = {**self.provenance, "lenavg_bayes_engine": VERSION,
                           "lenavg_bayes_engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
                           "lenavg_bayes": {"variant": "V4 head-in-the-loop", "noul": "logodds(p_len) + [logodds(head on) - logodds(head beta=0, infl=0)]",
                                            "head_input": "log averaged native (pre T_NOUL), base letter order; h from the base prompt",
                                            "extra_T": None}}
        self._act_path = os.environ.get("LENAVG_BAYES_ACTIVITY") or None

    @classmethod
    def from_engine(cls, eng, l_star=le.L_STAR, oavg="on"):
        e = le.LenAvgEngine.from_engine.__func__(cls, eng, l_star=l_star, oavg=oavg)
        e._stamp()
        return e

    # ---- the head's binary correction on (h_base, ell_avg)
    def _head_delta(self, h_row, ell_letters):
        import torch
        inner = getattr(self.head, "inner", self.head)
        st = inner.state()
        T = float(self.head_meta.get("temperature", 1.0))
        ks = torch.tensor([2])
        kinds = torch.tensor([vbll_core.KIND_CODE["noul"]])
        d = vbll_core.l01_extract(h_row.cpu(), ell_letters.cpu(), ks, kinds, st)
        p_on, mi = vbll_core.l01_probs(d, st, T, inner.c)
        p_off, _ = vbll_core.l01_probs(d, st, T, dict(inner.c, beta=0.0, infl=0.0))
        p_on, p_off = p_on[0].double().tolist(), p_off[0].double().tolist()
        return _logodds(p_on[0], p_on[1]) - _logodds(p_off[0], p_off[1]), float(mi[0]), p_on, p_off

    def _log(self, rec):
        if self._act_path:
            with open(self._act_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def run_prompts(self, items):
        import torch
        if not self.oavg:
            out = oe.OAVGEngine.run_prompts(self, items)            # averaging off: the dikit39 path (OAVGEngine returns it unchanged)
            if self._act_path:
                for qid, order, _ in items:
                    lr = self.last_raw.get(qid, {})
                    nat = lr.get("native")
                    self._log({"qid": qid, "kind": self._kind.get(qid, "choice"), "route": "unaveraged",
                               "abs_head_minus_native": float(np.max(np.abs(np.asarray(out[qid]) - np.asarray(nat)))) if nat is not None else None,
                               "mi": lr.get("epistemic")})
            return out
        expanded = []
        for qid, order, text in items:
            expanded.append((qid, order, text))
            for j, (o, t) in enumerate(self._alts.get(qid, [])):
                qj = f"{qid}#oa{j + 1}"
                self._kind[qj] = self._kind.get(qid, "choice")
                expanded.append((qj, o, t))
        # capture the hidden state / letter logits of every prompt, by position (run_prompts forwards consecutive slices in order)
        cap, pos = {}, [0]
        fwd0 = self._forward

        def fwd(texts):
            h, ell = fwd0(texts)
            for b in range(len(texts)):
                cap[expanded[pos[0] + b][0]] = (h[b:b + 1].detach(), ell[b:b + 1].detach())
            pos[0] += len(texts)
            return h, ell
        self._forward = fwd
        try:
            raw = oe.DikitEngine.run_prompts(self, expanded)
        finally:
            self._forward = fwd0
        if pos[0] != len(expanded):
            raise RuntimeError(f"lenavg_bayes: captured {pos[0]} prompts of {len(expanded)}")
        out = {}
        for qid, order, _ in items:
            alts = self._alts.get(qid, [])
            if not alts:
                out[qid] = raw[qid]
                self._log({"qid": qid, "kind": self._kind.get(qid, "choice"), "route": "no_alternatives"})
                continue
            names = [qid] + [f"{qid}#oa{j + 1}" for j in range(len(alts))]
            if self._kind.get(qid) == "noul":
                nats = [self.last_raw[q]["native"] for q in names]
                p_len = oe.combine(nats, oe.T_NOUL)
                avg = np.asarray(nats, dtype=np.float64).mean(0)
                h_b, ell_b = cap[qid]
                ell_letters = ell_b[:, :2].clone()
                ell_letters[0, 0] = math.log(max(avg[order[0]], 1e-300))
                ell_letters[0, 1] = math.log(max(avg[order[1]], 1e-300))
                delta, mi, p_on, p_off = self._head_delta(h_b, ell_letters)
                lo = _logodds(p_len[order[0]], p_len[order[1]]) + delta
                pa = 1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0
                p = np.zeros(2)
                p[order[0]], p[order[1]] = pa, 1.0 - pa
                out[qid] = p.tolist()
                self.last_raw[qid]["oavg"] = {"orders": len(names), "readout": "native+head_delta (V4)"}
                self.last_raw[qid]["bayes"] = {"delta": delta, "mi": mi, "p_len": p_len.tolist(), "p_v4": p.tolist(), "p_on": p_on, "p_off": p_off,
                                               "head_per_order": [raw[q] for q in names]}
                self._log({"qid": qid, "kind": "noul", "route": "averaged", "abs_delta": abs(delta),
                           "abs_p_v4_minus_p_len": float(np.max(np.abs(p - p_len))), "mi": mi})
            else:
                p = oe.combine([raw[q] for q in names], oe.T_HEAD)
                out[qid] = p.tolist()
                self.last_raw[qid]["oavg"] = {"orders": len(names), "readout": "head"}
                self._log({"qid": qid, "kind": self._kind.get(qid, "choice"), "route": "averaged",
                           "abs_head_minus_native": float(np.max(np.abs(np.asarray(raw[qid]) - np.asarray(self.last_raw[qid]["native"])))),
                           "mi": self.last_raw[qid].get("epistemic")})
            for q in names[1:]:
                self.last_raw.pop(q, None)
                self._kind.pop(q, None)
        return out
