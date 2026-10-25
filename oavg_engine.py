#!/usr/bin/env python3
"""oavg_engine v1 (MACKAY, 2026-10-22): the dikit39 engine (dikit_engine.DikitEngine, UNCHANGED components) + option-ORDER AVERAGING, the
recipe pre-registered in AGENT_NOTES 2026-10-22 (board confirmation of order averaging on allec):

  base order   the engine's own order for a question (dikit39: hash-derived non-identity order for choice, fixed order for yes / no).
               A re-lettering r shows, at letter position p, the option the base order shows at position r[p]; the prompt is rendered by
               the engine's own renderer (garas_core.build_prompt, noul_defs=True) for that order and probabilities are mapped back to
               options by the engine's own run_prompts.
  choice, K <= 52 (not a K > 52 chunk / final question): the served head's posterior averaged with equal weight over k = 4 orders
               r_0 = identity, r_j = np.random.default_rng(20261021 + 1000 j + K).permutation(K), j = 1, 2, 3 (duplicates kept as drawn),
               then p <- p^(1 / T_head) / sum, T_head = 0.95438.
  noul (yes / no): the NATIVE letter softmax averaged over the base and the swapped order, then p^(1 / T_noul), T_noul = 0.93738.
  K > 52 questions (chunks and the round-2 final): the dikit39 path unchanged.
All prompts of a request (the re-lettered ones included) go through ONE run_prompts call, i.e. they are packed into the same forward
pass(es). oavg=off (or k=1) returns DikitEngine's run_prompts output itself: byte-identical to dikit39.

    python -m decision_index run --engine oavg_engine:OAVGEngine --option model=<export> [--option oavg=off] ...
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import garas_core as core                                      # noqa: E402
from dikit_engine import DikitEngine                           # noqa: E402

VERSION = "oavg_engine v1 (2026-10-22)"
SEED = 20261021
T_HEAD = 0.95438
T_NOUL = 0.93738
K_AVG = 4


def random_orders(K, k=K_AVG):
    """the pre-registered re-letterings for a K-option choice: identity + k - 1 fixed-seed permutations (duplicates kept)"""
    return [list(range(K))] + [[int(x) for x in np.random.default_rng(SEED + 1000 * j + K).permutation(K)] for j in range(1, k)]


def relettered(base_order, r):
    """the displayed order after re-lettering r: position p shows base_order[r[p]]"""
    return [base_order[j] for j in r]


def combine(prob_lists, T):
    """equal-weight mean of probability vectors over the same options, then p^(1/T) renormalised (float64)"""
    P = np.asarray(prob_lists, dtype=np.float64).mean(0)
    Q = np.where(P > 0, np.power(np.maximum(P, 1e-300), 1.0 / T), 0.0)
    return Q / Q.sum()


class OAVGEngine(DikitEngine):
    name = "sd31-dikit-oavg"

    def __init__(self, model="", oavg="on", oavg_k=K_AVG, **options):
        super().__init__(model=model, **options)
        self.oavg = str(oavg).lower() not in ("off", "0", "false", "no") and int(oavg_k) > 1
        self.oavg_k = int(oavg_k)
        self._alts, self._k52 = {}, set()
        self.provenance = {**self.provenance, "oavg_engine": VERSION, "oavg_engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
                           "oavg": {"on": self.oavg, "k": self.oavg_k, "seed": SEED, "T_head": T_HEAD, "T_noul": T_NOUL,
                                    "choice": "served head, random set (identity + fixed-seed permutations), K <= 52",
                                    "noul": "native softmax, base + swapped order", "k52": "dikit39 path unchanged"}}

    @classmethod
    def from_engine(cls, eng, oavg="on", oavg_k=K_AVG):
        """(tests) an OAVGEngine sharing an already-loaded DikitEngine's model / tokenizer / head (one 31B copy on the GPU)"""
        e = cls.__new__(cls)
        e.__dict__.update(eng.__dict__)
        e.oavg = str(oavg).lower() not in ("off", "0", "false", "no") and int(oavg_k) > 1
        e.oavg_k = int(oavg_k)
        e._alts, e._k52, e._kind, e.last_raw = {}, set(), {}, {}
        e.provenance = {**eng.provenance, "oavg_engine": VERSION, "oavg_engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
                        "oavg": {"on": e.oavg, "k": e.oavg_k, "seed": SEED, "T_head": T_HEAD, "T_noul": T_NOUL,
                                 "choice": "served head, random set (identity + fixed-seed permutations), K <= 52",
                                 "noul": "native softmax, base + swapped order", "k52": "dikit39 path unchanged"}}
        return e

    def __call__(self, state, questions):
        self._alts = {}
        self._k52 = {k for k, q in questions.items() if q.get("type") == "choice" and len(q.get("criteria") or {}) > core.MAX_K}
        return super().__call__(state, questions)

    def _is_k52(self, qid):
        rest = qid.split("|", 1)[1] if "|" in qid else qid
        return any(rest == k or rest.startswith(k + "|") for k in self._k52)

    def prompt(self, u, qid):
        order, text = super().prompt(u, qid)
        if self.oavg and not self._is_k52(qid):
            K = len(order)
            if self._kind.get(qid) == "noul":
                rs = [list(range(K)), list(range(K))[::-1]] if K == 2 else [list(range(K))]
            else:
                rs = random_orders(K, self.oavg_k)
            alts = []
            for r in rs[1:]:
                o = relettered(order, r)
                t = core.build_prompt(self.tok, u, o, self.max_len, chat=self.chat, noul_defs=True)
                alts.append((o, t))
            self._alts[qid] = alts
        return order, text

    def run_prompts(self, items):
        if not self.oavg:
            return super().run_prompts(items)
        expanded = []
        for qid, order, text in items:
            expanded.append((qid, order, text))
            for j, (o, t) in enumerate(self._alts.get(qid, [])):
                qj = f"{qid}#oa{j + 1}"
                self._kind[qj] = self._kind.get(qid, "choice")
                expanded.append((qj, o, t))
        raw = super().run_prompts(expanded)
        out = {}
        for qid, order, _ in items:
            alts = self._alts.get(qid, [])
            if not alts:
                out[qid] = raw[qid]
                continue
            names = [qid] + [f"{qid}#oa{j + 1}" for j in range(len(alts))]
            if self._kind.get(qid) == "noul":
                p = combine([self.last_raw[q]["native"] for q in names], T_NOUL)
            else:
                p = combine([raw[q] for q in names], T_HEAD)
            out[qid] = p.tolist()
            self.last_raw[qid]["oavg"] = {"orders": len(names), "readout": "native" if self._kind.get(qid) == "noul" else "head"}
            for q in names[1:]:
                self.last_raw.pop(q, None)
                self._kind.pop(q, None)
        return out
