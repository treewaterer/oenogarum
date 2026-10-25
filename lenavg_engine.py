#!/usr/bin/env python3
"""lenavg_engine v1 (MACKAY, 2026-10-22): LENGTH-ADAPTIVE option-order averaging = oavg_engine.OAVGEngine (the dikit42 recipe, unchanged,
over the dikit39 components) applied to a request iff its length L <= L_STAR = 939, else the dikit39 single pass (OAVGEngine with
averaging off returns DikitEngine.run_prompts itself). Pre-registered in AGENT_NOTES 2026-10-22 (L* chosen from latency only):

  L(request) = sum over the request's questions of [len(state) + len(instructions) + sum over its criteria (len(key) + len(str(value)))]
               (Python str characters; computed from the request alone, no labels)

    python -m decision_index run --engine lenavg_engine:LenAvgEngine --option model=<export> ...
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import oavg_engine as oe                                       # noqa: E402

VERSION = "lenavg_engine v1 (2026-10-22)"
L_STAR = 939


def request_length(state, questions):
    st = state or ""
    tot = 0
    for q in (questions or {}).values():
        tot += len(st) + len(q.get("instructions") or "") + sum(len(str(k)) + len(str(v)) for k, v in (q.get("criteria") or {}).items())
    return tot


class LenAvgEngine(oe.OAVGEngine):
    name = "sd31-dikit-lenavg"

    def __init__(self, model="", l_star=L_STAR, **options):
        super().__init__(model=model, **options)
        self._avg_allowed = self.oavg
        self.l_star = int(l_star)
        self.last_choice = None
        self.provenance = {**self.provenance, "lenavg_engine": VERSION, "lenavg_engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
                           "lenavg": {"L_star": self.l_star, "measure": "sum over questions of len(state)+len(instructions)+sum(len(key)+len(str(value)))",
                                      "averaging_allowed": self._avg_allowed}}

    @classmethod
    def from_engine(cls, eng, l_star=L_STAR, oavg="on"):
        """(tests) a LenAvgEngine sharing an already-loaded DikitEngine's model"""
        e = oe.OAVGEngine.from_engine.__func__(cls, eng, oavg=oavg)
        e._avg_allowed = e.oavg
        e.l_star = int(l_star)
        e.last_choice = None
        e.provenance = {**e.provenance, "lenavg_engine": VERSION, "lenavg_engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
                        "lenavg": {"L_star": e.l_star, "measure": "sum over questions of len(state)+len(instructions)+sum(len(key)+len(str(value)))",
                                   "averaging_allowed": e._avg_allowed}}
        return e

    def __call__(self, state, questions):
        L = request_length(state, questions)
        self.oavg = self._avg_allowed and L <= self.l_star
        self.last_choice = {"L": L, "averaged": self.oavg}
        try:
            return super().__call__(state, questions)
        finally:
            self.oavg = self._avg_allowed
