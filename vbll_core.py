"""Residual variational Bayesian last layer (VBLL) decision head, inference only: a self-contained copy of the predictive math of the
training head (AuxVBLL, claude_lmtune.py e2c6100a). test_fullft.py checks this file against the training class (same state, same
inputs -> the same numbers).

For the option shown under letter j of a K-option prompt, from ONE backbone forward (final hidden state h at the decision token,
native letter logits ell):
    z_j = tau * ell_j + phi_j^T beta,   q(beta) = N(mu, S)
    phi_j = u * (c_j - mean_{k<K} c_k) / sqrt(F),   u = rms(P rms(h)),   c_j = the fixed letter codes
    answer = E_q softmax(z)   (Monte Carlo over q with a FIXED seeded set of draws: deterministic)
The head expresses uncertainty in its own weights CONDITIONAL on the backbone's features; it says nothing about backbone uncertainty.
"""
import json
import math
import os
import re

import torch

HEAD_FILE, HEAD_META = "vbll_head.safetensors", "vbll_head.json"


class ResidualVBLL(torch.nn.Module):
    def __init__(self, state, heads, dim):
        super().__init__()
        self.heads, self.F = [dict(h) for h in heads], int(dim)
        self.P = torch.nn.Parameter(state["P"].float(), requires_grad=False)
        self.mu = torch.nn.Parameter(state["mu"].float(), requires_grad=False)
        self.L_diag = torch.nn.Parameter(state["L_diag"].float(), requires_grad=False)
        self.L_off = torch.nn.Parameter(state["L_off"].float(), requires_grad=False) if "L_off" in state else None
        self.log_tau = torch.nn.Parameter(state["log_tau"].float(), requires_grad=False)
        self.register_buffer("codes", state["codes"].float())
        full = torch.tensor([h["cov"] == "full" for h in heads])
        if bool(full.any()) and self.L_off is None:
            raise ValueError("a full-covariance head without its off-diagonal factor")
        self.register_buffer("full", full.float()[:, None, None], persistent=False)

    @classmethod
    def from_dir(cls, d):
        from safetensors.torch import load_file
        import hashlib
        p, j = os.path.join(d, HEAD_FILE), os.path.join(d, HEAD_META)
        if not (os.path.exists(p) and os.path.exists(j)):
            raise FileNotFoundError(f"{d}: no residual VBLL head ({HEAD_FILE} + {HEAD_META}). This model's answers are the head's "
                                    "posterior predictive; it must not be served from the backbone's letter logits alone.")
        meta = json.load(open(j))
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != meta["tensor_sha256"]:
            raise ValueError(f"{p}: sha256 differs from {HEAD_META}")
        return cls(load_file(p), meta["heads"], meta["dim"]), meta

    def sigma(self):
        return torch.nn.functional.softplus(self.L_diag) + 1e-6

    def chol(self):
        return torch.tril(self.L_off, -1) * self.full + torch.diag_embed(self.sigma())

    def moments(self, h, ell, ks, want_phi=False):
        kmax = ell.shape[1]
        mask = torch.arange(kmax, device=ell.device)[None, :] < ks[:, None]
        m = mask.float()
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        u = torch.einsum("gfh,bh->gbf", self.P, h)
        u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6)
        w = self.codes[:, :kmax]
        wbar = torch.einsum("bk,gkf->gbf", m, w) / ks.to(w.dtype).clamp_min(1)[None, :, None]
        phi = u[:, :, None, :] * (w[:, None, :, :] - wbar[:, :, None, :]) * (m[None, :, :, None] / math.sqrt(self.F))
        mean = self.log_tau.exp()[:, None, None] * ell.float()[None] + torch.einsum("gbkf,gf->gbk", phi, self.mu)
        if self.L_off is None:
            var = (phi.pow(2) * self.sigma().pow(2)[:, None, None, :]).sum(-1)
        else:
            var = torch.einsum("gbkf,gfe->gbke", phi, self.chol()).pow(2).sum(-1)
        return (mean, var, mask, phi) if want_phi else (mean, var, mask)

    @torch.no_grad()
    def predict(self, h, ell, ks, n_samples=64, seed=0, draw_grid_size=None, draw_slot=0, temperature=1.0):
        """-> p [G, B, K] (E_q softmax), epistemic (mutual information) [G, B], aleatoric (expected entropy) [G, B]. The draws are those
        the head had in its training grid (draw_grid_size heads, this head's slot): the training-side Monte Carlo set, so the served
        predictive is exactly the one evaluated on DEV."""
        mean, var, mask, phi = self.moments(h, ell, ks, want_phi=True)
        G = self.mu.shape[0]
        n_grid = int(draw_grid_size or G)
        eps = torch.randn(n_grid, int(n_samples), self.F, generator=torch.Generator().manual_seed(int(seed)))
        eps = eps[int(draw_slot):int(draw_slot) + G] if n_grid != G else eps
        eps = eps.to(phi.device)
        if self.L_off is None:
            beta = self.mu[:, None] + eps * self.sigma()[:, None]
        else:
            beta = self.mu[:, None] + torch.einsum("gef,gsf->gse", self.chol(), eps)
        z = self.log_tau.exp()[:, None, None, None] * ell.float()[None, None] + torch.einsum("gbkf,gsf->gsbk", phi, beta)
        if float(temperature) != 1.0:                    # (stock_decision_v1) mean_s softmax(z_s / T): T divides each draw's logits
            z = z / float(temperature)
        ps = torch.softmax(z.masked_fill(~mask[None, None], -1e9), -1)
        p = ps.mean(1)

        def ent(q):
            return -(q * q.clamp_min(1e-12).log()).sum(-1)
        alea = ent(ps).mean(1)
        epi = (ent(p) - alea).clamp_min(0.0)
        return p, epi, alea


FIXEDE_SCHEMA = "jev-fixede-head-v1"


def fe_eps(F, pairs=32, seed=0):
    """(2026-10-15) the fixedE draws: seed-0 normals [pairs, F] (fp64), then their negatives (sdv1_head.fe_eps is the same code)"""
    e = torch.randn(int(pairs), int(F), generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float64)
    return torch.cat([e, -e])


class FixedEHead(torch.nn.Module):
    """(2026-10-15) the FIXED-E SUBSPACE head (sdv1_head v3.7 rule fixedE), inference only, norm 'none':
        phi_bk = (h V) * ((E_k - mean_{j<K_b} E_j) V) = (h V) * (codes_k - mean codes),   codes = E V (built at export)
        z_bk = ell_bk + phi_bk . mu + phi_bk . L eps_s,   eps: the export's antithetic draws,   p = mean_s softmax(z_s / T)
    The same operations, in the same order and precision, as sdv1_head.fe_phi / fe_probs."""

    def __init__(self, state, meta):
        super().__init__()
        self.heads = [dict(h) for h in meta["heads"]]
        self.F = int(meta["dim"])
        dr = meta.get("draws") or {}
        if not dr.get("antithetic") or meta.get("norm") != "none":
            raise ValueError("fixedE head: expected norm 'none' and antithetic draws")
        self.pairs, self.seed = int(dr["pairs"]), int(dr["seed"])
        for k in ("V", "codes", "mu", "L"):
            self.register_buffer(k, state[k].double())
        self.mu = self.mu.reshape(1, -1) if self.mu.dim() == 1 else self.mu     # [1, F]: G = 1, as ResidualVBLL

    @classmethod
    def from_dir(cls, d, meta):
        from safetensors.torch import load_file
        return cls(load_file(os.path.join(d, HEAD_FILE)), meta)

    def phi(self, h, ks, K):
        n = int(ks.shape[0])
        mask = torch.arange(K, device=h.device)[None, :] < ks[:, None]
        m = mask.double()
        hE = h.double() @ self.V
        ek = self.codes[:K][None].expand(n, K, -1)
        ebar = (ek * m[..., None]).sum(1, keepdim=True) / m.sum(1)[:, None, None]
        return hE[:, None, :] * (ek - ebar) * m[..., None], mask

    def predict(self, h, ell, ks, n_samples=None, seed=None, draw_grid_size=None, draw_slot=0, temperature=1.0):
        """-> p [1, B, K], epistemic [1, B], aleatoric [1, B]. The draws are antithetic pairs: n_samples (even; default the export's
        2 x pairs) and seed (default the export's) are HONOURED (n_samples / 2 seed-`seed` normals and their negatives); the
        temperature was fitted with the export's draw spec. A draw grid (the residual VBLL's grid slots) does not exist here: refused."""
        if draw_grid_size not in (None, 1) or int(draw_slot or 0) != 0:
            raise ValueError("fixedE head: no draw grid (draw_grid_size must be None / 1 and draw_slot 0)")
        n = 2 * self.pairs if n_samples is None else int(n_samples)
        if n <= 0 or n % 2:
            raise ValueError(f"fixedE head: n_samples must be even and > 0 (antithetic pairs), got {n_samples}")
        K = ell.shape[1]
        phi, mask = self.phi(h, ks, K)
        eps = fe_eps(self.F, n // 2, self.seed if seed is None else int(seed))
        mean = ell.double() + phi @ self.mu[0]
        noise = torch.einsum("bkf,sf->sbk", phi, eps.to(phi.device) @ self.L.T).float()
        z = (mean.float()[None] + noise) / float(temperature)
        ps = torch.softmax(z.masked_fill(~mask[None], -1e9), -1)
        p = ps.mean(0).masked_fill(~mask, 0.0)

        def ent(q):
            return -(q * q.clamp_min(1e-12).log()).sum(-1)
        alea = ent(ps).mean(0)
        epi = (ent(p) - alea).clamp_min(0.0)
        return p[None], epi[None], alea[None]


LLOC_SCHEMA = "jev-lloc-head-v1"
OPT_HEAD = "\n\nOptions:\n"
OPT_TAIL = "\n\nAnswer with the letter of the single best option and nothing else."
_YES, _NO = r"Yes(?:: .*)?", r"No(?:: .*)?"


def yes_no_index(prompt, k):
    """(v3.9) THE Yes / No flag of the L-loc readout, from the RENDERED prompt only (no dataset metadata): a K = 2 question whose rendered
    options are exactly Yes and No ('Yes' or 'Yes: <definition>', 'No' or 'No: <definition>', either order) -> (yes position, no
    position); anything else -> None. The options block is the text between the LAST '\\n\\nOptions:\\n' and the LAST '\\n\\nAnswer with
    the letter of the single best option and nothing else.' (garas_core.PROMPT / PROMPT_CASE and the stage-2 renderer).
    release/garas_vbll/vbll_core.yes_no_index is the same code (test_sdv1 checks the text)."""
    if int(k) != 2:
        return None
    i, j = prompt.rfind(OPT_HEAD), prompt.rfind(OPT_TAIL)
    if i < 0 or j < i:
        return None
    block = prompt[i + len(OPT_HEAD):j]
    if re.fullmatch(rf"A\) {_YES}\nB\) {_NO}", block, flags=re.S):
        return (0, 1)
    if re.fullmatch(rf"A\) {_NO}\nB\) {_YES}", block, flags=re.S):
        return (1, 0)
    return None


def yes_no_tensor(prompts, ks):
    """[B, 2] long: yes_no_index per rendered prompt (-1, -1 = not a Yes / No row)"""
    return torch.tensor([yes_no_index(t, int(k)) or (-1, -1) for t, k in zip(prompts, ks)], dtype=torch.long).reshape(-1, 2)


class LlocHead(torch.nn.Module):
    """(2026-10-15) the L-loc head (sdv1_head v3.9 rule lloc), inference only. On the Yes / No rows (yes_no_index of the RENDERED prompt;
    the caller passes yes_no [B, 2]):
        u = rms(h - (h V_E) V_E^T) V_perp,   m_s = (beta / 2) u . (mu + L eps_s),   z_yes = ell_yes + m_s,   z_no = ell_no - m_s
    every other row z = ell; p = mean_s softmax(z_s / T) over the export's antithetic draws. The same operations, in the same order and
    precision (fp64), as sdv1_head.lloc_prep / lloc_probs."""
    needs_yes_no = True

    def __init__(self, state, meta):
        super().__init__()
        self.heads = [dict(h) for h in meta["heads"]]
        self.F = int(meta["dim"])
        self.beta = float(meta["beta"])
        dr = meta.get("draws") or {}
        if not dr.get("antithetic"):
            raise ValueError("L-loc head: expected antithetic draws")
        self.pairs, self.seed = int(dr["pairs"]), int(dr["seed"])
        for k in ("V_E", "V_perp", "mu", "L"):
            self.register_buffer(k, state[k].double())

    @classmethod
    def from_dir(cls, d, meta):
        from safetensors.torch import load_file
        return cls(load_file(os.path.join(d, HEAD_FILE)), meta)

    def u(self, h):
        hd = h.double()
        hp = hd - (hd @ self.V_E) @ self.V_E.T
        return (hp * torch.rsqrt(hp.pow(2).mean(-1, keepdim=True) + 1e-12)) @ self.V_perp

    def predict(self, h, ell, ks, n_samples=None, seed=None, draw_grid_size=None, draw_slot=0, temperature=1.0, yes_no=None):
        """-> p [1, B, K] (fp64), epistemic [1, B], aleatoric [1, B]. yes_no [B, 2] (from the rendered prompts) is REQUIRED. Draws as
        FixedEHead: antithetic pairs, n_samples (even) and seed honoured, no draw grid."""
        if yes_no is None:
            raise ValueError("L-loc head: yes_no (yes_no_tensor of the rendered prompts) is required")
        if draw_grid_size not in (None, 1) or int(draw_slot or 0) != 0:
            raise ValueError("L-loc head: no draw grid (draw_grid_size must be None / 1 and draw_slot 0)")
        n = 2 * self.pairs if n_samples is None else int(n_samples)
        if n <= 0 or n % 2:
            raise ValueError(f"L-loc head: n_samples must be even and > 0 (antithetic pairs), got {n_samples}")
        yes_no = torch.as_tensor(yes_no, dtype=torch.long).reshape(-1, 2).to(ell.device)
        if yes_no.shape[0] != ell.shape[0] or bool(((yes_no[:, 0] >= 0) & (ks.to(yes_no.device) != 2)).any()):
            raise ValueError("L-loc head: yes_no must have one row per prompt, flagged rows K = 2")
        T = float(temperature)
        K = ell.shape[1]
        mask = torch.arange(K, device=ell.device)[None, :] < ks[:, None]
        e = ell.double()
        p = torch.softmax((e / T).masked_fill(~mask, float("-inf")), -1)
        alea = -(p * p.clamp_min(1e-12).log()).sum(-1)
        epi = torch.zeros_like(alea)
        rows = (yes_no[:, 0] >= 0).nonzero().flatten()
        if len(rows):
            u = self.u(h[rows])
            b2 = 0.5 * self.beta
            m, uL = b2 * (u @ self.mu), b2 * (u @ self.L)
            eps = fe_eps(self.F, n // 2, self.seed if seed is None else int(seed)).to(uL.device)
            ms = m[None] + (uL @ eps.T).T
            iy, ino = yes_no[rows, 0], yes_no[rows, 1]
            ar = torch.arange(len(rows), device=e.device)
            d = (e[rows, iy] - e[rows, ino])[None] + 2.0 * ms
            sy, sn = torch.sigmoid(d / T), torch.sigmoid(-d / T)
            py, pn = sy.mean(0), sn.mean(0)
            pr = torch.zeros(len(rows), K, dtype=torch.float64, device=e.device)
            pr[ar, iy], pr[ar, ino] = py, pn
            p[rows] = pr

            def ent2(a_, b_):
                return -(a_ * a_.clamp_min(1e-12).log() + b_ * b_.clamp_min(1e-12).log())
            al = ent2(sy, sn).mean(0)
            alea[rows] = al
            epi[rows] = (ent2(py, pn) - al).clamp_min(0.0)
        return p.masked_fill(~mask, 0.0)[None], epi[None], alea[None]


L01_SCHEMA = "jev-l01post-head-v1"
KIND_CODE = {"choice": 0, "noul": 1, "score": 2}
_GHX, _GHW = __import__("numpy").polynomial.hermite.hermgauss(16)
GH_Z = torch.from_numpy(_GHX * math.sqrt(2.0)).to(torch.float64)
GH_W = torch.from_numpy(_GHW / math.sqrt(math.pi)).to(torch.float64)
L01_FIT = {"chunk": 4096}


def lloc_hperp(h, VE):
    """rms(h - (h V_E) V_E^T) fp64 (the letter-orthogonal state, MACKAY-HEAD FastLatent01._h_perp)"""
    hd = h.double()
    hp = hd - (hd @ VE) @ VE.T
    return hp * torch.rsqrt(hp.pow(2).mean(-1, keepdim=True) + 1e-12)


def l01_ovr(mu_scaled, sigma_01, mask):
    """the reference ovr_scurve_integral (run_latent01_all8.py e80a40ed), same operations: -> p [n, K], mi [n]"""
    m = mask
    z = mu_scaled.masked_fill(~m, -1e9)
    lse_all = torch.logsumexp(z, dim=-1, keepdim=True)
    p0 = torch.softmax(z, dim=-1).masked_fill(~m, 0.0)
    margin = (z - lse_all) - torch.log1p(-p0.clamp_max(1.0 - 1e-12))
    kb = m.double().sum(dim=-1, keepdim=True).clamp_min(2.0)
    sig_eff = sigma_01 * (2.0 / kb).sqrt() * m.double()
    gz, gw = GH_Z.to(z.device), GH_W.to(z.device)
    arg = margin[None, :, :] + sig_eff[None, :, :] * gz[:, None, None]
    f_u = torch.sigmoid(arg).masked_fill(~m[None, :, :], 0.0)
    I_k = torch.einsum("q,qbk->bk", gw, f_u).masked_fill(~m, 0.0)
    p = I_k / I_k.sum(dim=-1, keepdim=True).clamp_min(1e-15)

    def bern(q):
        q_c = q.clamp(1e-15, 1.0 - 1e-15)
        return -(q_c * q_c.log() + (1.0 - q_c) * (1.0 - q_c).log())
    mi_k = (bern(I_k) - torch.einsum("q,qbk->bk", gw, bern(f_u))).clamp_min(0.0) * m.double()
    return p, mi_k.sum(dim=-1) / kb.squeeze(-1)


def l01_score(p_init, k_vec, mask, sig_kern, lam_uni, sig_uni):
    """the reference score_interval_integral (e80a40ed), same operations"""
    if sig_kern <= 0.0 and lam_uni <= 0.0:
        return p_init
    Kmax = p_init.shape[1]
    K_b = k_vec.to(p_init.dtype)[:, None]
    grid = torch.arange(Kmax, dtype=p_init.dtype, device=p_init.device)[None, :]
    u_j = (grid + 0.5) / K_b
    c_lo = grid / K_b
    c_hi = (grid + 1.0) / K_b
    p_out = p_init
    if sig_kern > 0.0:
        bw = (sig_kern / K_b)[:, :, None]
        z_hi = (c_hi[:, None, :] - u_j[:, :, None]) / bw
        z_lo = (c_lo[:, None, :] - u_j[:, :, None]) / bw
        cdf_diff = 0.5 * (torch.erf(z_hi / math.sqrt(2.0)) - torch.erf(z_lo / math.sqrt(2.0)))
        cdf_diff = cdf_diff.masked_fill(~mask[:, None, :], 0.0)
        cdf_diff = cdf_diff / cdf_diff.sum(dim=-1, keepdim=True).clamp_min(1e-15)
        p_out = torch.einsum("bj,bjk->bk", p_out, cdf_diff)
    if lam_uni > 0.0:
        mu_s = (p_init * u_j).sum(dim=-1, keepdim=True)
        var_s = (p_init * (u_j - mu_s).pow(2)).sum(dim=-1, keepdim=True)
        sd_s = (var_s + (sig_uni / K_b).pow(2)).sqrt()
        zhi = (c_hi - mu_s) / sd_s
        zlo = (c_lo - mu_s) / sd_s
        p_uni = (0.5 * (torch.erf(zhi / math.sqrt(2.0)) - torch.erf(zlo / math.sqrt(2.0)))).masked_fill(~mask, 0.0)
        p_uni = p_uni / p_uni.sum(dim=-1, keepdim=True).clamp_min(1e-15)
        p_out = (1.0 - lam_uni) * p_out + lam_uni * p_uni
    return p_out


def l01_extract(h, ell, ks, kinds, st, chunk=L01_FIT["chunk"]):
    """the reference FastLatent01.extract (e80a40ed) on the stored state st: ell_c, dz_c, mask, res, dm_noul, var_noul, row_rms (fp64)"""
    VE, Vp, mu_hp, E = st["V_E"], st["V_perp"], st["mu_hp"].reshape(1, -1), st["E"]
    res_l, un_l = [], []
    for s0 in range(0, h.shape[0], chunk):
        hp_raw = lloc_hperp(h[s0:s0 + chunk], VE)
        hp = hp_raw - mu_hp
        u = hp @ Vp
        res_l.append((hp - u @ Vp.T).pow(2).sum(-1).sqrt())
        un_l.append(hp_raw @ Vp)
    res = torch.cat(res_l) if res_l else torch.zeros(0, dtype=torch.float64)
    u_noul = torch.cat(un_l) if un_l else torch.zeros(0, Vp.shape[1], dtype=torch.float64)
    K = ell.shape[1]
    mask = torch.arange(K, device=ell.device)[None] < ks[:, None]
    m = mask.double()
    e = ell.double()
    z = (h.double() @ E.T)[:, :K]
    ell_c = (e - (e * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    z_c = (z - (z * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)) * m
    n = int(ks.shape[0])
    idx_n = (kinds == KIND_CODE["noul"]).nonzero().flatten()
    dm_noul = torch.zeros(n, dtype=torch.float64)
    var_noul = torch.zeros(n, dtype=torch.float64)
    if len(idx_n) > 0:
        un = u_noul[idx_n]
        dm_noul[idx_n] = (un @ st["mu"]) / 2.0
        var_noul[idx_n] = ((un @ st["S"]) * un).sum(-1).clamp_min(0.0) / 4.0
    row_rms = ((ell_c.pow(2) * m).sum(1) / m.sum(1).clamp_min(1.0) + 1.0).sqrt()
    return {"ell_c": ell_c, "dz_c": z_c - ell_c, "mask": mask, "res": res, "dm_noul": dm_noul, "var_noul": var_noul, "row_rms": row_rms,
            "kinds": kinds, "k": ks}


def l01_probs(d, st, T, c):
    """(v4.0) THE Latent01-post readout = MACKAY-HEAD predict2 (the reference predict with its constants exposed), same operations:
    -> p [n, K] fp64, mi [n] (the S-curve mutual information; zeros on the plain-softmax path)"""
    m = d["mask"]
    kinds = d["kinds"]
    mu = (d["ell_c"] + c["alpha"] * d["dz_c"]).clone()
    is_noul = kinds == KIND_CODE["noul"]
    if c["beta"] > 0.0 and bool(is_noul.any()):
        dm = c["beta"] * d["dm_noul"][is_noul]
        mu[is_noul, 0] += dm
        mu[is_noul, 1] -= dm
    if c["tau"] > 0.0:
        r_scale = torch.ones(len(mu), dtype=torch.float64)
        for kd in ("noul", "choice", "score"):
            r_scale[kinds == KIND_CODE[kd]] = 1.0 + c["tau"] * (st["rel_s"][kd] - 1.0)
        T_row = T * r_scale[:, None]
    else:
        T_row = T
    score_on = c["kern"] > 0.0 or c["lam_uni"] > 0.0
    if c["sig"] <= 0.0 and not score_on and c["tau"] <= 0.0:
        return torch.softmax((mu / T).masked_fill(~m, -1e9), -1).masked_fill(~m, 0.0), torch.zeros(len(mu), dtype=torch.float64)
    mu_scaled = mu / T_row
    rr = d["res"] / st["res_mean"] if c["dnorm"] == "ratio" else torch.ones_like(d["res"])
    sd_row = c["sig"] * rr * d["row_rms"]
    sigma = sd_row[:, None].expand_as(mu).clone()
    if bool(is_noul.any()):
        sigma[is_noul, :2] = (sigma[is_noul, :2].pow(2) + c["infl"] * d["var_noul"][is_noul, None]).sqrt()
    p, mi = l01_ovr(mu_scaled, (sigma / T_row) * m.double(), m)
    is_score = kinds == KIND_CODE["score"]
    if score_on and bool(is_score.any()):
        idx = is_score.nonzero().flatten()
        p[idx] = l01_score(p[idx], d["k"][idx], m[idx], c["kern"], c["lam_uni"], c["sig_uni"])
    return p, mi


class L01PostHead(torch.nn.Module):
    """(2026-10-15) the Latent01-post head (sdv1_head v4.0 rules l01post / l01post_di), inference only: the reference Latent01 readout
    (research/claude_bayes_handoff/latent01_src/run_latent01_all8.py e80a40ed; MACKAY-HEAD predict2 constants) on the stored state.
    The caller passes kinds [B] (choice 0 / noul 1 / score 2; the request's question type). The same functions, in the same order and
    precision (fp64), as sdv1_head.l01_extract / l01_probs (test_sdv1 checks the source text)."""
    needs_kinds = True

    def __init__(self, state, meta):
        super().__init__()
        self.heads = [dict(h) for h in meta["heads"]]
        self.c = dict(meta["constants"])
        self.variant = meta.get("variant", "full")
        self.rel_s = {k: float(v) for k, v in meta["rel_s"].items()}
        self.res_mean = float(meta["res_mean"])
        for k in ("E", "V_E", "V_perp", "mu_hp", "mu", "S"):
            self.register_buffer(k, state[k].double())

    @classmethod
    def from_dir(cls, d, meta):
        from safetensors.torch import load_file
        return cls(load_file(os.path.join(d, HEAD_FILE)), meta)

    def state(self):
        return {"E": self.E, "V_E": self.V_E, "V_perp": self.V_perp, "mu_hp": self.mu_hp, "mu": self.mu, "S": self.S, "rel_s": self.rel_s,
                "res_mean": self.res_mean}

    def predict(self, h, ell, ks, n_samples=None, seed=None, draw_grid_size=None, draw_slot=0, temperature=1.0, kinds=None):
        """-> p [1, B, K] (fp64), epistemic [1, B] (the S-curve mutual information), aleatoric [1, B] (H(p) - epistemic). kinds is REQUIRED.
        Deterministic (16-node quadrature): n_samples / seed are ignored; a draw grid is refused."""
        if kinds is None:
            raise ValueError("Latent01 head: kinds (the request's question types) are required")
        if draw_grid_size not in (None, 1) or int(draw_slot or 0) != 0:
            raise ValueError("Latent01 head: no draw grid")
        kinds = torch.as_tensor(kinds, dtype=torch.long).reshape(-1)
        if kinds.shape[0] != ell.shape[0] or bool(((kinds == KIND_CODE["noul"]) & (ks.cpu() != 2)).any()):
            raise ValueError("Latent01 head: one kind per prompt; noul rows K = 2")
        st = self.state()
        p, mi = l01_probs(l01_extract(h, ell, ks, kinds, st), st, float(temperature), self.c)
        hp_ = -(p * p.clamp_min(1e-12).log()).sum(-1)
        return p[None], mi[None], (hp_ - mi).clamp_min(0.0)[None]


def load_head(d):
    """(2026-10-15) the export's head: the fixedE schema -> FixedEHead; the L-loc schema -> LlocHead; the Latent01 schema -> L01PostHead;
    anything else -> ResidualVBLL (the old modes, unchanged)"""
    import hashlib
    p, j = os.path.join(d, HEAD_FILE), os.path.join(d, HEAD_META)
    if not (os.path.exists(p) and os.path.exists(j)):
        return ResidualVBLL.from_dir(d)
    meta = json.load(open(j))
    if meta.get("schema") not in (FIXEDE_SCHEMA, LLOC_SCHEMA, L01_SCHEMA):
        return ResidualVBLL.from_dir(d)
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != meta["tensor_sha256"]:
        raise ValueError(f"{p}: sha256 differs from {HEAD_META}")
    return {FIXEDE_SCHEMA: FixedEHead, LLOC_SCHEMA: LlocHead, L01_SCHEMA: L01PostHead}[meta["schema"]].from_dir(d, meta), meta


def letter_logits_fp32(h, E, cap):
    """the native readout the head was trained on: fp32 h @ E[letters]^T, then the model's final logit softcap."""
    z = h.float() @ E.float().T
    return cap * torch.tanh(z / cap) if cap else z


def readout_spec(spec):
    """(2026-10-15) the head meta's letter_readout: None / 'stock' -> ('stock', 0); 'blend:A' (0 < A < 1) -> ('blend', A); else refused"""
    spec = spec or "stock"
    if spec == "stock":
        return "stock", 0.0
    mode, _, a = spec.partition(":")
    try:
        av = float(a)
    except ValueError:
        av = -1.0
    if mode != "blend" or not (0.0 < av < 1.0 and math.isfinite(av)):
        raise ValueError(f"letter_readout {spec!r}: 'stock' or 'blend:A' with 0 < A < 1")
    return "blend", av


def letter_readout_fp32(h, E, cap, spec=None):
    """the letter readout the export was TRAINED with (fullft_head.apply_readout, the same expression in the same order): stock = the
    model's softcap; blend:A = (1 - A) cap tanh(z / cap) + A z, z = fp32 h @ E[letters]^T"""
    mode, a = readout_spec(spec)
    if mode == "stock":
        return letter_logits_fp32(h, E, cap)
    if not cap:
        raise ValueError("blend readout needs a softcap")
    z = h.float() @ E.float().T
    return (cap * torch.tanh(z / cap)) * (1.0 - a) + z * a
