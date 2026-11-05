"""Losses of the PILE-DeepONet (paper Eq. 12-18, 27-30).

All residuals are made dimensionless with the reference scales of the paper's
non-dimensionalised equations (Eq. 27-29), so that the data, physics, boundary
and initial-condition terms are of order one before the adaptive NTK weights
are applied.
"""

from __future__ import annotations

import torch

from .config import Config


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


class PILLoss:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.m = cfg.model
        self.p = cfg.physics
        self.R = self.p.radius
        span = self.m.T_max - self.m.T_min
        self.sT = self.p.rho * self.p.cp * span / self.m.lead_time      # W/m^3
        self.sC = 1.0 / self.m.lead_time                               # 1/s
        self.sG = self.p.Gas0 / self.m.lead_time                       # kg/s
        self.sBC = self.sT * self.R                                    # W/m^2
        self.span = span

    # ------------------------------------------------------------------ #
    def heater_temperature(self, t, ramp):
        p = self.p
        frac = torch.clamp(t / ramp.clamp_min(1.0), 0.0, 1.0)
        return p.T_init + (p.T_target - p.T_init) * frac

    # ------------------------------------------------------------------ #
    def residuals(self, model, b):
        """Dimensionless residuals of one batch.

        Returns dict with keys 'T','c','gas' (physics) and 'bc','ic_T','ic_c',
        'ic_gas' (boundary / initial) as (N,) tensors.
        """
        p, m = self.p, self.m
        x = b["x"].clone().requires_grad_(True)
        d = b["d"].clone().requires_grad_(True)
        T, c, gas = model(b["u1"], b["u2"], b["u3"], x, d)

        # one backward pass for both first derivatives of T, one for the
        # second spatial derivative, one per ODE field -> 4 vjps instead of 5
        one = torch.ones_like(T)
        dT_dx, dT_dd = torch.autograd.grad(
            T, (x, d), grad_outputs=one, create_graph=True, retain_graph=True
        )
        d2T_dx2 = _grad(dT_dx, x)
        dc_dd = _grad(c, d)
        dgas_dd = _grad(gas, d)

        rate = p.A * torch.exp(-p.E / (p.Rc * (T + 273.15))) * c
        r = x * self.R
        lap = dT_dx / (r * self.R) + d2T_dx2 / self.R ** 2

        res_T = (p.rho * p.cp * dT_dd / m.lead_time - p.k * lap - p.H * p.W * rate) / self.sT
        res_c = (dc_dd / m.lead_time + rate) / self.sC
        res_g = (dgas_dd / m.lead_time - (p.Gas0 - gas) * rate) / self.sG

        # surface boundary condition (Eq. 4 / Eq. 30)
        t = b["t_d"] + d * m.lead_time
        T_ext = self.heater_temperature(t, b["ramp"])
        h_cool = b["u1"] * 20.0
        switch = (c.detach() >= p.c_off).to(T.dtype)
        h_eff = switch * p.h_heater + h_cool
        flux = p.k * dT_dx / self.R
        res_bc = (flux - h_eff * (T_ext - T)) / self.sBC

        out = dict(T=res_T, c=res_c, gas=res_g, bc=res_bc)
        if "T_ic" in b:
            out["ic_T"] = (T - b["T_ic"]) / self.span
            out["ic_c"] = (c - b["c_ic"])
            out["ic_gas"] = (gas - b["gas_ic"]) / p.Gas0
        return out

    # ------------------------------------------------------------------ #
    def data_loss(self, model, b):
        T, c, gas = model(b["u1"], b["u2"], b["u3"], b["x"], b["d"])
        lT = ((T - b["T"]) / self.span) ** 2
        lc = (c - b["c"]) ** 2
        lg = ((gas - b["gas"]) / self.p.Gas0) ** 2
        return lT, lc, lg, (T, c, gas)

    # ------------------------------------------------------------------ #
    def terms(self, model, batches, use_physics=True, use_bc=True, use_ic=True):
        """Return a dict {term_name: (scalar loss, weight_hint)} for one iteration.

        Each term is the *mean* squared residual of its batch.
        """
        out = {}
        groups = []
        if use_physics and batches.get("phy") is not None:
            groups.append(("phy", batches["phy"]))
        if use_bc and batches.get("bc") is not None:
            groups.append(("bc", batches["bc"]))
        if use_ic and batches.get("ic") is not None:
            groups.append(("ic", batches["ic"]))

        if groups:
            # a single forward + backward pass over the concatenated collocation
            # batch: the residuals of the physics / boundary / initial terms only
            # differ in which slice of the batch they are evaluated on, so one
            # autodiff graph serves all of them (~3x faster than three calls).
            n_tot = sum(b["x"].shape[0] for _, b in groups)
            keys = set().union(*[set(b) for _, b in groups])
            cat = {}
            for k in keys:
                parts = []
                for _, b in groups:
                    v = b.get(k)
                    if v is None:
                        v = torch.zeros((b["x"].shape[0],) + tuple(b["x"].shape[1:]),
                                        dtype=b["x"].dtype, device=b["x"].device)
                    parts.append(v)
                cat[k] = torch.cat(parts, dim=0)
            r = self.residuals(model, cat)

            idx, spans = 0, {}
            for name, b in groups:
                n = b["x"].shape[0]
                spans[name] = slice(idx, idx + n)
                idx += n
            if "phy" in spans:
                sl = spans["phy"]
                out["phy_T"] = r["T"][sl].pow(2).mean()
                out["phy_c"] = r["c"][sl].pow(2).mean()
                out["phy_gas"] = r["gas"][sl].pow(2).mean()
            if "bc" in spans:
                out["bc_T"] = r["bc"][spans["bc"]].pow(2).mean()
            if "ic" in spans:
                sl = spans["ic"]
                out["ic_T"] = r["ic_T"][sl].pow(2).mean()
                out["ic_c"] = r["ic_c"][sl].pow(2).mean()
                out["ic_gas"] = r["ic_gas"][sl].pow(2).mean()
        if batches.get("data") is not None:
            lT, lc, lg, _ = self.data_loss(model, batches["data"])
            out["data_T"] = lT.mean()
            out["data_c"] = lc.mean()
            out["data_gas"] = lg.mean()
        return out
