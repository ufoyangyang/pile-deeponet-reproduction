"""Torch-side sampling utilities on top of `data.Dataset`."""

from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .data import Dataset


class WindowDataset:
    """Batched access to the prediction windows of an already generated dataset."""

    def __init__(self, ds: Dataset, cfg: Config, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.t = torch.as_tensor(ds.t, dtype=torch.float32, device=device)
        self.T = torch.as_tensor(ds.T, dtype=torch.float32, device=device)      # (N, nt, nr) degC
        self.c = torch.as_tensor(ds.c, dtype=torch.float32, device=device)      # (N, nt)
        self.gas = torch.as_tensor(ds.gas, dtype=torch.float32, device=device)  # (N, nt) kg
        self.p = torch.as_tensor(ds.p, dtype=torch.float32, device=device)
        self.u1 = torch.as_tensor(ds.u1, dtype=torch.float32, device=device).unsqueeze(-1)
        self.u2 = torch.as_tensor(ds.u2, dtype=torch.float32, device=device)
        self.u3 = torch.as_tensor(ds.u3, dtype=torch.float32, device=device)
        self.t_d = torch.as_tensor(ds.t_d, dtype=torch.float32, device=device)
        self.t_peak = torch.as_tensor(ds.t_peak, dtype=torch.float32, device=device)
        self.peak_T = torch.as_tensor(ds.peak_T, dtype=torch.float32, device=device)
        self.ramp = torch.as_tensor(ds.ramp, dtype=torch.float32, device=device)
        self.h_cool = torch.as_tensor(ds.h_cool, dtype=torch.float32, device=device)
        self.split = np.asarray(ds.split)
        self.x_phys = torch.as_tensor(ds.x, dtype=torch.float32, device=device)
        self.raw = ds
        self.n = len(ds)
        self.nr = self.T.shape[2]
        self.dt = float(ds.t[1] - ds.t[0]) if len(ds.t) > 1 else cfg.physics.save_dt

    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.n

    def time_index(self, cond, d):
        # `cond` is (n,) and `d` is (n,1): keep t_d as a column so that the
        # sum does NOT broadcast into an (n,n) matrix.
        tq = self.t_d[cond].unsqueeze(-1) + d * self.cfg.model.lead_time
        ti = torch.round(tq / self.dt).long().clamp(0, self.T.shape[1] - 1)
        return ti.squeeze(-1)

    def _norm_temp(self, v):
        """Map a raw temperature (degC) history vector onto [0, 1] (Eq. 26 style)."""
        m = self.cfg.model
        return (v - m.T_min) / (m.T_max - m.T_min)

    def _inputs(self, cond, x_norm, d):
        return dict(
            u1=self.u1[cond], u2=self._norm_temp(self.u2[cond]), u3=self._norm_temp(self.u3[cond]),
            x=x_norm, d=d, t_d=self.t_d[cond].unsqueeze(-1), ramp=self.ramp[cond].unsqueeze(-1),
        )

    # ------------------------------------------------------------------ #
    def _sample_d(self, n, gen: torch.Generator):
        """Sample the normalised lead time.

        `train.d_power > 1` pushes samples towards the end of the prediction
        window, where the thermal runaway accelerates exponentially -- uniform
        sampling vastly under-weights that part of the trajectory.
        """
        u = torch.rand(n, 1, generator=gen, device=self.device)
        g = float(getattr(self.cfg.train, "d_power", 1.0))
        if g <= 1.0:
            return u
        return 1.0 - (1.0 - u) ** g

    def sample_data(self, n, gen: torch.Generator):
        cond = torch.randint(0, self.n, (n,), generator=gen, device=self.device)
        d = self._sample_d(n, gen)
        x_norm = torch.rand(n, 1, generator=gen, device=self.device)
        xi = torch.round(x_norm * (self.nr - 1)).long()
        ti = self.time_index(cond, d)
        batch = self._inputs(cond, x_norm, d)
        batch["T"] = self.T[cond, ti, xi.squeeze(-1)].unsqueeze(-1)
        batch["c"] = self.c[cond, ti].unsqueeze(-1)
        batch["gas"] = self.gas[cond, ti].unsqueeze(-1)
        return batch

    def sample_physics(self, n, gen: torch.Generator):
        cond = torch.randint(0, self.n, (n,), generator=gen, device=self.device)
        d = self._sample_d(n, gen)
        x_norm = 0.02 + 0.96 * torch.rand(n, 1, generator=gen, device=self.device)
        return self._inputs(cond, x_norm, d)

    def sample_bc(self, n, gen: torch.Generator):
        cond = torch.randint(0, self.n, (n,), generator=gen, device=self.device)
        d = torch.rand(n, 1, generator=gen, device=self.device)
        x_norm = torch.ones(n, 1, device=self.device)
        return self._inputs(cond, x_norm, d)

    def sample_ic(self, n, gen: torch.Generator):
        """Initial condition at the start of a local temporal domain (paper)."""
        m = self.cfg.model
        n_win = max(1, int(round(m.lead_time / m.local_window)))
        cond = torch.randint(0, self.n, (n,), generator=gen, device=self.device)
        win = torch.randint(0, n_win, (n,), generator=gen, device=self.device)
        d = (win.float() * m.local_window / m.lead_time).unsqueeze(-1)
        x_norm = torch.rand(n, 1, generator=gen, device=self.device)
        xi = torch.round(x_norm * (self.nr - 1)).long()
        ti = self.time_index(cond, d)
        batch = self._inputs(cond, x_norm, d)
        batch["T_ic"] = self.T[cond, ti, xi.squeeze(-1)].unsqueeze(-1)
        batch["c_ic"] = self.c[cond, ti].unsqueeze(-1)
        batch["gas_ic"] = self.gas[cond, ti].unsqueeze(-1)
        return batch

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def window_grid(self, i, n_t=None):
        """Ground truth of condition i on the (n_t, nr) window grid."""
        m = self.cfg.model
        n_t = n_t or int(round(m.lead_time / self.dt)) + 1
        i0 = int(round(float(self.t_d[i]) / self.dt))
        i1 = min(i0 + n_t, self.T.shape[1])
        T = self.T[i, i0:i1]
        c = self.c[i, i0:i1]
        gas = self.gas[i, i0:i1]
        p = self.p[i, i0:i1]
        t = self.t[i0:i1]
        return t, T, c, gas, p

    def window_times(self, i, n_t=None):
        t, *_ = self.window_grid(i, n_t)
        return t
