"""Training loop with NTK-based adaptive loss weights (paper Sec. 2.2.4)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .batch import WindowDataset
from .config import Config
from .evaluate import evaluate_split
from .losses import PILLoss
from .model import build_model, count_parameters


class Trainer:
    def __init__(self, cfg: Config, train_wds: WindowDataset, val_wds: WindowDataset | None = None,
                 device: str = "cuda", out_dir: str | Path = "runs/base"):
        self.cfg = cfg
        self.t = cfg.train
        self.device = device
        self.wds = train_wds
        self.val_wds = val_wds
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(self.t.seed)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(self.t.seed)

        self.model = build_model(cfg.model, device)
        self.loss = PILLoss(cfg)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.t.lr)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=list(range(self.t.lr_decay_every, self.t.iters, self.t.lr_decay_every)),
            gamma=self.t.lr_decay,
        )
        self.weights = {}
        self.history = []
        self.best = float("inf")
        print(f"[model] parameters: {count_parameters(self.model):,}")

    # ------------------------------------------------------------------ #
    def sample(self, n_phy=None):
        b = dict(data=self.wds.sample_data(self.t.n_data, self.gen))
        if n_phy is None and self.cfg.train.use_physics:
            n_phy = self.t.n_phy
        if n_phy:
            b["phy"] = self.wds.sample_physics(n_phy, self.gen)
            b["bc"] = self.wds.sample_bc(self.t.n_bc, self.gen)
            b["ic"] = self.wds.sample_ic(self.t.n_ic, self.gen)
        return b

    def estimate_weights(self):
        """Term-level NTK weights: lambda_k = sqrt(mean(trace) / trace_k)."""
        K = self.t.ntk_points
        batches = dict(
            data=self.wds.sample_data(K, self.gen),
            phy=self.wds.sample_physics(K, self.gen),
            bc=self.wds.sample_bc(K, self.gen),
            ic=self.wds.sample_ic(K, self.gen),
        )
        terms = self.loss.terms(self.model, batches)
        params = [p for p in self.model.parameters() if p.requires_grad]
        tr = {}
        for name, val in terms.items():
            grads = torch.autograd.grad(val, params, retain_graph=True, allow_unused=True)
            tr[name] = float(sum(g.detach().pow(2).sum() for g in grads if g is not None))
            self.model.zero_grad(set_to_none=True)
        live = [v for v in tr.values() if v > 0]
        mean_tr = float(np.mean(live)) if live else 1.0
        lo, hi = self.t.ntk_clip
        self.weights = {k: float(np.clip(np.sqrt(mean_tr / max(v, 1e-12)), lo, hi)) for k, v in tr.items()}
        # normalise so that the data terms keep unit scale
        dsum = sum(self.weights[k] for k in self.weights if k.startswith("data"))
        if dsum > 0:
            self.weights = {k: v * len([k for k in self.weights if k.startswith("data")]) / dsum
                            for k, v in self.weights.items()}
        return self.weights

    def physics_factor(self, it: int) -> float:
        if not self.cfg.train.use_physics:
            return 0.0
        w, r = self.t.physics_warmup, max(1, self.t.physics_ramp)
        if it < w:
            return 0.0
        return float(min(1.0, (it - w) / r))

    # ------------------------------------------------------------------ #
    def step(self, it: int):
        pfac = self.physics_factor(it)
        need_phy = pfac > 0.0
        batches = self.sample(n_phy=self.t.n_phy if need_phy else 0)
        if not self.weights:
            self.weights = {k: 1.0 for k in
                            ["phy_T", "phy_c", "phy_gas", "bc_T", "ic_T", "ic_c", "ic_gas",
                             "data_T", "data_c", "data_gas"]}
        terms = self.loss.terms(self.model, batches, use_physics=need_phy,
                                use_bc=need_phy, use_ic=need_phy)

        loss = torch.zeros((), device=self.device)
        detail = {}
        for name, val in terms.items():
            if name.startswith("data"):
                ws = {"data_T": self.t.w_data_T, "data_c": self.t.w_data_c,
                      "data_gas": self.t.w_data_gas}.get(name, 1.0)
                w = self.weights.get(name, 1.0) * self.t.w_data * ws
            elif name.startswith("bc"):
                w = self.weights.get(name, 1.0) * pfac * self.t.w_bc
            elif name.startswith("ic"):
                w = self.weights.get(name, 1.0) * pfac * self.t.w_ic
            else:
                w = self.weights.get(name, 1.0) * pfac * self.t.w_phy
            loss = loss + w * val
            detail[name] = float(val.detach())

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.sched.step()
        return float(loss.detach()), detail

    # ------------------------------------------------------------------ #
    def fit(self):
        t0 = time.time()
        log_path = self.out / "train_log.jsonl"
        with open(log_path, "w", encoding="utf-8") as fh:
            for it in range(self.t.iters):
                if self.t.use_ntk and it % self.t.ntk_every == 0 and it > 0:
                    self.estimate_weights()
                loss, detail = self.step(it)
                if it % self.t.log_every == 0:
                    lr = self.opt.param_groups[0]["lr"]
                    msg = dict(iter=it, loss=round(loss, 6), lr=lr,
                               pfac=round(self.physics_factor(it), 3),
                               **{k: round(v, 6) for k, v in detail.items()})
                    print(f"[{it:6d}] loss={loss:.5f} lr={lr:.1e} " +
                          " ".join(f"{k}={v:.3e}" for k, v in detail.items()), flush=True)
                    fh.write(json.dumps(msg) + "\n")
                    fh.flush()
                if self.t.eval_every and it > 0 and it % self.t.eval_every == 0:
                    self.validate(it)
        self.save(self.out / "ckpt.pt", self.t.iters)
        print(f"[done] {time.time() - t0:.1f}s")

    def validate(self, it: int):
        if self.val_wds is None:
            return None
        mask = self.val_wds.split == "val"
        m = evaluate_split(self.model, self.val_wds, mask, self.cfg, self.device, verbose=True)
        self.history.append(dict(iter=it, **{k: v for k, v in m.items()}))
        (self.out / "val_metrics.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        if m["wape_total"] < self.best:
            self.best = m["wape_total"]
            self.save(self.out / "best.pt", it)
        return m

    def save(self, path, it):
        from dataclasses import asdict
        torch.save(dict(model=self.model.state_dict(), iter=it, cfg=asdict(self.cfg)), str(path))
