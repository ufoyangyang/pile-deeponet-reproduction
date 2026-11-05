"""Configuration objects for the PILE-DeepONet reproduction.

All physical constants come from Table B1 of

    J. Jeong, H. Kwak, J. Kim, S. Oh, "Physics-informed latent ensemble DeepONet
    with synthetic data for real-time thermal runaway prognosis of a lithium-ion
    battery", eTransportation 28 (2026) 100575.

The COMSOL 6.1 reference model of the paper is replaced here by an in-house
1-D radial finite-volume solver (see `physics.py`); it plays exactly the same
role: it produces the "synthetic data" that supervise the networks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Physics / data generation
# --------------------------------------------------------------------------- #
@dataclass
class PhysicsConfig:
    """Lumped (equivalent) LFP 4680 cell parameters -- paper Table B1."""

    # thermal
    rho: float = 2790.0          # kg/m^3
    cp: float = 990.0            # J/(kg K)
    k: float = 0.7               # W/(m K), radial

    # Arrhenius decomposition of the equivalent chemical component
    # NOTE: A and H were re-calibrated (A=2e7, H=2.6e5) so that the in-house
    # solver reproduces the reference run of the paper (peak 412.7 degC @ 2647 s
    # for a 2600 s ramp and h_cool = 7).  The paper reports calibrated values
    # A = 3e7, H = 1.6e5 for its COMSOL model.
    E: float = 99000.0           # J/mol
    A: float = 2.0e7             # 1/s
    H: float = 2.6e5             # J/kg
    W: float = 2291.3            # kg/m^3
    Rc: float = 8.314            # J/(mol K)  (printed as J kg^-1 K^-1 in Table B1)

    # gas / pressure
    Rg: float = 189.0            # J/(kg K)
    Gas0: float = 8.8e-3         # kg  (8.8 g in Table B1)

    # geometry
    radius: float = 0.023        # m   (46 mm diameter)
    height: float = 0.080        # m
    canister_volume: float = 3.95e-3   # m^3, sealed canister free volume

    # operating conditions
    T_init: float = 35.0         # degC
    T_target: float = 270.0      # degC
    T_cool: float = 35.0         # degC
    h_heater: float = 200.0      # W/(m^2 K)  -- not tabulated in the paper, calibrated
    c_off: float = 0.1           # heater shut-off threshold on c

    # numerics
    n_r: int = 31                # radial nodes (centre -> surface)
    dt: float = 0.05             # s, explicit integration step
    t_end: float = 5400.0        # s
    save_dt: float = 2.0         # s, storage interval

    @property
    def free_volume(self) -> float:
        """Gas free volume of the sealed canister."""
        v_cell = 3.141592653589793 * self.radius ** 2 * self.height
        return self.canister_volume - v_cell

    @property
    def dr(self) -> float:
        return self.radius / (self.n_r - 1)


# --------------------------------------------------------------------------- #
# Dataset split (paper Table 2)
# --------------------------------------------------------------------------- #
@dataclass
class SplitConfig:
    ramps_train: tuple = tuple(range(2100, 3001, 45))     # 2100:45:3000
    h_cool: tuple = (7.0, 10.0, 13.0, 16.0, 19.0)         # W/(m^2 K)
    ramps_ood: tuple = tuple(range(3100, 3301, 40))       # 3100:40:3300
    h_ood: tuple = (7.0, 13.0, 19.0)
    exp_condition: tuple = (2600.0, 7.0)                  # test-set 2
    n_train: int = 84
    n_val: int = 10
    n_test1: int = 11
    seed: int = 0


# --------------------------------------------------------------------------- #
# Network / training
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    q: int = 300                 # latent width of every branch / trunk net
    reduce: str = "sum"          # latent reduction: "sum" or "mean"
    hidden: int = 100
    depth: int = 4
    n_latent_features: int = 4   # p, latent features fed to the prediction operator
    history_len: int = 31        # v, historical samples (30 s apart)
    short_len: int = 11          # m, short-term samples (30 s apart)
    sample_dt: float = 30.0      # s
    t_threshold: float = 120.0   # degC, u2 anchor
    lead_time: float = 1200.0    # s, d0
    latent_anchors: tuple = (900.0, 990.0, 1110.0, 1200.0)
    local_window: float = 300.0  # s, local temporal domain length L
    t_pred: float = 1500.0       # s, default prediction start (experiment)

    # architecture switches (paper ablations)
    stan: bool = True
    separated_trunk: bool = True
    use_latent_operator: bool = True
    # activation of the branch / trunk output layer. The paper (Table 3) uses a
    # sigmoid, but in this implementation the product of more than two
    # (0, 1)-valued factors collapses the basis to a constant field (see
    # docs/implementation_notes.md); a linear output is used by default.
    out_act: str = "linear"      # "linear" | "sigmoid" | "tanh"
    # how the multi-input branch nets are merged: the paper multiplies them
    # ("product"); a product of four small factors has a vanishing gradient for
    # every individual branch, so the additive ensemble ("sum") is the default.
    combine: str = "sum"         # "sum" | "product"

    # output scaling
    T_min: float = 20.0
    T_max: float = 520.0
    gas_ref: float = 8.8e-3      # kg, saturation of the gas output


@dataclass
class TrainConfig:
    iters: int = 20000
    lr: float = 1e-3
    lr_decay_every: int = 2000
    lr_decay: float = 0.1
    n_phy: int = 128
    n_data: int = 256
    n_bc: int = 64
    n_ic: int = 64
    physics_warmup: int = 800      # iterations with physics off
    physics_ramp: int = 800        # ramp physics weight in
    w_phy: float = 0.1
    w_data: float = 1.0
    w_data_T: float = 1.0
    w_data_c: float = 1.0
    w_data_gas: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0
    use_ntk: bool = True
    ntk_every: int = 1000
    ntk_points: int = 32           # subset used to estimate the NTK diagonal
    ntk_clip: tuple = (0.05, 20.0)
    use_physics: bool = True
    log_every: int = 250
    # >1 concentrates the collocation points near the end of the window, where
    # the runaway transient is; 1.0 reproduces the paper's uniform sampling
    d_power: float = 1.0
    eval_every: int = 1000
    seed: int = 0
    out_dir: str = "runs/base"


@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def config_from_dict(raw: dict) -> Config:
    """Rebuild a Config from a (nested) plain dict, e.g. a checkpoint."""
    cfg = Config()
    for k, v in raw.items():
        if not hasattr(cfg, k):
            continue
        section = getattr(cfg, k)
        for kk, vv in v.items():
            if hasattr(section, kk):
                if isinstance(vv, list):
                    vv = tuple(vv)
                setattr(section, kk, vv)
    return cfg


def load_config(path) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = Config()
    for k, v in raw.items():
        section = getattr(cfg, k)
        for kk, vv in v.items():
            if hasattr(section, kk):
                if isinstance(vv, list):
                    vv = tuple(vv)
                setattr(section, kk, vv)
    return cfg
