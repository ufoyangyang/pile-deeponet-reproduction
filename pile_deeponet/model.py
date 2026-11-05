"""PILE-DeepONet: physics-informed latent-ensemble DeepONet (paper Sec. 2.2).

Two stacked DeepONet-with-encoders operators:

* latent-feature operator  G_F  : (u1, u2, u3) x (x, d)  -> (T_F, c_F, Gas_F)
* prediction operator      G_P  : (u1, u2, u3, u4) x (x, [d, tau]) -> (T, c, Gas)

with the "separated DeepONet with encoders" architecture of Appendix A:
every branch / trunk is an encoder MLP whose hidden state is gated between the
encoder embedding and the layer output, and whose output is a q-dimensional
vector; the final prediction is the sum over the latent index k of the
element-wise product of all branch and trunk outputs.

The q latent channels are split into three thirds, so one operator predicts
the three physical fields at once (Eq. 10-11).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class Stan(nn.Module):
    """Trainable-slope tanh activation, Eq. 21-22 of the paper.

        phi(x) = (1 + beta * x) * tanh(x)      phi'(x) = (1 + beta*x)*(1-tanh^2) + beta*tanh
    """

    def __init__(self, n_features: int, beta0: float = 0.1):
        super().__init__()
        self.beta = nn.Parameter(torch.full((n_features,), float(beta0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(x)
        return (1.0 + self.beta * x) * t


class EncoderMLP(nn.Module):
    """One branch / trunk network (Table 3).

    input layer -> `depth` gated hidden layers -> linear output layer, where the
    gating follows the skip-encoder of Appendix A:

        E   = phi(W_e u + b_e)                (encoder embedding)
        Z_l = (1 - H_l) * E + H_l * phi(W_l Z_{l-1} + b_l),  H_l = sigmoid(...)

    and the output layer is multiplied by a sigmoid so that every component of
    the basis functions lies in (0, 1) (paper: "output layer with sigmoid").
    """

    def __init__(self, in_dim: int, mcfg: ModelConfig):
        super().__init__()
        self.mcfg = mcfg
        h, d, q = mcfg.hidden, mcfg.depth, mcfg.q
        self.enc = nn.Linear(in_dim, h)
        make_act = (lambda: Stan(h)) if mcfg.stan else (lambda: nn.Tanh())
        self.act_in = make_act()
        self.gates = nn.ModuleList([nn.Linear(h, h) for _ in range(d)])
        self.mix = nn.ModuleList([nn.Linear(h, h) for _ in range(d)])
        self.acts = nn.ModuleList([make_act() for _ in range(d)])
        self.out = nn.Linear(h, q)
        for lin in [self.enc, *self.gates, *self.mix, self.out]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        e = self.act_in(self.enc(u))
        z = e
        for gate, mix, act in zip(self.gates, self.mix, self.acts):
            g = torch.sigmoid(gate(z))
            z = (1.0 - g) * e + g * act(mix(z))
        o = self.out(z)
        act = self.mcfg.out_act
        if act == "sigmoid":
            return torch.sigmoid(o)
        if act == "tanh":
            return torch.tanh(o)
        return o


class PILEOperator(nn.Module):
    """A DeepONet whose q channels are split into three physical fields."""

    def __init__(self, branch_dims, trunk_dims, mcfg: ModelConfig, separated_trunk=True):
        super().__init__()
        self.mcfg = mcfg
        self.branches = nn.ModuleList([EncoderMLP(d, mcfg) for d in branch_dims])
        self.trunks = nn.ModuleList([EncoderMLP(d, mcfg) for d in trunk_dims])
        self.separated_trunk = separated_trunk
        self.combine = getattr(mcfg, "combine", "product")
        q3 = mcfg.q // 3
        self.q3 = q3

    def forward(self, branch_inputs, trunk_inputs):
        """branch_inputs / trunk_inputs: lists of (N, dim) tensors."""
        prod = self.branches[0](branch_inputs[0])
        for net, u in zip(self.branches[1:], branch_inputs[1:]):
            other = net(u)
            prod = prod * other if self.combine == "product" else prod + other
        tp = self.trunks[0](trunk_inputs[0])
        for net, u in zip(self.trunks[1:], trunk_inputs[1:]):
            tp = tp * net(u)
        v = prod * tp
        q3 = self.q3
        return v[:, :q3], v[:, q3:2 * q3], v[:, 2 * q3:3 * q3]


# --------------------------------------------------------------------------- #
# the two operators
# --------------------------------------------------------------------------- #
class PILEDeepONet(nn.Module):
    def __init__(self, mcfg: ModelConfig):
        super().__init__()
        self.m = mcfg
        sep = mcfg.separated_trunk
        self.latent = PILEOperator(
            branch_dims=[1, mcfg.history_len, mcfg.short_len],
            trunk_dims=[1, 1] if sep else [2],
            mcfg=mcfg,
            separated_trunk=sep,
        )
        nb4 = mcfg.n_latent_features if mcfg.use_latent_operator else 0
        self.pred = PILEOperator(
            branch_dims=[1, mcfg.history_len, mcfg.short_len] + ([nb4] if nb4 else []),
            trunk_dims=[1, 2] if sep else [3],
            mcfg=mcfg,
            separated_trunk=sep,
        )
        self.register_buffer(
            "anchor_d", torch.tensor(mcfg.latent_anchors, dtype=torch.float32) / mcfg.lead_time,
            persistent=False,
        )
        # learnable output biases absorb the offset of the sigmoid parameterisation
        self.b_T = nn.Parameter(torch.zeros(1))
        self.b_c = nn.Parameter(torch.full((1,), -2.0))
        self.b_g = nn.Parameter(torch.full((1,), -2.0))
        self.gain = 1.0 / (mcfg.q // 3) ** 0.5

    # -- latent features --------------------------------------------------- #
    def _trunk_inputs(self, x, d, tau):
        if self.m.separated_trunk:
            return [x, torch.cat([d, tau], dim=1)]
        return [torch.cat([x, d, tau], dim=1)]

    def latent_features(self, u1, u2, u3, x, d):
        if self.m.separated_trunk:
            trunks = [x, d]
        else:
            trunks = [torch.cat([x, d], dim=1)]
        return self.latent([u1, u2, u3], trunks)

    def u4(self, u1, u2, u3, x):
        """p latent scalars: latent T basis evaluated at the anchor lead times."""
        N = x.shape[0]
        outs = []
        for dj in self.anchor_d:
            d = dj.expand(N, 1)
            a, _, _ = self.latent_features(u1, u2, u3, x, d)
            outs.append(a.mean(dim=1, keepdim=True))
        return torch.cat(outs, dim=1)

    # -- forward ----------------------------------------------------------- #
    def forward(self, u1, u2, u3, x, d):
        """Predict (T, c, Gas) at normalised coordinates (x, d).

        x : (N,1) normalised radius      d : (N,1) normalised lead time
        Returns physical values: T [degC], c [-], Gas [kg].
        """
        m = self.m
        tau = torch.remainder(d * m.lead_time, m.local_window) / m.local_window
        branches = [u1, u2, u3]
        if m.use_latent_operator:
            branches = branches + [self.u4(u1, u2, u3, x)]
        tT, tc, tg = self.pred(branches, self._trunk_inputs(x, d, tau))

        if m.reduce == "sum":
            red = lambda v: v.sum(dim=1, keepdim=True) * self.gain
        else:
            red = lambda v: v.mean(dim=1, keepdim=True)
        T = m.T_min + (m.T_max - m.T_min) * torch.sigmoid(red(tT) + self.b_T)
        # c and Gas are almost binary (1 -> 0 and 0 -> Gas0 at thermal runaway).
        # They are parameterised through the *reaction extent* xi = 1 - c so that
        # the head starts inside the informative part of the sigmoid instead of
        # being parked on its saturated plateau.
        xi_c = torch.sigmoid(red(tc) + self.b_c)
        xi_g = torch.sigmoid(red(tg) + self.b_g)
        c = (1.0 - xi_c).clamp(0.0, 1.0)
        gas = (m.gas_ref * xi_g).clamp_min(0.0)
        return T, c, gas

    # -- convenience ------------------------------------------------------- #
    def predict_field(self, u1, u2, u3, x, d):
        """Return T, c, Gas plus the raw latent fields (for the losses)."""
        T, c, gas = self.forward(u1, u2, u3, x, d)
        return dict(T=T, c=c, gas=gas)


def build_model(mcfg: ModelConfig, device="cpu") -> PILEDeepONet:
    model = PILEDeepONet(mcfg).to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
