"""1-D radial finite-volume model of thermal runaway.

Stand-in for the COMSOL 6.1 multiphysics model of the paper (Section 3.1).
Governing equations (paper Eq. 1-6 and 30):

    eta * rho * cp * dT/dt = div(k grad T) + H * W * R
    R                      = -dc/dt = A exp(-Ea/(Rc T)) c
    Gas(t)                 = Gas0 (1 - exp(-int R dt))
    P                      = Gas * Rg * T / V
    k dT/dn |surface       = S(c) h_heater (T_heater - T) + h_cool (T_cool - T)
    S(c)                   = 1 if c >= 0.1 else 0

The heater temperature T_heater is prescribed as a linear ramp
T_init -> T_target over `t_ramp` seconds (paper Fig. 4a, Table 2), after which
it is held at T_target; the heater is switched off by S(c) once c < 0.1.
"""

from __future__ import annotations

import numpy as np

from .config import PhysicsConfig


def _grid(cfg: PhysicsConfig):
    nr, R = cfg.n_r, cfg.radius
    dr = R / (nr - 1)
    r = np.arange(nr) * dr
    r_face = np.minimum((np.arange(nr) + 0.5) * dr, R)      # r_{i+1/2}
    r_face_in = np.maximum((np.arange(nr) - 0.5) * dr, 0.0)  # r_{i-1/2}
    vol = np.pi * (r_face ** 2 - r_face_in ** 2)             # per unit height
    a_mid = 2.0 * np.pi * r_face[:-1]                        # internal faces
    a_surf = 2.0 * np.pi * R
    return dr, r, vol, a_mid, a_surf


def simulate(
    cfg: PhysicsConfig,
    h_cool,
    t_ramp,
    t_end: float | None = None,
    dt: float | None = None,
    save_dt: float | None = None,
    progress: bool = False,
):
    """Vectorised simulation over a batch of thermal-abuse conditions.

    Parameters
    ----------
    h_cool : array_like (B,)   effective cooling coefficient [W/(m^2 K)]
    t_ramp : array_like (B,)   heater ramp time to 270 degC [s]

    Returns
    -------
    dict with
        t      (n_t,)          time grid [s]
        T      (B, n_t, nr)    temperature [degC]
        c      (B, n_t)        equivalent chemical component [-]
        gas    (B, n_t)        gas mass [kg]
        p      (B, n_t)        pressure [Pa]
        t_thr  (B,)            time at which the surface reaches 120 degC
        peak_T (B,)            max mean temperature [degC]
        peak_t (B,)            time of the max mean temperature [s]
    """
    h_cool = np.atleast_1d(np.asarray(h_cool, dtype=np.float64))
    t_ramp = np.atleast_1d(np.asarray(t_ramp, dtype=np.float64))
    B = h_cool.size
    t_end = float(cfg.t_end if t_end is None else t_end)
    dt = float(cfg.dt if dt is None else dt)
    save_dt = float(cfg.save_dt if save_dt is None else save_dt)

    dr, r, vol, a_mid, a_surf = _grid(cfg)
    rho_cp = cfg.rho * cfg.cp
    g_face = cfg.k * a_mid / dr                       # (nr-1,)
    g_node = 1.0 / (rho_cp * vol)                     # (nr,)
    g_surf = a_surf / (rho_cp * vol[-1])
    hw = cfg.H * cfg.W
    kelvin = 273.15

    rng = np.random.default_rng(0)

    T = np.full((B, cfg.n_r), cfg.T_init, dtype=np.float64)
    c = np.ones(B)
    gas = np.zeros(B)
    T_k = T + kelvin

    n_steps = int(round(t_end / dt))
    save_every = max(1, int(round(save_dt / dt)))
    n_save = n_steps // save_every + 1

    T_hist = np.empty((B, n_save, cfg.n_r), dtype=np.float32)
    c_hist = np.empty((B, n_save), dtype=np.float32)
    gas_hist = np.empty((B, n_save), dtype=np.float32)
    t_hist = np.empty(n_save, dtype=np.float64)
    t_thr = np.full(B, np.nan)

    si = 0
    T_hist[:, si] = T
    c_hist[:, si] = c
    gas_hist[:, si] = gas
    t_hist[si] = 0.0

    for step in range(1, n_steps + 1):
        t = step * dt
        # --- external heating / cooling (Eq. 30) ---------------------------
        T_heater = cfg.T_init + (cfg.T_target - cfg.T_init) * np.minimum(t / t_ramp, 1.0)
        s = (c >= cfg.c_off).astype(np.float64)
        h_tot = s * cfg.h_heater + h_cool
        T_ext = (s * cfg.h_heater * T_heater + h_cool * cfg.T_cool) / np.maximum(h_tot, 1e-12)

        # --- reaction (Eq. 3) ---------------------------------------------
        # the equivalent chemical component is a lumped (per-cell) variable,
        # driven by the mean cell temperature
        T_mean_now = T.mean(axis=1)
        rate = cfg.A * np.exp(-cfg.E / (cfg.Rc * (T_mean_now + kelvin)))
        q = hw * rate[:, None] * c[:, None]

        # --- heat conduction + source -------------------------------------
        flux = g_face * (T[:, :-1] - T[:, 1:])           # (B, nr-1), outward +
        dT = np.zeros_like(T)
        dT[:, 1:] += flux * g_node[1:]
        dT[:, :-1] -= flux * g_node[:-1]
        dT[:, -1] -= h_tot * (T[:, -1] - T_ext) * g_surf
        dT += q / rho_cp
        T = T + dt * dT

        # --- chemistry (exact integration of the linear ODE) --------------
        decay = np.exp(-rate * dt)
        c = c * decay
        gas = cfg.Gas0 - (cfg.Gas0 - gas) * decay

        if step % save_every == 0:
            si += 1
            T_hist[:, si] = T
            c_hist[:, si] = c
            gas_hist[:, si] = gas
            t_hist[si] = t
            surf = T[:, -1]
            newly = np.isnan(t_thr) & (surf >= 120.0)
            t_thr[newly] = t
        if progress and step % (n_steps // 10) == 0:
            print(f"  step {step}/{n_steps}")

    T_hist = T_hist[:, : si + 1]
    c_hist = c_hist[:, : si + 1]
    gas_hist = gas_hist[:, : si + 1]
    t_hist = t_hist[: si + 1]

    # pressure of the sealed canister (Eq. 6), using the mean cell temperature
    T_mean = T_hist.mean(axis=2)
    p_hist = gas_hist * cfg.Rg * (T_mean + kelvin) / cfg.free_volume

    peak_idx = T_mean.argmax(axis=1)
    peak_T = T_mean[np.arange(B), peak_idx]
    peak_t = t_hist[peak_idx]

    return dict(
        t=t_hist.astype(np.float32),
        T=T_hist,
        c=c_hist,
        gas=gas_hist,
        p=p_hist.astype(np.float32),
        t_thr=t_thr,
        peak_T=peak_T,
        peak_t=peak_t,
    )


def simulate_one(cfg: PhysicsConfig, h_cool: float, t_ramp: float, **kw):
    out = simulate(cfg, [h_cool], [t_ramp], **kw)
    return {k: (v[0] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1 else v)
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
# command line: python -m pile_deeponet.physics --ramp 2600 --h-cool 7
# --------------------------------------------------------------------------- #
def _main():
    import argparse

    from .config import Config

    ap = argparse.ArgumentParser(description="Run the 1-D radial thermal-runaway model.")
    ap.add_argument("--ramp", type=float, default=2600.0, help="ramp-up time to T_target (s)")
    ap.add_argument("--h-cool", type=float, default=7.0, help="cooling coefficient W/(m2 K)")
    ap.add_argument("--save-dt", type=float, default=1.0)
    ap.add_argument("--csv", default=None, help="optional output csv path")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.config:
        from .config import load_config
        cfg = load_config(args.config)

    out = simulate_one(cfg.physics, args.h_cool, args.ramp, save_dt=args.save_dt)
    T, t = out["T"], out["t"]
    T_mean = T.mean(axis=1)
    ip = int(np.argmax(T_mean))
    print(f"cells            : {cfg.physics.n_r}")
    print(f"peak temperature : {T_mean[ip]:.2f} C at t={t[ip]:.0f}s "
          f"(surface {T[ip, -1]:.2f} C, centre {T[ip, 0]:.2f} C)")
    print(f"peak pressure    : {out['p'].max():.2f} kPa")
    print(f"final c / gas    : {out['c'][-1]:.4f} / {out['gas'][-1] * 1e3:.3f} g")
    print(f"threshold t_thr  : {out['t_thr'].item():.0f}s")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "T_mean", "T_centre", "T_surface", "c", "gas_g", "p_kPa"])
            for k in range(len(t)):
                w.writerow([f"{t[k]:.1f}", f"{T_mean[k]:.4f}", f"{T[k, 0]:.4f}",
                            f"{T[k, -1]:.4f}", f"{out['c'][k]:.6f}",
                            f"{out['gas'][k] * 1e3:.6f}", f"{out['p'][k]:.4f}"])
        print(f"csv -> {args.csv}")
    return out


if __name__ == "__main__":
    _main()
