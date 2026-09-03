#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte-Carlo evaluation of the deterministic MIP schedule under random inflow.

Q_pump below is the fixed schedule produced by the deterministic MIP
(example_pyomo.py), copied by hand from
rtc-tools/examples/mixed_integer/output/timeseries_export.csv. It is
simulated (not re-optimised) against N_SIMS = 30 independent AR(1)
log-normal inflow trajectories drawn with
ar1_lognormal_inflow.simulate_ar1_lognormal.

The simulation enforces the same constraints as example_pyomo.py /
"stochastic approximation.py":
  * mass balance   A*(H[t]-H[t-1]) = dt*(Q_in[t-1] - Q_pump[t-1] - Q_orifice[t-1])
  * H[0] == H_start
  * 0 <= Q_pump[t] <= PUMP_MAX               (satisfied by construction: the
                                               MIP schedule already respects it)
  * downhill-only gravity orifice, 0 <= Q_orifice[t] <= ORIFICE_MAX, capped
    by w*C*d*sqrt(2*g*(H-H_sea)) and by the water actually available
  * the pump cannot draw more water than is physically in the basin

Unlike "stochastic approximation.py" there is NO emergency weir capping the
level. STORAGE_MAX = 0.5 m is therefore a genuine hard constraint
(H_storage <= STORAGE_MAX in example_pyomo.py) that this fixed,
non-recourse schedule can actually violate once the inflow no longer
matches the deterministic forecast it was optimised for. A scenario where
the level ever exceeds STORAGE_MAX is charged a FLAT penalty of
PENALTY = 1e6, once, regardless of how many hours it stays above the bound.

This is a one-off event penalty, not a per-hour one, on purpose: the 07-07
SPSA script originally summed (H_t - 0.5)+ every hour it persisted, which
was later identified as a bug (see MEMORY.md, "penalty double-counting") -
it inflated long floods disproportionately and let the optimiser game flood
timing. There is no optimiser here (the schedule is fixed, not adapted to
the penalty), so that gaming risk does not apply, but the flat penalty is
kept for consistency with the rest of the thesis's constraint-violation
convention ("infeasible under this scenario" = one failure event). Hours
violated per scenario are still reported as a diagnostic, just not used to
scale the penalty.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ar1_lognormal_inflow import simulate_ar1_lognormal, PHI, SIGMA_EPS, MU_Y

SEED = 42
N_SIMS = 3000

# ------------------------------------------------------------------
# 1. Physical constants (example_pyomo.py / stochastic approximation.py)
# ------------------------------------------------------------------
BASE = '/Users/martijnkrikke/Documents/Scriptie/Scriptie_martijn'
time_series = pd.read_csv(f'{BASE}/timeseries_import.csv')

T = 21
H_sea = time_series["H_sea"].to_numpy()[:T]

STORAGE_MAX = 0.5      # m       hard upper bound on H_storage
H_start = 0.4          # m       H_storage[1] == H_start
A = 1e6                # m^2     basin surface area
dt = 3600.0            # s       time step
PUMP_MAX = 7.0         # m3/s    bound on Q_pump
ORIFICE_MAX = 10.0     # m3/s    bound on Q_orifice

w = 3.0   # m       width of orifice
d = 0.8   # m       height of orifice
C = 1.0   # none    orifice constant
g = 9.8   # m/s^2   gravitational acceleration

PENALTY = 1e6   # charged once per scenario if H_storage > STORAGE_MAX at any hour

# ------------------------------------------------------------------
# 2. Fixed pumping schedule, manually copied from
#    rtc-tools/examples/mixed_integer/output/timeseries_export.csv (Q_pump)
# ------------------------------------------------------------------
Q_PUMP = np.array([
    0.000000, 0.000000, 0.000000, 6.118018, 3.388742, 2.823449, 3.015661,
    2.871256, 2.698051, 2.501853, 2.107379, 1.663642, 1.420790, 1.283716,
    1.181330, 0.795502, 1.640000, 0.248242, 0.000000, 0.000000, 0.000000,
])
assert Q_PUMP.shape == (T,)


# ------------------------------------------------------------------
# 3. Simulation of the fixed schedule under one inflow realisation
# ------------------------------------------------------------------
def simulate_storage(q_pump, q_in):
    """Simulate storage/orifice/pump under a fixed schedule, no weir.

    Returns (H, q_pmp, q_ori). H can exceed STORAGE_MAX - that is exactly
    the event penalised_cost() penalises.
    """
    H = np.empty(T)
    q_ori = np.empty(T)
    q_pmp = np.empty(T)

    level = H_start
    for t in range(T):
        if t > 0:
            level = H[t - 1] + dt / A * (q_in[t - 1] - q_pmp[t - 1] - q_ori[t - 1])
        H[t] = level

        # the pump cannot lift water that is not in the basin
        q_pmp[t] = min(q_pump[t], A / dt * H[t] + q_in[t])

        head = H[t] - H_sea[t]
        if head > 0.0:
            available = A / dt * H[t] + q_in[t] - q_pmp[t]
            q_ori[t] = min(w * C * d * np.sqrt(2.0 * g * head),
                           ORIFICE_MAX,
                           max(available, 0.0))
        else:
            q_ori[t] = 0.0

    return H, q_pmp, q_ori


def penalised_cost(H, q_pump):
    """Pumped volume (fixed schedule, m3) + flat PENALTY if STORAGE_MAX is ever broken.

    One-off event penalty (not scaled by how many hours it persists) - see
    the module docstring for why.
    """
    pumped_volume = np.sum(q_pump) * dt
    n_violations = int(np.sum(H > STORAGE_MAX))
    penalty = PENALTY if n_violations > 0 else 0.0
    return pumped_volume + penalty, n_violations


# ------------------------------------------------------------------
# 4. Run N_SIMS = 30 simulations
# ------------------------------------------------------------------
if __name__ == '__main__':
    q_in_scenarios = simulate_ar1_lognormal(N_SIMS, T, PHI, MU_Y, SIGMA_EPS, seed=SEED)

    H_all = np.empty((N_SIMS, T))
    records = []
    for i, q_in in enumerate(q_in_scenarios):
        H, q_pmp, q_ori = simulate_storage(Q_PUMP, q_in)
        cost, n_violations = penalised_cost(H, Q_PUMP)
        H_all[i] = H
        records.append({
            'sim': i,
            'max H (m)': H.max(),
            'hours > 0.5m': n_violations,
            'violated': n_violations > 0,
            'cost incl. penalty (m3)': cost,
        })

    summary = pd.DataFrame(records)
    print(summary.to_string(index=False))
    print()
    print(f"P(0.5m constraint broken)        = {summary['violated'].mean():.3f}")
    if summary['violated'].any():
        print(f"E[hours violated | broken]        = "
              f"{summary.loc[summary['violated'], 'hours > 0.5m'].mean():.2f}")
    print(f"E[total cost incl. penalty] (m3) = {summary['cost incl. penalty (m3)'].mean():.3e}")
    print(f"pumped volume alone (m3)         = {(Q_PUMP.sum() * dt):.1f}")

    # ------------------------------------------------------------------
    # 5. Plot the 30 storage trajectories against the 0.5 m bound
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    hours = np.arange(T)
    for i in range(N_SIMS):
        broken = summary.loc[i, 'violated']
        ax.plot(hours, H_all[i], color='#af1b1b' if broken else '#2a78d6',
                lw=0.9, alpha=0.7 if broken else 0.35)
    ax.axhline(STORAGE_MAX, color='#0b0b0b', lw=1.2, ls='--',
               label=f'STORAGE_MAX = {STORAGE_MAX} m')
    ax.set_xlabel('time step t (h)')
    ax.set_ylabel('H_storage (m)')
    ax.set_title(f'{N_SIMS} Monte-Carlo simulations of the fixed MIP schedule\n'
                 f'(red = 0.5m constraint broken, flat penalty = {PENALTY:.0e})')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = f'{BASE}/fixed_schedule_montecarlo.png'
    fig.savefig(out_path, dpi=150)
    plt.show()
