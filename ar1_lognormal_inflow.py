#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR(1) recursion on the log scale for the inflow Q_t:

    Y_t = mu_Y + phi * (Y_{t-1} - mu_Y) + eps_t,   eps_t ~ N(0, sigma_eps^2)
    Q_t = exp(Y_t)

Q_t is then log-normal with sigma_Y^2 = sigma_eps^2 / (1 - phi^2), and
mu_Y is chosen so that E[Q_t] = Q_MEAN exactly:

    mu_Y = ln(Q_MEAN) - 0.5 * sigma_Y^2

Simulates N_SIMS independent trajectories of length T and plots them.
"""

import numpy as np
import matplotlib.pyplot as plt

SEED = 42
T = 24            # number of time steps
N_SIMS = 100        # number of simulated trajectories

PHI = 0.8           # AR(1) persistence
SIGMA_EPS = 0.15     # innovation std on the log scale
Q_MEAN = 1.0        # target E[Q_t]

SIGMA_Y2 = SIGMA_EPS ** 2 / (1.0 - PHI ** 2)
MU_Y = np.log(Q_MEAN) - 0.5 * SIGMA_Y2


def simulate_ar1_lognormal(n_sims, T, phi, mu_y, sigma_eps, seed=SEED):
    """Return an (n_sims, T) array of Q_t trajectories."""
    rng = np.random.default_rng(seed)
    Y = np.empty((n_sims, T))
    # start each trajectory at its stationary mean
    Y[:, 0] = mu_y
    eps = rng.normal(0.0, sigma_eps, size=(n_sims, T))
    for t in range(1, T):
        Y[:, t] = mu_y + phi * (Y[:, t - 1] - mu_y) + eps[:, t]
    return np.exp(Y)


if __name__ == '__main__':
    phi_values = [0.7, 0.8, 0.9]
    time = np.arange(T)

    fig, axes = plt.subplots(1, len(phi_values), figsize=(15, 5),
                              sharey=True)

    for ax, phi in zip(axes, phi_values):
        sigma_y2 = SIGMA_EPS ** 2 / (1.0 - phi ** 2)
        mu_y = np.log(Q_MEAN) - 0.5 * sigma_y2
        Q = simulate_ar1_lognormal(N_SIMS, T, phi, mu_y, SIGMA_EPS)

        for i in range(N_SIMS):
            ax.plot(time, Q[i], color='#2a78d6', lw=0.6, alpha=0.15)
        ax.plot(time, Q.mean(axis=0), color='#eda100', lw=2,
                 label='sample mean across trajectories')
        ax.axhline(Q_MEAN, color='#0b0b0b', lw=1, ls='--',
                   label=f'target E[Q] = {Q_MEAN}')

        ax.set_xlabel('time step t')
        ax.set_title(f'phi = {phi}')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r'$Q_t$')
#    axes[0].set_ylim(top=4)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f'{N_SIMS} simulated AR(1) log-normal trajectories '
                 f'(sigma_eps={SIGMA_EPS})')

    fig.tight_layout()
    out_path = '/Users/martijnkrikke/Documents/Scriptie/Scriptie_martijn/ar1_lognormal_simulation.png'
    fig.savefig(out_path, dpi=150)
    plt.show()
