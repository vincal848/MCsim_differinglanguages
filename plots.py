"""Figures for docs/img. Separate from pricing, like black_scholes.py and mc.py --
pricing returns numbers, plotting is a separate call that nothing else depends on.

    python plots.py

regenerates docs/img/paths_fan.png, docs/img/terminal_hist.png and
docs/img/convergence.png.
"""

import math
import os

import numpy as np

import mc

DOCS_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "img")


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plotting needs matplotlib: pip install matplotlib")
    return plt


def paths_fan(S0=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=252, n_paths=30, seed=7, ax=None):
    """A handful of simulated paths, to look at rather than to price from."""
    plt = _plt()
    paths = mc.simulate_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    t = np.linspace(0, T, n_steps + 1)

    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.plot(t, paths.T, lw=0.8, alpha=0.7)
    ax.set(xlabel="time (years)", ylabel="price",
           title="%d simulated risk-neutral GBM paths, S0=%s r=%s sigma=%s"
                 % (n_paths, S0, r, sigma))
    return ax


def terminal_hist(S0=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=252, n_paths=20000,
                  seed=123, ax=None):
    """Terminal price histogram against the exact lognormal density.

    S_T is lognormal by construction under GBM -- this checks that the simulator
    reproduces that distribution, which is a different (and weaker) claim than
    checking the option price against Black-Scholes.
    """
    plt = _plt()
    ST = mc.simulate_terminal(S0, r, sigma, T, n_steps, n_paths, seed)

    mu_log = math.log(S0) + (r - 0.5 * sigma * sigma) * T
    sd_log = sigma * math.sqrt(T)
    x = np.linspace(ST.min(), ST.max(), 400)
    density = (1.0 / (x * sd_log * math.sqrt(2 * math.pi))
              * np.exp(-(np.log(x) - mu_log) ** 2 / (2 * sd_log ** 2)))

    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.hist(ST, bins=80, density=True, alpha=0.6, label="simulated $S_T$")
    ax.plot(x, density, lw=2, label="exact lognormal density")
    ax.set(xlabel="$S_T$", ylabel="density",
           title="Terminal price distribution, n_paths=%d" % n_paths)
    ax.legend()
    return ax


def convergence(S0=100.0, K=100.0, r=0.03, sigma=0.2, T=1.0, n_steps=252,
                ns=(100, 300, 1000, 3000, 10000, 30000), seed=123, ax=None):
    """Standard error against number of paths, against the 1/sqrt(N) line it
    should follow -- the standard Monte Carlo convergence check.
    """
    plt = _plt()
    se = [mc.price_european(S0, [K], r, sigma, T, n_steps, n, seed)[0]["call_se"]
         for n in ns]

    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    ax.loglog(ns, se, "o-", label="Monte Carlo standard error")
    ref = se[0] * np.sqrt(ns[0] / np.asarray(ns, dtype=float))
    ax.loglog(ns, ref, "--", color="0.5", label=r"$1/\sqrt{N}$")
    ax.set(xlabel="n_paths", ylabel="standard error",
           title="SE of the K=%s call vs n_paths" % K)
    ax.legend()
    return ax


def main():
    plt = _plt()
    os.makedirs(DOCS_IMG, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    paths_fan(ax=ax)
    fig.savefig(os.path.join(DOCS_IMG, "paths_fan.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    terminal_hist(ax=ax)
    fig.savefig(os.path.join(DOCS_IMG, "terminal_hist.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    convergence(ax=ax)
    fig.savefig(os.path.join(DOCS_IMG, "convergence.png"), dpi=150)


if __name__ == "__main__":
    main()
