"""How volatility and time horizon change option prices and outcome
distributions."""
from __future__ import annotations

import numpy as np

from .analytics import distribution_stats
from .paths import simulate_gbm_terminal
from .pricing import black_scholes, mc_european

DEFAULT_VOLS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
DEFAULT_MATURITIES = (1 / 12, 0.25, 0.5, 1.0, 2.0, 5.0)


def option_price_grid(S0, K, r, vols=DEFAULT_VOLS, maturities=DEFAULT_MATURITIES,
                      kind="call", n_paths=200_000, seed=1):
    """Monte Carlo price (control variate) and Black-Scholes price for every
    (volatility, maturity) pair."""
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in vols:
        for T in maturities:
            res = mc_european(S0, K, T, r, sigma, kind=kind, n_paths=n_paths,
                              method="control", seed=rng)
            bs = black_scholes(S0, K, T, r, sigma, kind=kind)
            rows.append({
                "sigma": sigma, "T": T, "mc_price": res.price,
                "ci_low": res.ci_low, "ci_high": res.ci_high,
                "bs_price": bs, "pct_of_spot": res.price / S0,
                "ci_contains_bs": res.contains(bs),
            })
    return rows


def horizon_risk(S0, mu, vols=(0.12, 0.18, 0.30),
                 horizons=(0.25, 0.5, 1, 2, 3, 5, 10), n_paths=200_000, seed=2):
    """Real-world outcome distribution for each (volatility, horizon) pair:
    probability of loss, 5th/95th percentile, and annualized return."""
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in vols:
        for T in horizons:
            ST = simulate_gbm_terminal(S0, mu, sigma, T, n_paths, seed=rng)
            s = distribution_stats(ST, S0, T)
            rows.append({
                "sigma": sigma, "T": T, "prob_loss": s["prob_loss"],
                "p05": s["p05"], "p50": s["p50"], "p95": s["p95"],
                "expected_return": s["expected_return"],
                "median_annualized": (1 + s["p50"]) ** (1 / T) - 1,
            })
    return rows
