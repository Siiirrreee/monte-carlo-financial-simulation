"""Statistics on simulated price distributions, with closed-form GBM
benchmarks to check the simulation against."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def distribution_stats(terminal, S0, T, confidence=0.95):
    """Summary statistics of simulated terminal prices.

    Returns a dict of plain floats. Returns are simple returns S(T)/S0 - 1
    over the whole horizon; VaR and CVaR are reported as positive loss
    fractions of the starting value.
    """
    terminal = np.asarray(terminal, dtype=float)
    ret = terminal / S0 - 1.0
    n = ret.size
    alpha = 1.0 - confidence
    var_cut = np.quantile(ret, alpha)
    tail = ret[ret <= var_cut]
    mean_ret = ret.mean()
    se = ret.std(ddof=1) / np.sqrt(n)
    z = norm.ppf(0.5 + confidence / 2)
    log_ret = np.log(terminal / S0)
    return {
        "n_paths": int(n),
        "expected_price": float(terminal.mean()),
        "median_price": float(np.median(terminal)),
        "expected_return": float(mean_ret),
        "expected_return_ci": (float(mean_ret - z * se), float(mean_ret + z * se)),
        "annualized_return": float((1 + mean_ret) ** (1 / T) - 1),
        "volatility_annual": float(log_ret.std(ddof=1) / np.sqrt(T)),
        "p05": float(np.quantile(ret, 0.05)),
        "p25": float(np.quantile(ret, 0.25)),
        "p50": float(np.quantile(ret, 0.50)),
        "p75": float(np.quantile(ret, 0.75)),
        "p95": float(np.quantile(ret, 0.95)),
        "prob_loss": float((ret < 0).mean()),
        "prob_loss_10": float((ret < -0.10).mean()),
        "prob_loss_20": float((ret < -0.20).mean()),
        "prob_gain_20": float((ret > 0.20).mean()),
        "var": float(-var_cut),
        "cvar": float(-tail.mean()),
        "skew": float(((log_ret - log_ret.mean()) ** 3).mean() / log_ret.std() ** 3),
        "excess_kurtosis": float(
            ((log_ret - log_ret.mean()) ** 4).mean() / log_ret.std() ** 4 - 3
        ),
    }


def max_drawdowns(paths):
    """Largest peak-to-trough fall along each path, as a positive fraction."""
    running_max = np.maximum.accumulate(paths, axis=1)
    return (1.0 - paths / running_max).max(axis=1)


def gbm_theory(S0, mu, sigma, T, q=0.0):
    """Closed-form moments of GBM terminal price, for validating simulations.

    log S(T) ~ Normal(log S0 + (mu - q - sigma^2/2) T, sigma^2 T)
    """
    m = np.log(S0) + (mu - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    mean = S0 * np.exp((mu - q) * T)
    return {
        "expected_price": float(mean),
        "median_price": float(np.exp(m)),
        "std_price": float(mean * np.sqrt(np.exp(s**2) - 1)),
        "expected_return": float(np.exp((mu - q) * T) - 1),
        "prob_loss": float(norm.cdf((np.log(S0) - m) / s)),
        "p05": float(np.exp(m + s * norm.ppf(0.05)) / S0 - 1),
        "p95": float(np.exp(m + s * norm.ppf(0.95)) / S0 - 1),
    }


def lognormal_return_pdf(x, mu, sigma, T, q=0.0):
    """Density of the simple return R = S(T)/S0 - 1 under GBM."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    ok = x > -1
    y = 1 + x[ok]
    m = (mu - q - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    out[ok] = norm.pdf((np.log(y) - m) / s) / (y * s)
    return out
