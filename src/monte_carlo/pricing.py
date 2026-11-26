"""Monte Carlo option pricing with confidence intervals and variance
reduction, plus the closed-form prices used to check it.

All Monte Carlo prices are computed under the risk-neutral measure: the
asset drifts at r - q, and payoffs are discounted at r.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .paths import _normals, _rng, simulate_gbm


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------
def black_scholes(S0, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price of a European option."""
    if T <= 0 or sigma <= 0:
        fwd_payoff = S0 * np.exp(-q * T) - K * np.exp(-r * T)
        return max(fwd_payoff, 0.0) if kind == "call" else max(-fwd_payoff, 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    if kind == "put":
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1))
    raise ValueError("kind must be 'call' or 'put'")


def implied_vol(price, S0, K, T, r, q=0.0, kind="call"):
    """Volatility that makes the Black-Scholes price equal `price`."""
    f = lambda s: black_scholes(S0, K, T, r, s, q, kind) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except ValueError:
        return float("nan")


def merton_price(S0, K, T, r, sigma, lam, jump_mean, jump_std, q=0.0,
                 kind="call", n_terms=60):
    """Merton (1976) jump-diffusion price: a Poisson-weighted sum of
    Black-Scholes prices, one per number of jumps."""
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    lam_p = lam * (1 + k)
    total = 0.0
    for n in range(n_terms):
        weight = np.exp(-lam_p * T) * (lam_p * T) ** n / factorial(n)
        sigma_n = np.sqrt(sigma**2 + n * jump_std**2 / T)
        r_n = r - lam * k + n * np.log1p(k) / T
        # Weights use lam' = lam(1+k), which absorbs the change of discount
        # rate from r to r_n (Hull, Options, Futures and Other Derivatives).
        total += weight * black_scholes(S0, K, T, r_n, sigma_n, q, kind)
    return float(total)


def geometric_asian_price(S0, K, T, r, sigma, n_steps, q=0.0, kind="call"):
    """Closed-form price of a discretely monitored geometric-average Asian
    option (fixings at T/n, 2T/n, ..., T). Used as a control variate."""
    n = n_steps
    mu_g = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var_g = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    s_g = np.sqrt(var_g)
    d1 = (mu_g - np.log(K) + var_g) / s_g
    d2 = d1 - s_g
    fwd = np.exp(mu_g + 0.5 * var_g)
    df = np.exp(-r * T)
    if kind == "call":
        return float(df * (fwd * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(df * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1)))


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
@dataclass
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    method: str

    @property
    def ci_width(self):
        return self.ci_high - self.ci_low

    def contains(self, value):
        return self.ci_low <= value <= self.ci_high

    def as_dict(self):
        return {
            "price": self.price, "std_error": self.std_error,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "n_paths": self.n_paths, "method": self.method,
        }


def _summarize(samples, n_paths, method, confidence):
    """Mean, standard error and normal-approximation CI of i.i.d. samples."""
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(samples.size)
    z = norm.ppf(0.5 + confidence / 2)
    return MCResult(float(mean), float(se), float(mean - z * se),
                    float(mean + z * se), int(n_paths), method)


def _payoff(ST, K, kind):
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def mc_european(S0, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000,
                method="plain", seed=None, confidence=0.95):
    """Monte Carlo price of a European option.

    method
      "plain"       independent draws
      "antithetic"  pairs (Z, -Z); the pair average is one i.i.d. sample, so
                    the standard error accounts for the pairing
      "control"     control variate on the discounted terminal price, whose
                    risk-neutral expectation S0*exp(-qT) is known exactly
    """
    rng = _rng(seed)
    df = np.exp(-r * T)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    if method == "antithetic":
        half = n_paths // 2
        z = rng.standard_normal(half)
        up = _payoff(S0 * np.exp(drift + vol * z), K, kind)
        dn = _payoff(S0 * np.exp(drift - vol * z), K, kind)
        return _summarize(df * 0.5 * (up + dn), 2 * half, method, confidence)

    z = rng.standard_normal(n_paths)
    ST = S0 * np.exp(drift + vol * z)
    disc_payoff = df * _payoff(ST, K, kind)

    if method == "plain":
        return _summarize(disc_payoff, n_paths, method, confidence)
    if method == "control":
        control = df * ST
        control_mean = S0 * np.exp(-q * T)
        cov = np.cov(disc_payoff, control)
        beta = cov[0, 1] / cov[1, 1]
        adjusted = disc_payoff - beta * (control - control_mean)
        return _summarize(adjusted, n_paths, method, confidence)
    raise ValueError("method must be 'plain', 'antithetic' or 'control'")


def mc_asian(S0, K, T, r, sigma, q=0.0, kind="call", n_steps=252,
             n_paths=50_000, control_variate=True, seed=None, confidence=0.95):
    """Arithmetic-average Asian option (no closed form exists).

    With control_variate=True the geometric-average Asian, which has a
    closed form and is highly correlated with the arithmetic one, is used
    as the control.
    """
    _, paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, q=q, seed=seed)
    fixings = paths[:, 1:]
    df = np.exp(-r * T)
    arith = df * _payoff(fixings.mean(axis=1), K, kind)
    if not control_variate:
        return _summarize(arith, n_paths, "plain", confidence)
    geo = df * _payoff(np.exp(np.log(fixings).mean(axis=1)), K, kind)
    geo_exact = geometric_asian_price(S0, K, T, r, sigma, n_steps, q, kind)
    cov = np.cov(arith, geo)
    beta = cov[0, 1] / cov[1, 1]
    return _summarize(arith - beta * (geo - geo_exact), n_paths, "control", confidence)


def mc_merton(S0, K, T, r, sigma, lam, jump_mean, jump_std, q=0.0,
              kind="call", n_paths=200_000, seed=None, confidence=0.95):
    """Monte Carlo price under Merton jump-diffusion (terminal sampling)."""
    rng = _rng(seed)
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    n_jumps = rng.poisson(lam * T, n_paths)
    log_ST = (np.log(S0) + (r - q - lam * k - 0.5 * sigma**2) * T
              + sigma * np.sqrt(T) * rng.standard_normal(n_paths)
              + n_jumps * jump_mean
              + np.sqrt(n_jumps) * jump_std * rng.standard_normal(n_paths))
    disc = np.exp(-r * T) * _payoff(np.exp(log_ST), K, kind)
    return _summarize(disc, n_paths, "plain", confidence)


def convergence_study(S0, K, T, r, sigma, q=0.0, kind="call",
                      sizes=(100, 300, 1_000, 3_000, 10_000, 30_000,
                             100_000, 300_000, 1_000_000),
                      methods=("plain", "antithetic", "control"),
                      seed=0):
    """Price at increasing path counts with each estimator; returns rows of
    dicts including the error against the Black-Scholes price."""
    exact = black_scholes(S0, K, T, r, sigma, q, kind)
    rng = np.random.default_rng(seed)
    rows = []
    for method in methods:
        for n in sizes:
            res = mc_european(S0, K, T, r, sigma, q, kind, n, method, seed=rng)
            rows.append({
                "method": method, "n_paths": n, "price": res.price,
                "std_error": res.std_error, "ci_low": res.ci_low,
                "ci_high": res.ci_high, "exact": exact,
                "abs_error": abs(res.price - exact),
                "ci_contains_exact": res.contains(exact),
            })
    return rows
