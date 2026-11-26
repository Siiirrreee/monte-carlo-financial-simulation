"""Asset-price path simulation.

Two processes are implemented:

* Geometric Brownian motion (GBM), the Black-Scholes world:
      dS = (mu - q) S dt + sigma S dW
  simulated with the exact log-space solution, so there is no discretisation
  error at any step size:
      S(t+dt) = S(t) * exp((mu - q - sigma^2/2) dt + sigma sqrt(dt) Z)

* Merton jump-diffusion: GBM plus Poisson-arriving lognormal jumps, used to
  model crash-prone markets. The drift is compensated so that the expected
  growth rate is still mu - q:
      dS/S = (mu - q - lam*k) dt + sigma dW + (J - 1) dN,  k = E[J] - 1
"""
from __future__ import annotations

import numpy as np


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _normals(rng, n_paths, n_steps, antithetic):
    """Standard normals of shape (n_paths, n_steps); antithetic pairs Z and -Z."""
    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal((half, n_steps))
        return np.concatenate([z, -z])[:n_paths]
    return rng.standard_normal((n_paths, n_steps))


def simulate_gbm(S0, mu, sigma, T, n_steps=252, n_paths=10_000, q=0.0,
                 antithetic=False, seed=None):
    """Simulate GBM price paths.

    Returns
    -------
    times : ndarray (n_steps + 1,)
    paths : ndarray (n_paths, n_steps + 1), first column equal to S0
    """
    if sigma < 0 or T <= 0 or n_steps < 1 or n_paths < 1:
        raise ValueError("need sigma >= 0, T > 0, n_steps >= 1, n_paths >= 1")
    rng = _rng(seed)
    dt = T / n_steps
    z = _normals(rng, n_paths, n_steps, antithetic)
    log_increments = (mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    times = np.linspace(0.0, T, n_steps + 1)
    return times, S0 * np.exp(log_paths)


def simulate_gbm_terminal(S0, mu, sigma, T, n_paths=100_000, q=0.0,
                          antithetic=False, seed=None):
    """Terminal prices only (one exact step). Much faster when the payoff
    depends only on S(T)."""
    rng = _rng(seed)
    z = _normals(rng, n_paths, 1, antithetic)[:, 0]
    return S0 * np.exp((mu - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def simulate_merton(S0, mu, sigma, T, lam, jump_mean, jump_std, n_steps=252,
                    n_paths=10_000, q=0.0, seed=None):
    """Merton jump-diffusion paths.

    lam        jumps per year (Poisson intensity)
    jump_mean  mean of log jump size, e.g. -0.08 for an average 8% drop
    jump_std   standard deviation of log jump size
    """
    rng = _rng(seed)
    dt = T / n_steps
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0  # mean jump size E[J] - 1
    drift = (mu - q - lam * k - 0.5 * sigma**2) * dt
    z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
    # Sum of n lognormal log-jumps is normal with mean n*m and variance n*s^2.
    jumps = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal(
        (n_paths, n_steps)
    )
    log_increments = drift + sigma * np.sqrt(dt) * z + jumps
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    times = np.linspace(0.0, T, n_steps + 1)
    return times, S0 * np.exp(log_paths)
