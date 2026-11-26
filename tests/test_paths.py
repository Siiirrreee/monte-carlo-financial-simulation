import numpy as np
import pytest

from monte_carlo.analytics import distribution_stats, gbm_theory, max_drawdowns
from monte_carlo.paths import simulate_gbm, simulate_gbm_terminal, simulate_merton
from monte_carlo.scenarios import SCENARIOS


def test_shapes_and_start():
    t, p = simulate_gbm(100, 0.05, 0.2, 1.0, n_steps=50, n_paths=200, seed=1)
    assert t.shape == (51,) and p.shape == (200, 51)
    assert np.allclose(p[:, 0], 100) and (p > 0).all()


def test_reproducible_with_seed():
    a = simulate_gbm(100, 0.05, 0.2, 1.0, 20, 100, seed=7)[1]
    b = simulate_gbm(100, 0.05, 0.2, 1.0, 20, 100, seed=7)[1]
    assert np.array_equal(a, b)


def test_terminal_moments_match_theory():
    S0, mu, sigma, T = 100, 0.07, 0.25, 2.0
    ST = simulate_gbm_terminal(S0, mu, sigma, T, n_paths=400_000, seed=3)
    th = gbm_theory(S0, mu, sigma, T)
    se = th["std_price"] / np.sqrt(ST.size)
    assert abs(ST.mean() - th["expected_price"]) < 4 * se
    assert abs(np.median(ST) - th["median_price"]) / th["median_price"] < 0.005
    assert abs((ST < S0).mean() - th["prob_loss"]) < 0.005


def test_path_and_terminal_simulators_agree():
    """Multi-step exact simulation has the same terminal law as one step."""
    _, p = simulate_gbm(100, 0.05, 0.3, 1.0, 12, 200_000, seed=4)
    log_ret = np.log(p[:, -1] / 100)
    assert abs(log_ret.mean() - (0.05 - 0.045)) < 0.003
    assert abs(log_ret.std() - 0.3) < 0.003


def test_antithetic_pairs_mirror():
    _, p = simulate_gbm(100, 0.0, 0.2, 1.0, 1, 10, antithetic=True, seed=5)
    drift = -0.5 * 0.04
    log_ret = np.log(p[:, 1] / 100) - drift
    assert np.allclose(log_ret[:5], -log_ret[5:])


def test_zero_vol_is_deterministic():
    _, p = simulate_gbm(100, 0.05, 0.0, 2.0, 10, 5, seed=1)
    assert np.allclose(p[:, -1], 100 * np.exp(0.10))


def test_merton_mean_is_compensated():
    """Jumps change the shape, not the expected growth rate."""
    _, p = simulate_merton(100, 0.07, 0.15, 1.0, lam=2.0, jump_mean=-0.1,
                           jump_std=0.1, n_steps=50, n_paths=200_000, seed=6)
    ST = p[:, -1]
    se = ST.std() / np.sqrt(ST.size)
    assert abs(ST.mean() - 100 * np.exp(0.07)) < 4 * se


def test_crash_scenario_has_base_total_vol_and_fatter_left_tail():
    base, crash = SCENARIOS["base"], SCENARIOS["crash_prone"]
    assert crash.total_vol == pytest.approx(base.total_vol)
    _, pb = simulate_gbm(100, base.mu, base.sigma, 1.0, 50, 100_000, seed=8)
    _, pc = simulate_merton(100, crash.mu, crash.sigma, 1.0, n_steps=50,
                            n_paths=100_000, seed=9, **crash.jumps)
    sb = distribution_stats(pb[:, -1], 100, 1.0)
    sc = distribution_stats(pc[:, -1], 100, 1.0)
    assert sc["skew"] < sb["skew"] and sc["excess_kurtosis"] > sb["excess_kurtosis"]
    assert sc["cvar"] > sb["cvar"]


def test_distribution_stats_known_values():
    ST = np.array([80, 90, 100, 110, 120, 130, 140, 150, 160, 170], float)
    s = distribution_stats(ST, 100, 1.0, confidence=0.9)
    assert s["prob_loss"] == pytest.approx(0.2)
    assert s["expected_return"] == pytest.approx(0.25)
    assert s["cvar"] >= s["var"]


def test_max_drawdown():
    p = np.array([[100, 120, 90, 130, 104]], float)
    assert max_drawdowns(p)[0] == pytest.approx(0.25)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, -0.1, 1.0)
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, 0.2, 0.0)
