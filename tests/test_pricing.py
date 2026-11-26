import numpy as np
import pytest

from monte_carlo.pricing import (
    black_scholes, convergence_study, geometric_asian_price, implied_vol,
    mc_asian, mc_european, mc_merton, merton_price,
)
from monte_carlo.scenarios import MARKET_CHECK

ARGS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_black_scholes_textbook():
    # Hull's classic example values.
    assert black_scholes(**ARGS, kind="call") == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes(**ARGS, kind="put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_matches_black_scholes(method, kind):
    exact = black_scholes(**ARGS, kind=kind)
    res = mc_european(**ARGS, kind=kind, n_paths=400_000, method=method, seed=11)
    assert abs(res.price - exact) < 4 * res.std_error
    assert res.ci_low < res.price < res.ci_high


def test_variance_reduction_lowers_std_error():
    se = {m: mc_european(**ARGS, n_paths=200_000, method=m, seed=2).std_error
          for m in ("plain", "antithetic", "control")}
    assert se["antithetic"] < se["plain"]
    assert se["control"] < se["antithetic"]


def test_std_error_scales_with_inverse_sqrt_n():
    a = mc_european(**ARGS, n_paths=10_000, seed=3).std_error
    b = mc_european(**ARGS, n_paths=1_000_000, seed=4).std_error
    assert a / b == pytest.approx(10, rel=0.1)


def test_put_call_parity_in_simulation():
    """Same draws for call and put: parity holds path by path in expectation."""
    c = mc_european(**ARGS, kind="call", n_paths=500_000, method="control", seed=5)
    p = mc_european(**ARGS, kind="put", n_paths=500_000, method="control", seed=5)
    parity = ARGS["S0"] - ARGS["K"] * np.exp(-ARGS["r"] * ARGS["T"])
    assert abs((c.price - p.price) - parity) < 4 * np.hypot(c.std_error, p.std_error)


def test_ci_coverage_close_to_nominal():
    exact = black_scholes(**ARGS)
    rng = np.random.default_rng(0)
    hits = [mc_european(**ARGS, n_paths=2_000, seed=rng).contains(exact)
            for _ in range(400)]
    assert 0.92 < np.mean(hits) < 0.98


def test_market_check_reproduces_spx_contract():
    m = MARKET_CHECK
    bs = black_scholes(m["S0"], m["K"], m["T"], m["r"], m["sigma"])
    res = mc_european(m["S0"], m["K"], m["T"], m["r"], m["sigma"],
                      n_paths=500_000, method="control", seed=6)
    assert res.contains(bs)
    assert abs(res.price - m["market_mid"]) / m["market_mid"] < 0.01


def test_merton_reduces_to_black_scholes_without_jumps():
    assert merton_price(**ARGS, lam=0.0, jump_mean=-0.1, jump_std=0.1) == \
        pytest.approx(black_scholes(**ARGS))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_merton_matches_closed_form(kind):
    jumps = dict(lam=1.5, jump_mean=-0.08, jump_std=0.12)
    exact = merton_price(**ARGS, kind=kind, **jumps)
    res = mc_merton(**ARGS, kind=kind, n_paths=600_000, seed=7, **jumps)
    assert abs(res.price - exact) < 4 * res.std_error


def test_jumps_create_downward_skew():
    jumps = dict(lam=1.0, jump_mean=-0.08, jump_std=0.10)
    S0, T, r, s = 100.0, 0.25, 0.05, 0.13
    ivs = []
    for K in (85.0, 100.0, 110.0):
        kind = "put" if K < 100 else "call"
        ivs.append(implied_vol(merton_price(S0, K, T, r, s, kind=kind, **jumps),
                               S0, K, T, r, kind=kind))
    assert ivs[0] > ivs[1] > ivs[2]


def test_implied_vol_round_trip():
    price = black_scholes(**ARGS, kind="put")
    assert implied_vol(price, 100, 100, 1.0, 0.05, kind="put") == pytest.approx(0.2)


def test_asian_geometric_mc_matches_closed_form():
    from monte_carlo.paths import simulate_gbm
    n = 50
    _, p = simulate_gbm(100, 0.05, 0.2, 1.0, n, 300_000, seed=8)
    geo = np.exp(np.log(p[:, 1:]).mean(axis=1))
    disc = np.exp(-0.05) * np.maximum(geo - 100, 0)
    exact = geometric_asian_price(100, 100, 1.0, 0.05, 0.2, n)
    assert abs(disc.mean() - exact) < 4 * disc.std() / np.sqrt(disc.size)


def test_asian_cheaper_than_european_and_control_variate_helps():
    cv = mc_asian(**ARGS, n_steps=50, n_paths=40_000, seed=9)
    plain = mc_asian(**ARGS, n_steps=50, n_paths=40_000, control_variate=False, seed=9)
    geo = geometric_asian_price(**ARGS, n_steps=50)
    assert geo < cv.price < black_scholes(**ARGS)  # AM >= GM, averaging lowers vol
    assert abs(cv.price - plain.price) < 4 * plain.std_error
    assert cv.std_error < plain.std_error / 10


def test_convergence_study_rows():
    rows = convergence_study(**ARGS, sizes=(1_000, 10_000), seed=1)
    assert len(rows) == 6
    assert {r["method"] for r in rows} == {"plain", "antithetic", "control"}


def test_bad_method():
    with pytest.raises(ValueError):
        mc_european(**ARGS, method="magic")
