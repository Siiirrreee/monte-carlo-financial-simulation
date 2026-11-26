"""Run the full Monte Carlo pipeline: simulate, price, analyse, chart, and
write results (CSV, JSON, Markdown) plus the README results section.

    python main.py                 # default settings
    python main.py --paths 50000   # fewer paths for a quick run
    python main.py --seed 7        # different random seed
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from monte_carlo import visuals  # noqa: E402
from monte_carlo.analytics import distribution_stats, gbm_theory, max_drawdowns  # noqa: E402
from monte_carlo.paths import simulate_gbm, simulate_merton  # noqa: E402
from monte_carlo.pricing import (  # noqa: E402
    black_scholes, convergence_study, implied_vol, mc_asian, mc_european,
    mc_merton, merton_price,
)
from monte_carlo.scenarios import MARKET_CHECK, RISK_FREE, S0, SCENARIOS  # noqa: E402
from monte_carlo.sensitivity import (  # noqa: E402
    DEFAULT_MATURITIES, DEFAULT_VOLS, horizon_risk, option_price_grid,
)

RESULTS = ROOT / "results"
VISUALS = ROOT / "visuals"
README = ROOT / "README.md"
K_ATM = 7200.0


def pct(x, d=1):
    return f"{x:.{d}%}"


def run(n_paths: int, seed: int) -> dict:
    RESULTS.mkdir(exist_ok=True)
    VISUALS.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    stats: dict = {"settings": {"n_paths": n_paths, "seed": seed, "S0": S0,
                                "r": RISK_FREE}}

    # 1. Market check: price the real SPX contract from the options project.
    m = MARKET_CHECK
    bs = black_scholes(m["S0"], m["K"], m["T"], m["r"], m["sigma"])
    mc = mc_european(m["S0"], m["K"], m["T"], m["r"], m["sigma"],
                     n_paths=max(n_paths * 5, 100_000), method="control", seed=rng)
    stats["market_check"] = {**m, "bs_price": bs, **{f"mc_{k}": v for k, v in mc.as_dict().items()},
                             "ci_contains_bs": mc.contains(bs)}

    # 2. Price paths for the base case.
    base = SCENARIOS["base"]
    times, paths = simulate_gbm(S0, base.mu, base.sigma, 1.0, 252, 10_000, seed=rng)
    visuals.plot_path_fan(times, paths, S0 * np.exp(base.mu * times),
                          f"10,000 simulated one-year paths, base case "
                          f"(μ = {base.mu:.0%}, σ = {base.sigma:.0%})", VISUALS)

    # 3. Market conditions: one-year outcome distribution per scenario.
    scen_rows, dist_for_chart = [], {}
    n_scen = max(n_paths // 5, 20_000)
    for key, sc in SCENARIOS.items():
        if sc.has_jumps:
            _, p = simulate_merton(S0, sc.mu, sc.sigma, 1.0, n_steps=252,
                                   n_paths=n_scen, seed=rng, **sc.jumps)
        else:
            _, p = simulate_gbm(S0, sc.mu, sc.sigma, 1.0, 252, n_scen, seed=rng)
        s = distribution_stats(p[:, -1], S0, 1.0)
        dd = max_drawdowns(p)
        theory = None if sc.has_jumps else gbm_theory(S0, sc.mu, sc.sigma, 1.0)
        scen_rows.append({
            "key": key, "scenario": sc.name, "mu": sc.mu, "sigma": sc.sigma,
            "total_vol": sc.total_vol, **{k: v for k, v in s.items()
                                          if k != "expected_return_ci"},
            "expected_return_ci_low": s["expected_return_ci"][0],
            "expected_return_ci_high": s["expected_return_ci"][1],
            "theory_expected_return": theory["expected_return"] if theory else np.nan,
            "theory_prob_loss": theory["prob_loss"] if theory else np.nan,
            "median_max_drawdown": float(np.median(dd)),
            "p95_max_drawdown": float(np.quantile(dd, 0.95)),
            "prob_drawdown_20": float((dd > 0.20).mean()),
        })
        dist_for_chart[sc.name] = (p[:, -1] / S0 - 1, s["var"])
    scen_df = pd.DataFrame(scen_rows)
    scen_df.to_csv(RESULTS / "scenario_stats.csv", index=False)
    visuals.plot_scenario_distributions(dist_for_chart, VISUALS)
    stats["scenarios"] = scen_rows

    # 4. Convergence and variance reduction (1-year ATM call, sigma 18%).
    conv = convergence_study(S0, K_ATM, 1.0, RISK_FREE, 0.18, seed=seed)
    conv_df = pd.DataFrame(conv)
    conv_df.to_csv(RESULTS / "convergence.csv", index=False)
    visuals.plot_convergence(conv, conv[0]["exact"], VISUALS)
    at_1m = conv_df[conv_df.n_paths == conv_df.n_paths.max()].set_index("method")
    se_plain = at_1m.loc["plain", "std_error"]
    stats["convergence"] = {
        "exact": conv[0]["exact"],
        "coverage": float(conv_df.ci_contains_exact.mean()),
        "at_max_paths": at_1m[["price", "std_error", "abs_error"]].to_dict("index"),
        "variance_reduction": {mth: float((se_plain / at_1m.loc[mth, "std_error"]) ** 2)
                               for mth in at_1m.index},
    }

    # 5. Volatility x maturity price grid.
    grid = option_price_grid(S0, K_ATM, RISK_FREE, n_paths=max(n_paths, 50_000),
                             seed=seed + 1)
    grid_df = pd.DataFrame(grid)
    grid_df.to_csv(RESULTS / "price_sensitivity.csv", index=False)
    visuals.plot_price_heatmap(grid, DEFAULT_VOLS, DEFAULT_MATURITIES, K_ATM, VISUALS)
    stats["price_grid"] = {"coverage": float(grid_df.ci_contains_bs.mean()),
                           "max_abs_error": float((grid_df.mc_price - grid_df.bs_price).abs().max())}

    # 6. Horizon risk (real-world drift 7%).
    hz = horizon_risk(S0, base.mu, n_paths=max(n_paths, 50_000), seed=seed + 2)
    hz_df = pd.DataFrame(hz)
    hz_df.to_csv(RESULTS / "horizon_risk.csv", index=False)
    visuals.plot_horizon_risk(hz, VISUALS)
    stats["horizon_risk"] = hz

    # 7. Path-dependent option: arithmetic Asian vs European.
    eu = black_scholes(S0, K_ATM, 1.0, RISK_FREE, 0.18)
    asian_cv = mc_asian(S0, K_ATM, 1.0, RISK_FREE, 0.18, n_paths=50_000, seed=seed + 3)
    asian_pl = mc_asian(S0, K_ATM, 1.0, RISK_FREE, 0.18, n_paths=50_000,
                        control_variate=False, seed=seed + 3)
    stats["asian"] = {"european_bs": eu, "asian_cv": asian_cv.as_dict(),
                      "asian_plain": asian_pl.as_dict(),
                      "variance_reduction": (asian_pl.std_error / asian_cv.std_error) ** 2}

    # 8. Jump-diffusion skew: 3-month options under the crash-prone scenario.
    cp = SCENARIOS["crash_prone"]
    T_s = 0.25
    F = S0 * np.exp(RISK_FREE * T_s)
    skew_rows = []
    for mny in np.arange(0.80, 1.151, 0.025):
        K = round(F * mny, 2)
        kind = "put" if mny < 1 else "call"
        cf = merton_price(S0, K, T_s, RISK_FREE, cp.sigma, kind=kind, **cp.jumps)
        mcr = mc_merton(S0, K, T_s, RISK_FREE, cp.sigma, kind=kind,
                        n_paths=max(n_paths * 2, 100_000), seed=rng, **cp.jumps)
        iv = lambda p: implied_vol(p, S0, K, T_s, RISK_FREE, kind=kind)
        skew_rows.append({
            "moneyness": float(mny), "strike": K, "kind": kind,
            "price_closed_form": cf, "price_mc": mcr.price, "mc_ci_low": mcr.ci_low,
            "mc_ci_high": mcr.ci_high, "iv_closed_form": iv(cf), "iv_mc": iv(mcr.price),
            "iv_ci_low": iv(max(mcr.ci_low, 1e-6)), "iv_ci_high": iv(mcr.ci_high),
            "ci_contains_closed_form": mcr.contains(cf),
        })
    skew_df = pd.DataFrame(skew_rows)
    skew_df.to_csv(RESULTS / "jump_skew.csv", index=False)
    atm_iv = float(np.interp(1.0, skew_df.moneyness, skew_df.iv_closed_form))
    visuals.plot_jump_skew(skew_rows, atm_iv, VISUALS)
    stats["jump_skew"] = {"T": T_s, "atm_iv": atm_iv,
                          "iv_80": float(skew_df.iv_closed_form.iloc[0]),
                          "iv_115": float(skew_df.iv_closed_form.iloc[-1]),
                          "coverage": float(skew_df.ci_contains_closed_form.mean())}

    (RESULTS / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
    md = build_summary(stats, scen_df, conv_df, grid_df, hz_df, skew_df)
    (RESULTS / "summary.md").write_text(md)
    update_readme(md)
    return stats


def build_summary(stats, scen_df, conv_df, grid_df, hz_df, skew_df) -> str:
    st = stats["settings"]
    mk = stats["market_check"]
    cv = stats["convergence"]
    asian = stats["asian"]
    js = stats["jump_skew"]
    base = scen_df.set_index("key").loc["base"]
    crash = scen_df.set_index("key").loc["crash_prone"]
    stressed = scen_df.set_index("key").loc["stressed"]

    scen_tbl = pd.DataFrame({
        "Scenario": scen_df.scenario,
        "μ / total σ": [f"{a:.0%} / {b:.0%}" + (" incl. jumps" if k == "crash_prone" else "")
                        for a, b, k in zip(scen_df.mu, scen_df.total_vol, scen_df.key)],
        "Expected return": scen_df.expected_return.map(pct),
        "Median return": scen_df.p50.map(pct),
        "5th–95th pct": [f"{a:.0%} to {b:+.0%}" for a, b in zip(scen_df.p05, scen_df.p95)],
        "P(loss)": scen_df.prob_loss.map(pct),
        "95% VaR": scen_df["var"].map(pct),
        "95% CVaR": scen_df.cvar.map(pct),
        "Median max drawdown": scen_df.median_max_drawdown.map(pct),
    }).to_markdown(index=False)

    conv_tbl = pd.DataFrame([{
        "Estimator": {"plain": "Plain", "antithetic": "Antithetic variates",
                      "control": "Control variate"}[mth],
        "Price": f"{v['price']:.2f}",
        "Std. error": f"{v['std_error']:.3f}",
        "Error vs BS": f"{v['abs_error']:.3f}",
        "Variance reduction": f"{cv['variance_reduction'][mth]:.1f}×",
    } for mth, v in cv["at_max_paths"].items()]).to_markdown(index=False)

    hz = hz_df[hz_df["T"].isin([1, 5, 10])]
    hz_tbl = hz.pivot(index="T", columns="sigma", values="prob_loss")
    hz_tbl.columns = [f"P(loss), σ = {c:.0%}" for c in hz_tbl.columns]
    hz_tbl.index = [f"{int(t)} yr" for t in hz_tbl.index]
    hz_tbl = hz_tbl.map(pct).rename_axis("Horizon").reset_index().to_markdown(index=False)

    n_max = int(conv_df.n_paths.max())
    lines = [
        f"_S&P 500 starting level {st['S0']:,.0f}, r = {st['r']:.1%}, seed {st['seed']}. "
        "All prices are risk-neutral; scenario drifts are illustrative assumptions. "
        "Full detail in [results/summary.md](results/summary.md)._",
        "",
        "**Market check: pricing the real SPX contract from the options project**",
        "",
        f"| Contract | Market mid | Black-Scholes | Monte Carlo (95% CI) | Paths |",
        "|:--|--:|--:|--:|--:|",
        f"| {mk['label']}, σ = {mk['sigma']:.1%} | {mk['market_mid']:.2f} | "
        f"{mk['bs_price']:.2f} | {mk['mc_price']:.2f} ({mk['mc_ci_low']:.2f}–{mk['mc_ci_high']:.2f}) "
        f"| {mk['mc_n_paths']:,} |",
        "",
        "**One-year outcomes under four market conditions**",
        "",
        scen_tbl,
        "",
        f"**Convergence: 1-year {K_ATM:,.0f} call, σ = 18%, {n_max:,} paths "
        f"(Black-Scholes ${cv['exact']:,.2f})**",
        "",
        conv_tbl,
        "",
        "**Probability of losing money by holding period (μ = 7%)**",
        "",
        hz_tbl,
        "",
        "**Key findings**",
        "",
        f"- The simulation reproduces the closed form: the Monte Carlo price of the "
        f"SPX Dec-18 7,200 call is **${mk['mc_price']:.2f}** against Black-Scholes "
        f"${mk['bs_price']:.2f} and a market mid of ${mk['market_mid']:.2f}. Across the "
        f"convergence study, the 95% interval contained the exact price in "
        f"**{cv['coverage']:.0%}** of {len(conv_df)} runs (in line with the 95% "
        f"nominal rate), and across the {len(grid_df)}-cell "
        f"volatility × maturity grid in **{stats['price_grid']['coverage']:.0%}**.",
        f"- Error shrinks at the textbook 1/√n rate: 100× more paths buys only 10× "
        f"more precision. Variance reduction is cheaper: the control variate cut "
        f"variance **{cv['variance_reduction']['control']:.1f}×** and antithetic "
        f"variates **{cv['variance_reduction']['antithetic']:.1f}×**, equivalent to "
        f"running that many times more paths.",
        f"- In the base case (μ = 7%, σ = 18%) the expected one-year return is "
        f"**{base.expected_return:.1%}**, but the median is lower at "
        f"{base.p50:.1%}: lognormal returns are right-skewed, so the average is pulled "
        f"up by a few large winners. There is a **{base.prob_loss:.0%}** chance of "
        f"ending the year down.",
        f"- The crash-prone scenario has the **same expected return and the same 18% "
        f"total volatility** as the base case, yet a fatter left tail: 95% CVaR rises "
        f"from {base.cvar:.1%} to **{crash.cvar:.1%}** and the chance of losing more "
        f"than 20% goes from {base.prob_loss_20:.1%} to {crash.prob_loss_20:.1%}. "
        f"Ordinary years are actually calmer (P(loss) {crash.prob_loss:.1%} vs "
        f"{base.prob_loss:.1%}): the risk is concentrated in rare crashes. Volatility "
        f"alone understates risk when moves come in jumps. The "
        f"high-volatility regime is worst on every risk measure "
        f"(P(loss) {stressed.prob_loss:.0%}, CVaR {stressed.cvar:.1%}).",
        f"- Time diversifies volatility only partly: at σ = 18% the chance of a loss "
        f"falls from {hz_df.query('sigma == 0.18 and T == 1').prob_loss.iloc[0]:.0%} "
        f"over 1 year to {hz_df.query('sigma == 0.18 and T == 10').prob_loss.iloc[0]:.0%} "
        f"over 10, while at σ = 30% it barely moves "
        f"({hz_df.query('sigma == 0.3 and T == 1').prob_loss.iloc[0]:.0%} → "
        f"{hz_df.query('sigma == 0.3 and T == 10').prob_loss.iloc[0]:.0%}), because the "
        f"volatility drag σ²/2 eats most of a 7% drift.",
        f"- An arithmetic-average Asian call, which has no closed form, prices at "
        f"**${asian['asian_cv']['price']:.2f}** vs ${asian['european_bs']:.2f} for the "
        f"European: averaging dampens volatility. Using the geometric Asian as a "
        f"control variate cut variance **{asian['variance_reduction']:,.0f}×**.",
        f"- Adding jumps explains the skew found in the options project: priced under "
        f"the crash-prone model, 3-month options show implied volatility of "
        f"**{js['iv_80']:.1%}** at the 80% strike, {js['atm_iv']:.1%} at the money and "
        f"{js['iv_115']:.1%} at 115%, a downward-sloping skew that a single "
        f"Black-Scholes volatility cannot produce.",
    ]
    return "\n".join(lines) + "\n"


def update_readme(md: str) -> None:
    if not README.exists():
        return
    text = README.read_text()
    new = re.sub(r"(<!-- RESULTS:START -->\n).*?(<!-- RESULTS:END -->)",
                 lambda m_: m_.group(1) + md + m_.group(2), text, flags=re.S)
    README.write_text(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=int, default=200_000,
                    help="base number of simulated paths (default 200,000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()
    stats = run(args.paths, args.seed)
    mk = stats["market_check"]
    print(f"Market check: MC {mk['mc_price']:.2f} vs BS {mk['bs_price']:.2f} "
          f"vs market {mk['market_mid']:.2f}")
    print(f"Results written to {RESULTS}, charts to {VISUALS}")


if __name__ == "__main__":
    main()
