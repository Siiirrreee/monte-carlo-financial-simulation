"""Matplotlib charts for the README and results folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow) and text inks.
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]
TEXT = "#191919"
MUTED = "#666666"
GRID = "#e6e6e6"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"]
)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": TEXT,
    "axes.edgecolor": "#bbbbbb", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "legend.frameon": False, "lines.linewidth": 2,
})

_dollars = FuncFormatter(lambda x, _: f"{x:,.0f}")


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_path_fan(times, paths, theory_mean, title, out_dir, n_show=120):
    """Sample paths plus percentile bands."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, paths[:n_show].T, color=SERIES[0], alpha=0.12, linewidth=0.7)
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(times, p5, p95, color=SERIES[0], alpha=0.12, linewidth=0,
                    label="5th–95th percentile")
    ax.fill_between(times, p25, p75, color=SERIES[0], alpha=0.22, linewidth=0,
                    label="25th–75th percentile")
    ax.plot(times, p50, color="#0e3d7e", label="Simulated median")
    ax.plot(times, theory_mean, color=SERIES[1], linestyle="--",
            label="Theoretical mean  S₀·e^(μt)")
    ax.set_xlim(times[0], times[-1])
    ax.yaxis.set_major_formatter(_dollars)
    ax.set_xlabel("Years")
    ax.set_ylabel("Index level")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "price_paths.png")


def plot_scenario_distributions(returns_by_scenario, out_dir):
    """Distribution of one-year returns under each market condition."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-0.6, 0.8, 141)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for color, (name, (ret, var)) in zip(SERIES, returns_by_scenario.items()):
        density, _ = np.histogram(ret, bins=bins, density=True)
        ax.plot(centers, density, color=color, label=f"{name}  (95% VaR {var:.1%})")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("One-year return")
    ax.set_ylabel("Probability density")
    ax.set_title("One-year return distribution by market condition")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "scenario_distributions.png")


def plot_convergence(rows, exact, out_dir):
    """Left: plain-MC estimate with 95% CI vs path count. Right: CI half-width
    for each estimator against the 1/sqrt(n) rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    plain = [r for r in rows if r["method"] == "plain"]
    n = np.array([r["n_paths"] for r in plain])
    price = np.array([r["price"] for r in plain])
    lo = np.array([r["ci_low"] for r in plain])
    hi = np.array([r["ci_high"] for r in plain])
    ax1.fill_between(n, lo, hi, color=SERIES[0], alpha=0.18, linewidth=0,
                     label="95% confidence interval")
    ax1.plot(n, price, color=SERIES[0], marker="o", markersize=5,
             label="Monte Carlo estimate")
    ax1.axhline(exact, color=SERIES[1], linestyle="--",
                label=f"Black-Scholes ${exact:,.2f}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Simulated paths")
    ax1.set_ylabel("Call price ($)")
    ax1.set_title("Estimate converges to the closed form")
    ax1.legend(loc="upper right")

    labels = {"plain": "Plain", "antithetic": "Antithetic variates",
              "control": "Control variate"}
    for color, method in zip(SERIES, ["plain", "antithetic", "control"]):
        sub = [r for r in rows if r["method"] == method]
        ax2.plot([r["n_paths"] for r in sub],
                 [r["ci_high"] - r["price"] for r in sub],
                 color=color, marker="o", markersize=5, label=labels[method])
    ref_n = np.array([n.min(), n.max()])
    # Anchor the reference on the most precise plain estimate.
    last = plain[-1]["ci_high"] - plain[-1]["price"]
    ax2.plot(ref_n, last * np.sqrt(n.max() / ref_n) * 1.6, color=MUTED,
             linestyle=":", linewidth=1.5, label="1/√n reference")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Simulated paths")
    ax2.set_ylabel("95% CI half-width ($)")
    ax2.set_title("Variance reduction shrinks the interval")
    ax2.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, out_dir, "convergence.png")


def plot_price_heatmap(grid_rows, vols, maturities, K, out_dir):
    """At-the-money call price (% of spot) across volatility and maturity."""
    z = np.array([[next(r["pct_of_spot"] for r in grid_rows
                        if r["sigma"] == s and r["T"] == t) for t in maturities]
                  for s in vols])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(z, cmap=SEQ_BLUE, aspect="auto", origin="lower")
    ax.grid(False)
    tlabels = ["1M" if t < 0.1 else "3M" if t == 0.25 else "6M" if t == 0.5
               else f"{t:g}Y" for t in maturities]
    ax.set_xticks(range(len(maturities)), tlabels)
    ax.set_yticks(range(len(vols)), [f"{v:.0%}" for v in vols])
    ax.set_xlabel("Time to expiry")
    ax.set_ylabel("Volatility")
    vmax = z.max()
    for i in range(len(vols)):
        for j in range(len(maturities)):
            ax.text(j, i, f"{z[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if z[i, j] > 0.55 * vmax else TEXT)
    cb = fig.colorbar(im, ax=ax, format=PercentFormatter(1.0, decimals=0))
    cb.set_label("Call price as % of spot")
    ax.set_title(f"Monte Carlo call price, strike {K:,.0f}")
    return _save(fig, out_dir, "price_sensitivity.png")


def plot_horizon_risk(rows, out_dir):
    """Probability of a loss against investment horizon, by volatility."""
    fig, ax = plt.subplots(figsize=(9, 5))
    vols = sorted({r["sigma"] for r in rows})
    for color, s in zip(SERIES, vols):
        sub = [r for r in rows if r["sigma"] == s]
        x = [r["T"] for r in sub]
        y = [r["prob_loss"] for r in sub]
        ax.plot(x, y, color=color, marker="o", markersize=6, label=f"σ = {s:.0%}")
        ax.annotate(f"{y[-1]:.0%}", (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=TEXT)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, None)
    ax.set_xlabel("Holding period (years)")
    ax.set_ylabel("Probability of ending below the start")
    ax.set_title("Longer horizons cut the chance of loss, unless volatility is high")
    ax.legend(loc="lower left")
    return _save(fig, out_dir, "horizon_risk.png")


def plot_jump_skew(rows, flat_vol, out_dir):
    """Implied volatility across strikes for Merton jump-diffusion prices."""
    fig, ax = plt.subplots(figsize=(9, 5))
    m = np.array([r["moneyness"] for r in rows])
    ax.plot(m, [r["iv_closed_form"] for r in rows], color=SERIES[0],
            label="Jump-diffusion, closed form")
    ax.errorbar(m, [r["iv_mc"] for r in rows],
                yerr=[[r["iv_mc"] - r["iv_ci_low"] for r in rows],
                      [r["iv_ci_high"] - r["iv_mc"] for r in rows]],
                fmt="o", markersize=6, color=SERIES[1], capsize=3, linewidth=1.5,
                label="Jump-diffusion, Monte Carlo (95% CI)")
    ax.axhline(flat_vol, color=MUTED, linestyle="--", linewidth=1.5,
               label=f"Black-Scholes: flat {flat_vol:.0%}")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Strike as % of forward")
    ax.set_ylabel("Black-Scholes implied volatility")
    ax.set_title("Crash risk produces the volatility skew Black-Scholes misses")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "jump_skew.png")
