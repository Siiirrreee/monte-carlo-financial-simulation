"""Market-condition scenarios and model inputs.

The starting point is calibrated to the same S&P 500 (SPX) snapshot used in
the options-pricing project (2026-11-24 close):

* S0 = 7,136, the spot level implied by the put-call-parity forward
  (F = 7,155.7 for 24 days at r = 4.2%, so S0 = F * exp(-rT)). Using the
  parity-implied spot handles dividends and the quote-timing gap the same way
  the options project did.
* r = 4.2%, the 3-month T-bill rate.
* ATM implied volatility 23.4% (Dec-18 7,200 call, market mid $150.35).

The scenario drifts and volatilities are illustrative assumptions for
real-world ("P-measure") return distributions, not forecasts. Option pricing
always uses the risk-neutral drift r, never these drifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

S0 = 7136.0
RISK_FREE = 0.042

# Market check from the options-pricing project.
MARKET_CHECK = {
    "label": "SPX Dec-18 7,200 call",
    "S0": 7136.0,
    "K": 7200.0,
    "T": 24 / 365,
    "r": 0.042,
    "sigma": 0.234,
    "market_mid": 150.35,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    mu: float               # expected annual return (real-world drift)
    sigma: float            # annual diffusion volatility
    description: str
    jumps: dict = field(default_factory=dict)  # lam, jump_mean, jump_std

    @property
    def has_jumps(self) -> bool:
        return bool(self.jumps)

    @property
    def total_vol(self) -> float:
        """Annual volatility including the jump component."""
        if not self.jumps:
            return self.sigma
        j = self.jumps
        jump_var = j["lam"] * (j["jump_mean"] ** 2 + j["jump_std"] ** 2)
        return (self.sigma**2 + jump_var) ** 0.5


SCENARIOS = {
    "calm_bull": Scenario(
        "Calm bull", mu=0.10, sigma=0.12,
        description="Low volatility, strong trend (e.g. 2017, 2021)",
    ),
    "base": Scenario(
        "Base case", mu=0.07, sigma=0.18,
        description="Long-run equity-index averages",
    ),
    "stressed": Scenario(
        "High volatility", mu=0.02, sigma=0.30,
        description="Choppy, low-return, high-vol regime (e.g. 2008, 2022)",
    ),
    # Diffusion vol chosen so total volatility equals the base case's 18%:
    # sigma^2 + lam*(m^2 + s^2) = 0.18^2. Same mean, same volatility, so any
    # difference in the tails comes from the shape of the distribution.
    "crash_prone": Scenario(
        "Crash-prone (jumps)", mu=0.07,
        sigma=(0.18**2 - 1.0 * (0.08**2 + 0.10**2)) ** 0.5,
        description="Same mean and total volatility as base; ~1 jump/yr averaging -8%",
        jumps={"lam": 1.0, "jump_mean": -0.08, "jump_std": 0.10},
    ),
}
