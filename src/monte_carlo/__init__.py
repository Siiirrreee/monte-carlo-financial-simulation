"""Monte Carlo financial simulation: GBM and jump-diffusion paths, outcome
distributions, and option pricing with confidence intervals."""
from .analytics import distribution_stats, gbm_theory, max_drawdowns
from .paths import simulate_gbm, simulate_gbm_terminal, simulate_merton
from .pricing import (
    MCResult, black_scholes, geometric_asian_price, implied_vol, mc_asian,
    mc_european, mc_merton, merton_price,
)
from .scenarios import MARKET_CHECK, RISK_FREE, S0, SCENARIOS, Scenario

__all__ = [
    "simulate_gbm", "simulate_gbm_terminal", "simulate_merton",
    "distribution_stats", "gbm_theory", "max_drawdowns",
    "MCResult", "black_scholes", "implied_vol", "merton_price",
    "geometric_asian_price", "mc_european", "mc_asian", "mc_merton",
    "SCENARIOS", "Scenario", "S0", "RISK_FREE", "MARKET_CHECK",
]
