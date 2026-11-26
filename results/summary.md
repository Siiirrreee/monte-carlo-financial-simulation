_S&P 500 starting level 7,136, r = 4.2%, seed 42. All prices are risk-neutral; scenario drifts are illustrative assumptions. Full detail in [results/summary.md](results/summary.md)._

**Market check: pricing the real SPX contract from the options project**

| Contract | Market mid | Black-Scholes | Monte Carlo (95% CI) | Paths |
|:--|--:|--:|--:|--:|
| SPX Dec-18 7,200 call, σ = 23.4% | 150.35 | 150.16 | 150.19 (149.94–150.44) | 1,000,000 |

**One-year outcomes under four market conditions**

| Scenario            | μ / total σ          | Expected return   | Median return   | 5th–95th pct   | P(loss)   | 95% VaR   | 95% CVaR   | Median max drawdown   |
|:--------------------|:---------------------|:------------------|:----------------|:---------------|:----------|:----------|:-----------|:----------------------|
| Calm bull           | 10% / 12%            | 10.5%             | 9.6%            | -10% to +33%   | 22.0%     | 10.0%     | 14.4%      | 9.5%                  |
| Base case           | 7% / 18%             | 7.3%              | 5.6%            | -21% to +42%   | 38.3%     | 21.4%     | 26.9%      | 15.9%                 |
| High volatility     | 2% / 30%             | 1.9%              | -2.4%           | -40% to +60%   | 53.3%     | 40.4%     | 47.3%      | 28.3%                 |
| Crash-prone (jumps) | 7% / 18% incl. jumps | 7.3%              | 7.0%            | -23% to +38%   | 34.7%     | 23.2%     | 30.9%      | 12.9%                 |

**Convergence: 1-year 7,200 call, σ = 18%, 1,000,000 paths (Black-Scholes $627.94)**

| Estimator           |   Price |   Std. error |   Error vs BS | Variance reduction   |
|:--------------------|--------:|-------------:|--------------:|:---------------------|
| Plain               |  628.59 |        0.91  |         0.649 | 1.0×                 |
| Antithetic variates |  627.73 |        0.658 |         0.202 | 1.9×                 |
| Control variate     |  627.83 |        0.37  |         0.11  | 6.1×                 |

**Probability of losing money by holding period (μ = 7%)**

| Horizon   | P(loss), σ = 12%   | P(loss), σ = 18%   | P(loss), σ = 30%   |
|:----------|:-------------------|:-------------------|:-------------------|
| 1 yr      | 29.9%              | 38.3%              | 46.8%              |
| 5 yr      | 11.9%              | 25.2%              | 42.5%              |
| 10 yr     | 5.0%               | 17.2%              | 39.6%              |

**Key findings**

- The simulation reproduces the closed form: the Monte Carlo price of the SPX Dec-18 7,200 call is **$150.19** against Black-Scholes $150.16 and a market mid of $150.35. Across the convergence study, the 95% interval contained the exact price in **93%** of 27 runs (in line with the 95% nominal rate), and across the 36-cell volatility × maturity grid in **100%**.
- Error shrinks at the textbook 1/√n rate: 100× more paths buys only 10× more precision. Variance reduction is cheaper: the control variate cut variance **6.1×** and antithetic variates **1.9×**, equivalent to running that many times more paths.
- In the base case (μ = 7%, σ = 18%) the expected one-year return is **7.3%**, but the median is lower at 5.6%: lognormal returns are right-skewed, so the average is pulled up by a few large winners. There is a **38%** chance of ending the year down.
- The crash-prone scenario has the **same expected return and the same 18% total volatility** as the base case, yet a fatter left tail: 95% CVaR rises from 26.9% to **30.9%** and the chance of losing more than 20% goes from 6.1% to 7.0%. Ordinary years are actually calmer (P(loss) 34.7% vs 38.3%): the risk is concentrated in rare crashes. Volatility alone understates risk when moves come in jumps. The high-volatility regime is worst on every risk measure (P(loss) 53%, CVaR 47.3%).
- Time diversifies volatility only partly: at σ = 18% the chance of a loss falls from 38% over 1 year to 17% over 10, while at σ = 30% it barely moves (47% → 40%), because the volatility drag σ²/2 eats most of a 7% drift.
- An arithmetic-average Asian call, which has no closed form, prices at **$334.31** vs $627.94 for the European: averaging dampens volatility. Using the geometric Asian as a control variate cut variance **1,548×**.
- Adding jumps explains the skew found in the options project: priced under the crash-prone model, 3-month options show implied volatility of **24.2%** at the 80% strike, 16.4% at the money and 15.3% at 115%, a downward-sloping skew that a single Black-Scholes volatility cannot produce.
