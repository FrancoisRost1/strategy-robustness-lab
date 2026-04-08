# Strategy Robustness Lab — Investment Analysis

## The Overfitting Problem in Quantitative Finance

Every systematic strategy is born from a backtest. The researcher tests hundreds or thousands of parameter combinations — lookback windows, rebalance frequencies, weighting schemes, signal thresholds — and reports the best one. The pitch deck shows a Sharpe of 2.0 and a smooth equity curve.

The problem: **when you test enough combinations, at least one will look great by pure chance.** This is selection bias at scale. The backtest isn't wrong — the strategy really did produce those returns on that data. But the performance was driven by noise, not signal, and it will not persist out-of-sample.

This is not a theoretical concern. Lopez de Prado (2018) estimates that the majority of published backtested strategies are overfit. White (2000) showed that standard train/test splits severely underestimate overfitting when many strategies are evaluated. The standard holdout approach gives you a single point estimate of out-of-sample quality — one unlucky split and you either reject a good strategy or accept a bad one.

**The core question this tool answers:** Given N strategy variations tested on the same data, what is the probability that the best one is overfit?

---

## How PBO Works

### Why Simple Train/Test Fails

A single 70/30 train/test split gives you one data point: "the IS-best strategy had a Sharpe of X out-of-sample." You don't know if that result is robust to the specific split point. Was the OOS period unusually favorable? What if you split at a different date?

### Combinatorial Symmetric Cross-Validation (CSCV)

CSCV solves this by generating **thousands of IS/OOS splits** from the same data:

1. Split the time series into S contiguous blocks (e.g., S=16, each block ~157 trading days)
2. For each of the C(16,8) = 12,870 ways to assign 8 blocks to IS and 8 to OOS:
   - Rank all N strategies by their IS performance
   - Record where the IS-best strategy ranks OOS

The result is a **distribution of OOS ranks** for the in-sample winner. If the IS-best consistently ranks near the top OOS, the strategy is genuinely robust. If it scatters randomly or clusters at the bottom, it's overfit.

### The PBO Statistic

PBO = the fraction of CSCV combinations where the IS-best strategy ranks in the bottom half OOS.

- **PBO < 0.25** — The IS-best consistently performs well OOS. Low overfitting risk.
- **PBO 0.25–0.50** — Borderline. The strategy may be partially overfit.
- **PBO > 0.50** — The IS-best is more likely to underperform than outperform OOS. High overfitting probability.

The logit transformation maps relative OOS rank to (-inf, +inf), making the distribution more interpretable and amenable to statistical analysis.

---

## Parameter Stability as a Second Line of Defense

PBO tells you whether the **best** configuration is overfit. Parameter stability tells you whether the strategy has a **broad plateau of good performance** or a narrow peak.

A robust strategy works well across a wide range of parameters — it captures a real economic signal that isn't sensitive to the exact lookback or rebalance frequency. An overfit strategy performs well only at one specific parameter combination.

**Plateau detection** measures what fraction of the parameter grid produces performance within 10% of the best. A fraction above 30% indicates stability; below 10% suggests fragility.

**Sensitivity curves** vary one parameter at a time while holding others at baseline, showing whether performance degrades smoothly or collapses at specific values.

For an allocator, parameter stability answers: "If the real-world dynamics shift slightly from the backtest period, will this strategy still work?"

---

## Deflated Sharpe Ratio — Why Most Published Sharpes Are Inflated

When a researcher tests 500 strategy variants and reports the best Sharpe, the expected maximum Sharpe under pure noise is already well above zero. This is the **multiple testing problem**.

The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) adjusts for:

1. **Number of trials tested** — more trials = higher bias
2. **Non-normality of returns** — skewness and kurtosis affect the sampling distribution of the Sharpe estimator
3. **Sample length** — shorter samples have more estimation error

DSR = P(SR* > 0), where SR* is the Sharpe ratio adjusted for the expected maximum under the null. A DSR below 0.95 means the observed Sharpe is not statistically significant after accounting for data snooping.

**Practical interpretation:** If a fund reports a Sharpe of 1.5 but tested 200 parameter combinations, the DSR might be 0.60 — meaning there's a 40% chance that Sharpe is entirely explained by selection bias.

---

## Real-World Application

### How a Portfolio Manager Would Use This

1. **Strategy development:** After building a factor model or momentum strategy, run the full parameter grid through the robustness lab before committing capital. A PBO > 0.50 means the strategy needs fundamental rethinking, not parameter tuning.

2. **Manager due diligence:** When evaluating an external manager's track record, ask how many strategy variants they tested. If they can provide the trial matrix, PBO quantifies the overfitting risk directly.

3. **Position sizing:** Even for strategies that pass PBO, the degradation analysis provides a "haircut" — the expected performance decay from backtest to live. Use this to set realistic return expectations: if median degradation is 0.60, budget for 40% less than the backtest shows.

4. **Ongoing monitoring:** Rerun the analysis periodically as new data arrives. A strategy that was robust in 2020 may become borderline in 2024 if market dynamics shift.

### How an Allocator Would Use This

The traffic light verdict provides a simple decision framework:

- **ROBUST (GREEN):** Allocate with confidence. PBO < 0.25, DSR significant, stable parameter plateau.
- **LIKELY ROBUST (GREEN):** Allocate but monitor. PBO is low but supporting metrics are mixed.
- **BORDERLINE (YELLOW):** Reduce allocation or require additional evidence.
- **LIKELY OVERFIT (RED):** Do not allocate based on backtest alone. Require live track record.
- **OVERFIT (RED):** Reject. The backtest is almost certainly noise.

---

## Limitations and Assumptions

### What This Tool Cannot Do

- **It cannot prove a strategy is NOT overfit.** Low PBO means the strategy is robust to data partitioning, not that it will perform well in the future. Regime changes, structural breaks, and implementation costs are separate risks.
- **It cannot detect lookahead bias in the input data.** If the trial matrix was generated with future information, PBO will validate a flawed strategy. Garbage in, garbage out.
- **It assumes returns are the right metric.** For strategies with complex risk profiles (options, tail hedging), return-based PBO may miss important dynamics.

### Simplifying Assumptions

- **Risk-free rate = 0** for Sharpe computation (configurable in config.yaml)
- **Transaction costs not modeled** in the trial matrix — strategies are compared gross. Rationale: PBO tests relative ranking stability, not absolute performance. A strategy that ranks well IS and OOS will continue to rank well after costs.
- **Equal-length time blocks** in CSCV — if total days aren't divisible by S, leading days are truncated. This discards a small amount of data from the earliest period.
- **Stochastic dominance benchmark** = equal-weight average of all trials (configurable)

### When PBO Gives Misleading Results

- **Very few trials (N < 10):** PBO is mechanically unstable. Use bootstrap inference instead.
- **Very short data (< 3 years):** Blocks become too small for reliable metric estimation within each CSCV partition.
- **Highly correlated trials:** If all strategy variants produce nearly identical returns (e.g., varying a parameter that doesn't matter), PBO will be near 0.50 by construction — not because of overfitting, but because ranking is noisy.
- **Non-stationary data:** CSCV assumes the data-generating process is roughly stationary across blocks. If there's a structural break in the middle, IS and OOS will come from different regimes, inflating PBO artificially.

---

*Analysis by Francois Rostaing — April 2026*
*Part of the Finance Lab portfolio: github.com/FrancoisRost1*
