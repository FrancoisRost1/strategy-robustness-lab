# CLAUDE.md — Strategy Robustness Lab (Project 7)

> Local source of truth for this project. Read fully before doing anything.
> Master index lives at `~/Documents/CODE/CLAUDE.md` — update it when this project hits a milestone.

---

## What this project is

A framework for testing whether backtested trading strategies are overfit to historical data.
Implements the full Bailey, Borwein, López de Prado & Zhu (2014) methodology: Probability of
Backtest Overfitting (PBO) via Combinatorial Symmetric Cross-Validation (CSCV), plus
out-of-sample degradation analysis, deflated Sharpe ratio, stochastic dominance, parameter
stability heatmaps, and bootstrap inference.

This is a **meta-strategy** project — it doesn't build a strategy, it stress-tests strategies
for robustness. It consumes output from prior projects (factor-backtest-engine, tsmom-engine)
or any generic returns series.

**Reference paper:** Bailey, Borwein, López de Prado & Zhu (2014) — "Probability of Backtest Overfitting"

---

## Status

🔲 NOT STARTED

---

## Inputs

### Option A: Built-in strategy connectors
- **Factor Engine connector** (Project 3): varies factor weights, lookback windows, rebalance frequency
- **TSMOM connector** (Project 6): varies momentum lookback, vol target, position caps

Each connector takes a parameter grid defined in `config.yaml`, runs every combination over the
same historical data, and produces a **trial matrix** — an S × T matrix where S = number of
strategy variations and T = number of time periods, each cell = daily return.

### Option B: Generic input
- **CSV upload**: user provides a pre-computed trial matrix (rows = dates, columns = strategy variations)
- **Returns series**: single strategy returns for bootstrap-only analysis

### Data source for underlying assets
- **Default**: yfinance (consistent with Projects 3, 5, 6)
- **Override**: user-supplied CSV
- **Default lookback**: 10 years (2016–2026), configurable in `config.yaml`

---

## Core methodology

### 1. Combinatorial Symmetric Cross-Validation (CSCV)

Given S partitions of the data (default S=16, configurable):
1. Split time series into S contiguous, non-overlapping blocks of equal length
2. For each combination C(S, S/2): assign S/2 blocks to in-sample (IS), S/2 to out-of-sample (OOS)
3. For each combination:
   a. Compute ranking metric (default: Sharpe ratio) for every trial on IS data
   b. Identify the IS-best trial (rank #1)
   c. Record that trial's OOS rank (its relative rank among all trials on OOS data)
4. Total combinations: C(16, 8) = 12,870 for default S=16

### 2. Probability of Backtest Overfitting (PBO)

PBO = fraction of CSCV combinations where the IS-best trial ranks below median OOS.

Formally:
```
w_c = OOS_rank(IS_best_trial) for combination c
relative_rank_c = w_c / N_trials
logit_c = ln(relative_rank_c / (1 - relative_rank_c))
PBO = proportion of logit_c < 0 (i.e., IS-best underperforms OOS median)
```

The logit transformation follows the paper — it maps the relative rank to (-∞, +∞) and the
distribution of logit values reveals how consistently the IS-best trial degrades OOS.

### 3. Out-of-Sample Degradation Analysis

For each CSCV combination:
```
degradation_ratio_c = OOS_sharpe(IS_best) / IS_sharpe(IS_best)
```

Metrics computed:
- Distribution of degradation ratios across all combinations
- Mean, median, std of degradation
- Fraction of combinations where OOS Sharpe < 0 (sign flip rate)
- IS vs OOS Sharpe scatter plot

**Bootstrap inference** (folded into this analysis):
- Standard bootstrap: resample daily returns with replacement (default 1,000 resamples)
- Block bootstrap: resample blocks of consecutive days (block size = 21 trading days default)
  to preserve autocorrelation structure
- Output: bootstrapped Sharpe distribution + 95% confidence interval
- Both applied to the overall best strategy and to the median strategy

### 4. Deflated Sharpe Ratio (DSR)

Bailey & López de Prado (2014): adjusts Sharpe ratio for multiple testing.
```
DSR = Prob(SR* > 0 | N_trials, skew, kurtosis, var(SR))

SR* = (SR_observed - SR_0) / sqrt(V(SR))
V(SR) = (1 + 0.5 * SR^2 - skew * SR + ((kurt - 3) / 4) * SR^2) / (T - 1)
SR_0 = sqrt(V(SR)) * ((1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/N * e^{-1}))
```
where gamma ≈ 0.5772 (Euler-Mascheroni), N = number of trials, T = number of observations.

If DSR < 0.95 → the Sharpe ratio is not statistically significant after accounting for
the number of strategies tried.

### 5. Stochastic Dominance

Test whether the IS-best strategy's OOS return distribution first-order stochastically
dominates a benchmark (equal-weight of all trials or a naive strategy).

Method: Kolmogorov-Smirnov test on the CDFs.
```
H0: IS_best OOS returns are drawn from same distribution as benchmark OOS returns
Reject if: KS statistic > critical value at 5% level
```

Additionally compute second-order stochastic dominance (integrated CDF comparison) for
risk-averse investor perspective.

### 6. Parameter Stability Analysis

For the built-in connectors (factor engine, TSMOM), analyze how performance varies
across the parameter grid:

**Heatmaps**: Sharpe ratio (or selected metric) across 2D parameter grid. For grids with
>2 dimensions, show all pairwise 2D slices with other parameters held at their grid-median.

**Sensitivity curves**: vary one parameter at a time, hold all others at baseline (grid-median).
Plot metric value ± 1 std (from bootstrap) vs parameter value.

**Plateau detection**: automatically flag if the strategy has a stable region.
```
plateau_fraction = (count of grid cells with metric > best_metric × (1 - tolerance)) / total_cells
```
Default tolerance = 0.10 (within 10% of best). Configurable in `config.yaml`.
- plateau_fraction > 0.30 → STABLE (robust to parameter choice)
- plateau_fraction 0.10–0.30 → MODERATE
- plateau_fraction < 0.10 → FRAGILE (likely overfit to specific parameters)

---

## Ranking metric

**Default**: Sharpe Ratio (annualized, excess of risk-free rate)
**Configurable alternatives** (set in `config.yaml`):
- Sortino Ratio
- Calmar Ratio
- Information Ratio
- Custom (any function that takes a returns series → scalar)

The selected metric is used consistently across: CSCV ranking, PBO computation, degradation
analysis, parameter heatmaps, and trial explorer.

```
sharpe = mean(excess_returns) / std(excess_returns) * sqrt(252)
sortino = mean(excess_returns) / downside_std(excess_returns) * sqrt(252)
calmar = cagr / abs(max_drawdown)
```

---

## Traffic light verdict

Based on PBO + supporting metrics, produce a final classification:

| Verdict | Color | Condition |
|---------|-------|-----------|
| ROBUST | 🟢 GREEN | PBO < 0.25 AND DSR > 0.95 AND plateau_fraction > 0.30 |
| LIKELY ROBUST | 🟢 GREEN | PBO < 0.25 |
| BORDERLINE | 🟡 YELLOW | 0.25 ≤ PBO ≤ 0.50 |
| LIKELY OVERFIT | 🔴 RED | PBO > 0.50 |
| OVERFIT | 🔴 RED | PBO > 0.50 AND DSR < 0.95 AND plateau_fraction < 0.10 |

Thresholds configurable in `config.yaml`.

---

## Built-in parameter grids

### Factor Engine (Project 3) connector

| Parameter | Grid values | Description |
|-----------|-------------|-------------|
| factor_weights | equal, value-tilt, momentum-tilt, quality-tilt | Weight scheme across 5 factors |
| lookback_months | 6, 9, 12 | Factor computation lookback |
| rebalance_freq | monthly, quarterly | Portfolio rebalance frequency |
| weighting | equal_weight, cap_weight | Portfolio weighting |
| n_quantiles | 5, 10 | Quantile sort granularity |

Default grid: 4 × 3 × 2 × 2 × 2 = **96 trials**

### TSMOM Engine (Project 6) connector

| Parameter | Grid values | Description |
|-----------|-------------|-------------|
| momentum_lookback | 63, 126, 189, 252 | Signal lookback in trading days |
| vol_target | 0.10, 0.15, 0.20 | Annualized vol target |
| position_cap | 1.5, 2.0, 3.0 | Max per-asset weight |
| gross_cap | 2.0, 3.0, 4.0 | Max portfolio gross exposure |
| rebalance_freq | monthly, biweekly | Rebalance frequency |

Default grid: 4 × 3 × 3 × 3 × 2 = **216 trials**

---

## Dashboard — 6 tabs (Streamlit, Bloomberg dark mode)

### Tab 1 — Strategy Input
- Select mode: Factor Engine / TSMOM / CSV Upload
- Parameter grid configuration (editable table or form)
- Ranking metric selector (Sharpe / Sortino / Calmar)
- CSCV partitions (S) — slider, default 16
- Lookback period selector
- "Run Analysis" button

### Tab 2 — Overview & Verdict
- Traffic light verdict (large, prominent)
- PBO score with interpretation
- Degradation ratio (mean IS→OOS decay)
- Deflated Sharpe ratio p-value
- Stochastic dominance test result
- Plateau fraction from parameter stability
- Summary table of all diagnostic scores

### Tab 3 — CSCV Analysis
- Histogram of OOS ranks for the IS-best trial across all C(S, S/2) combinations
- Logit-transformed rank distribution with PBO threshold line
- PBO convergence plot (PBO vs number of combinations evaluated)
- IS rank vs OOS rank scatter (all trials, all combinations)
- Combination-level detail table (sortable)

### Tab 4 — Degradation Analysis
- IS Sharpe vs OOS Sharpe scatter (one dot per CSCV combination)
- 45-degree line (no degradation) + regression fit
- Degradation ratio distribution histogram
- Sign flip rate bar (% of combos where OOS Sharpe < 0)
- Bootstrapped Sharpe distribution (standard + block bootstrap)
- 95% confidence interval bands
- Haircut analysis: "expect X% performance decay out-of-sample"

### Tab 5 — Parameter Stability
- 2D heatmaps of ranking metric across parameter grid (all pairwise slices)
- Sensitivity curves (one parameter varied, others at grid-median)
- Plateau detection result with visual overlay on heatmap
- Stability classification (STABLE / MODERATE / FRAGILE)
- Best region identification (which parameter ranges are in the plateau)
- Note: only available for built-in connectors (not CSV upload)

### Tab 6 — Trial Explorer
- Full performance matrix: every trial's Sharpe, Sortino, Calmar, MaxDD, CAGR
- Parameter values for each trial
- Equity curves (selectable, overlay up to 5)
- Sortable / filterable by any metric or parameter
- Highlight IS-best vs OOS-best discrepancy
- Export to CSV button

---

## File structure

```
strategy-robustness-lab/
├── CLAUDE.md
├── config.yaml
├── main.py                          # orchestrator — runs pipeline or launches dashboard
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml                  # Bloomberg dark theme
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # yfinance fetch + CSV loader + cache
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── factor_connector.py      # generates trial matrix from factor engine params
│   │   ├── tsmom_connector.py       # generates trial matrix from TSMOM params
│   │   └── csv_connector.py         # loads external trial matrix
│   ├── grid_engine.py               # parameter grid generation + sweep orchestration
│   ├── cscv.py                      # CSCV partition logic + combination generation
│   ├── pbo.py                       # PBO computation (logit model)
│   ├── metrics.py                   # Sharpe, Sortino, Calmar, CAGR, MaxDD, etc.
│   ├── degradation.py               # IS vs OOS degradation analysis
│   ├── deflated_sharpe.py           # deflated Sharpe ratio (Bailey & LdP 2014)
│   ├── stochastic_dominance.py      # KS test + 2nd order SD
│   ├── bootstrap.py                 # standard + block bootstrap
│   ├── parameter_stability.py       # heatmaps, sensitivity, plateau detection
│   ├── verdict.py                   # traffic light classification
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py         # YAML config loader (single load, pass as dict)
├── app/
│   ├── app.py                       # Streamlit entry point
│   ├── style_inject.py              # Bloomberg dark mode injection
│   ├── tab_input.py                 # Tab 1 — Strategy Input
│   ├── tab_overview.py              # Tab 2 — Overview & Verdict
│   ├── tab_cscv.py                  # Tab 3 — CSCV Analysis
│   ├── tab_degradation.py           # Tab 4 — Degradation Analysis
│   ├── tab_stability.py             # Tab 5 — Parameter Stability
│   └── tab_explorer.py              # Tab 6 — Trial Explorer
├── tests/
│   ├── __init__.py
│   ├── test_cscv.py
│   ├── test_pbo.py
│   ├── test_metrics.py
│   ├── test_degradation.py
│   ├── test_deflated_sharpe.py
│   ├── test_stochastic_dominance.py
│   ├── test_bootstrap.py
│   ├── test_parameter_stability.py
│   ├── test_verdict.py
│   ├── test_grid_engine.py
│   ├── test_connectors.py
│   └── test_integration.py          # end-to-end on synthetic data
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── docs/
│   └── analysis.md                  # investment write-up (secret sauce)
└── outputs/
```

---

## Key design decisions

1. **Trial matrix is the universal interface.** Every connector (factor, TSMOM, CSV) produces
   the same output: a DataFrame with DatetimeIndex (rows = dates) and columns = trial IDs,
   values = daily returns. All downstream analytics consume this format.

2. **CSCV operates on time blocks, not random splits.** Contiguous blocks preserve temporal
   structure (autocorrelation, regime effects). This is critical for financial data.

3. **Partition count S must be even** (required for symmetric CV). Enforce in config validation.

4. **Block bootstrap block size = 21 trading days** (≈ 1 month). Preserves monthly seasonality
   and short-term autocorrelation. Configurable.

5. **Parameter stability is only available for built-in connectors** — CSV upload doesn't have
   a parameter grid to analyze. Tab 5 shows a clear message for CSV mode.

6. **No lookahead in connectors.** Factor engine and TSMOM connectors must replicate the exact
   signal timing from the original projects (signal at t, trade at t+1).

---

## Simplifying assumptions (document inline)

- Risk-free rate = 0 for Sharpe computation (or configurable in config.yaml)
- Transaction costs not modeled in the trial matrix (strategies are compared gross)
  — rationale: PBO tests relative ranking stability, not absolute performance
- Equal-length time blocks in CSCV (if total days not divisible by S, truncate from start)
- Stochastic dominance benchmark = equal-weight average of all trials (unless overridden)

---

## Dependencies

```
pandas
numpy
scipy                  # KS test, stats functions
yfinance               # data download
pyyaml                 # config
streamlit              # dashboard
plotly                 # interactive charts
seaborn                # heatmaps (parameter stability)
matplotlib             # fallback plotting
pytest                 # testing
itertools              # combinations (stdlib)
```

---

## Cross-project reuse

| Component | Source project | Usage here |
|-----------|---------------|------------|
| Factor computation pipeline | factor-backtest-engine (P3) | Factor connector rebuilds factor signals with varied params |
| Backtesting engine | factor-backtest-engine (P3) | Reused for running factor strategy trials |
| TSMOM signal + vol targeting | tsmom-engine (P6) | TSMOM connector rebuilds signals with varied params |
| yfinance data fetch | pe-target-screener (P2) | data_loader.py pattern |
| Bloomberg dark mode | all prior projects | style_inject.py + config.toml |

---

## Edge cases to handle

- S not even → raise ValueError with clear message
- S > number of time blocks feasible (< 60 trading days per block) → warn
- Trial matrix with NaN → forward-fill then drop remaining, log count
- All trials have negative Sharpe → PBO still valid (ranking still works), note in output
- Single trial provided → skip PBO (need ≥ 2), run bootstrap only
- CSV with mismatched dates → align to common date range, warn about dropped dates
- C(S, S/2) too large for memory → configurable max_combinations with random sampling fallback
- Division by zero in Sharpe (zero std) → return np.nan, exclude from ranking

---

## Streamlit Cloud pass (2026-04-13)

- Migrated `use_container_width` → `width="stretch"/"content"` across the app layer ahead of the Streamlit deprecation-to-error window.
- Rewrote the verdict banner at `app/tab_overview.py:42` from a multi-line indented triple-quoted f-string to single-line concatenated f-strings. Leading whitespace in `st.markdown` input is parsed as a Markdown code block even when `unsafe_allow_html=True`, which was leaking `</div>` as visible text in the Streamlit Cloud render.

---

*Project CLAUDE.md — Created: 2026-04-08*
*Last updated: 2026-04-13*
