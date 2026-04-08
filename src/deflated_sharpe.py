"""
Deflated Sharpe Ratio (DSR).

Financial rationale: when N strategy variations are tested, the best
Sharpe ratio is upward-biased — even random strategies will produce
a positive best-of-N Sharpe. The DSR adjusts for this multiple-testing
bias by computing the probability that the observed Sharpe exceeds
the expected maximum Sharpe under the null (all strategies are noise).

DSR < 0.95 means the best Sharpe is not statistically significant
after accounting for the number of trials, skewness, and kurtosis.

Reference: Bailey & Lopez de Prado (2014) — "The Deflated Sharpe Ratio."
"""

import numpy as np
from scipy import stats


# Euler-Mascheroni constant
_GAMMA = 0.5772156649015329


def _expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe ratio under the null (all strategies are noise).

    Uses the approximation from Bailey & Lopez de Prado (2014), Eq. (6):
    E[max(SR)] ~ sqrt(V(SR)) * ((1-gamma) * Z_{1-1/N} + gamma * Z_{1-1/(N*e)})

    where Z_p = Phi^{-1}(p) is the standard normal quantile.
    """
    if n_trials <= 1 or var_sr <= 0:
        return 0.0

    std_sr = np.sqrt(var_sr)
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(std_sr * ((1 - _GAMMA) * z1 + _GAMMA * z2))


def _variance_of_sr(sr: float, skew: float, kurt: float, t: int) -> float:
    """Variance of the Sharpe ratio estimator.

    Accounts for non-normality of returns via skewness and excess kurtosis.
    Formula from Lo (2002), extended by Bailey & Lopez de Prado (2014).
    """
    if t <= 1:
        return np.nan
    return (1 + 0.5 * sr**2 - skew * sr + ((kurt - 3) / 4) * sr**2) / (t - 1)


def deflated_sharpe_ratio(
    observed_sr: float,
    returns: np.ndarray,
    n_trials: int,
    config: dict,
) -> dict:
    """Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    observed_sr : float
        Sharpe ratio of the best (IS-selected) strategy.
    returns : np.ndarray
        Daily returns of that strategy (for skew/kurtosis estimation).
    n_trials : int
        Total number of strategy variations tested.
    config : dict
        Full config (uses deflated_sharpe.significance_level).

    Returns
    -------
    dict
        - dsr: float — probability that SR* > 0
        - sr_zero: float — expected max SR under null
        - var_sr: float — variance of SR estimator
        - sr_star: float — test statistic
        - is_significant: bool — whether DSR exceeds significance level
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    t = len(returns)

    if t < 3 or n_trials < 1:
        return {
            "dsr": np.nan,
            "sr_zero": np.nan,
            "var_sr": np.nan,
            "sr_star": np.nan,
            "is_significant": False,
        }

    # Compute DAILY Sharpe ratio from returns — keeps all inputs on the same
    # frequency (daily) for the variance formula.  The observed_sr parameter
    # (annualised) is retained for reporting but NOT used in the z-score.
    ret_std = np.std(returns, ddof=1)
    if ret_std == 0 or np.isnan(ret_std):
        return {
            "dsr": np.nan,
            "sr_zero": np.nan,
            "var_sr": np.nan,
            "sr_star": np.nan,
            "is_significant": False,
        }
    daily_sr = float(np.mean(returns) / ret_std)

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # excess=False → raw kurtosis

    # Use daily SR with daily skew/kurtosis — all same frequency
    var_sr = _variance_of_sr(daily_sr, skew, kurt, t)
    if np.isnan(var_sr) or var_sr <= 0:
        return {
            "dsr": np.nan,
            "sr_zero": np.nan,
            "var_sr": np.nan,
            "sr_star": np.nan,
            "is_significant": False,
        }

    sr_zero = _expected_max_sr(n_trials, var_sr)

    # Test statistic: standardised distance from null expected max (daily scale)
    sr_star = (daily_sr - sr_zero) / np.sqrt(var_sr)

    # DSR = Prob(SR* > 0) = Phi(sr_star)
    dsr = float(stats.norm.cdf(sr_star))

    sig_level = config.get("deflated_sharpe", {}).get("significance_level", 0.95)

    return {
        "dsr": dsr,
        "sr_zero": float(sr_zero),
        "var_sr": float(var_sr),
        "sr_star": float(sr_star),
        "is_significant": dsr >= sig_level,
    }
