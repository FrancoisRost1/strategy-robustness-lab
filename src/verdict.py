"""
Traffic light verdict — final robustness classification.

Financial rationale: individual diagnostics (PBO, DSR, plateau) each
capture a different failure mode of overfitting. The verdict combines
them into a single actionable signal that an allocator can use to decide
whether to trust a backtested strategy.

ROBUST requires ALL three conditions (low PBO + significant DSR + stable params).
OVERFIT requires ALL three failure conditions. Everything in between is graded.
"""

import numpy as np


def classify(
    pbo: float,
    dsr: float,
    plateau_fraction: float,
    config: dict,
) -> dict:
    """Produce traffic light verdict from diagnostic scores.

    Parameters
    ----------
    pbo : float
        Probability of Backtest Overfitting (0 to 1).
    dsr : float
        Deflated Sharpe Ratio probability (0 to 1).
    plateau_fraction : float or None
        Fraction of parameter grid in the plateau. None if CSV mode.
    config : dict
        Uses verdict.* thresholds.

    Returns
    -------
    dict
        - verdict: str (ROBUST / LIKELY ROBUST / BORDERLINE / LIKELY OVERFIT / OVERFIT)
        - color: str (GREEN / YELLOW / RED)
        - details: str (human-readable explanation)
        - scores: dict of input values
    """
    scores = {
        "pbo": pbo,
        "dsr": dsr,
        "plateau_fraction": plateau_fraction,
    }

    # Single-trial or missing PBO — cannot classify without CSCV
    if pbo is None or (isinstance(pbo, float) and np.isnan(pbo)):
        return {
            "verdict": "INSUFFICIENT DATA",
            "color": "GRAY",
            "details": "PBO cannot be computed (need >= 2 strategy trials for CSCV).",
            "scores": scores,
        }

    v_cfg = config.get("verdict", {})
    robust_pbo = v_cfg.get("robust_pbo_max", 0.25)
    robust_dsr = v_cfg.get("robust_dsr_min", 0.95)
    robust_plat = v_cfg.get("robust_plateau_min", 0.30)
    overfit_pbo = v_cfg.get("overfit_pbo_min", 0.50)
    overfit_dsr = v_cfg.get("overfit_dsr_max", 0.95)
    overfit_plat = v_cfg.get("overfit_plateau_max", 0.10)

    pbo_cfg = config.get("pbo", {})
    green_thresh = pbo_cfg.get("green_threshold", 0.25)
    yellow_thresh = pbo_cfg.get("yellow_threshold", 0.50)

    # Handle missing plateau (CSV mode) — use neutral value
    plat = plateau_fraction if plateau_fraction is not None else 0.20

    # Check ROBUST: all three conditions met
    if pbo < robust_pbo and dsr >= robust_dsr and plat > robust_plat:
        return {
            "verdict": "ROBUST",
            "color": "GREEN",
            "details": (
                f"PBO {pbo:.2f} — low overfitting risk\n"
                f"DSR {dsr:.2f} — statistically significant after multiple-testing\n"
                f"Plateau {plat:.2f} — robust to parameter choice"
            ),
            "scores": scores,
        }

    # Check OVERFIT: all three failure conditions met
    if pbo > overfit_pbo and dsr < overfit_dsr and plat < overfit_plat:
        return {
            "verdict": "OVERFIT",
            "color": "RED",
            "details": (
                f"PBO {pbo:.2f} — {'extremely high, strong evidence of overfitting' if pbo > 0.75 else 'above threshold, likely overfit'}\n"
                f"DSR {dsr:.2f} — {'not significant, performance likely due to noise' if dsr < 0.95 else 'significant'}\n"
                f"Plateau {plat:.2f} — {'very low, fragile parameter sensitivity' if plat < 0.10 else 'below threshold'}"
            ),
            "scores": scores,
        }

    # Graded classification based on PBO alone
    if pbo < green_thresh:
        return {
            "verdict": "LIKELY ROBUST",
            "color": "GREEN",
            "details": (
                f"PBO {pbo:.2f} — low overfitting risk\n"
                f"DSR {dsr:.2f} — {'significant' if dsr >= 0.95 else 'not significant'}\n"
                f"Plateau {plat:.2f} — {'stable' if plat > 0.30 else 'moderate' if plat > 0.10 else 'fragile'}"
            ),
            "scores": scores,
        }

    if pbo <= yellow_thresh:
        return {
            "verdict": "BORDERLINE",
            "color": "YELLOW",
            "details": (
                f"PBO {pbo:.2f} — borderline overfitting risk\n"
                f"DSR {dsr:.2f} — {'significant' if dsr >= 0.95 else 'not significant'}\n"
                f"Plateau {plat:.2f} — {'stable' if plat > 0.30 else 'moderate' if plat > 0.10 else 'fragile'}"
            ),
            "scores": scores,
        }

    # PBO > 0.90: catastrophic overfitting — stronger language
    if pbo > 0.90:
        return {
            "verdict": "LIKELY OVERFIT",
            "color": "RED",
            "details": (
                f"PBO {pbo:.2f} — catastrophic, likely noise\n"
                f"DSR {dsr:.2f} — {'significant' if dsr >= 0.95 else 'not significant, performance likely due to noise'}\n"
                f"Plateau {plat:.2f} — {'stable' if plat > 0.30 else 'moderate' if plat > 0.10 else 'fragile parameter sensitivity'}"
            ),
            "scores": scores,
        }

    # PBO > yellow_threshold but not full OVERFIT
    return {
        "verdict": "LIKELY OVERFIT",
        "color": "RED",
        "details": (
            f"PBO {pbo:.2f} — high overfitting risk\n"
            f"DSR {dsr:.2f} — {'significant' if dsr >= 0.95 else 'not significant, performance likely due to noise'}\n"
            f"Plateau {plat:.2f} — {'stable' if plat > 0.30 else 'moderate' if plat > 0.10 else 'fragile parameter sensitivity'}"
        ),
        "scores": scores,
    }
