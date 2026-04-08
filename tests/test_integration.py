"""End-to-end integration test on synthetic data.

Generates a synthetic trial matrix, runs the full CSCV → PBO → degradation
→ DSR → bootstrap → stochastic dominance → verdict pipeline, and verifies
all expected keys are present and types are correct.
"""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import run_bootstrap
from src.cscv import run_cscv
from src.deflated_sharpe import deflated_sharpe_ratio
from src.degradation import compute_degradation, haircut_summary
from src.metrics import compute_all_metrics, compute_metric, sharpe_ratio
from src.parameter_stability import build_metric_grid, plateau_detection
from src.pbo import compute_pbo, pbo_convergence
from src.stochastic_dominance import run_stochastic_dominance
from src.utils.config_loader import load_config
from src.verdict import classify


@pytest.fixture
def config():
    cfg = load_config()
    # Override for fast tests
    cfg["cscv"]["n_partitions"] = 4
    cfg["cscv"]["max_combinations"] = None
    cfg["bootstrap"]["n_resamples"] = 50
    return cfg


@pytest.fixture
def synthetic_trial_matrix():
    """20 noise strategies + trial '0' with planted signal."""
    rng = np.random.RandomState(42)
    n_trials, n_days = 20, 252
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    data = {}
    for i in range(n_trials):
        alpha = 0.0005 if i == 0 else 0.0
        data[str(i)] = rng.normal(alpha, 0.01, n_days)
    return pd.DataFrame(data, index=dates)


class TestFullPipeline:
    def test_pipeline_completes(self, synthetic_trial_matrix, config):
        """Full pipeline runs without error and returns all expected keys."""
        tm = synthetic_trial_matrix

        # Step 1: CSCV
        cscv_results = run_cscv(tm, config)
        assert len(cscv_results) > 0

        # Step 2: PBO
        pbo_result = compute_pbo(cscv_results)
        assert 0.0 <= pbo_result["pbo"] <= 1.0

        # Step 3: Convergence
        conv = pbo_convergence(cscv_results, step=2)
        assert len(conv) > 0

        # Step 4: Degradation
        degrad = compute_degradation(cscv_results)
        assert "mean_degradation" in degrad
        assert "sign_flip_rate" in degrad
        haircut = haircut_summary(degrad)
        assert isinstance(haircut, str)

        # Step 5: IS-best
        full_metrics = {col: compute_metric(tm[col], config) for col in tm.columns}
        is_best = max(full_metrics, key=lambda k: full_metrics[k] if not np.isnan(full_metrics[k]) else -np.inf)
        is_best_returns = tm[is_best]

        # Step 6: DSR
        observed_sr = sharpe_ratio(is_best_returns, config)
        dsr = deflated_sharpe_ratio(observed_sr, is_best_returns.values, 20, config)
        assert "dsr" in dsr
        assert isinstance(dsr["is_significant"], bool)

        # Step 7: Bootstrap
        boot = run_bootstrap(is_best_returns, config)
        assert "standard" in boot
        assert "block" in boot
        assert len(boot["standard"]["metric_distribution"]) == 50

        # Step 8: Stochastic dominance
        benchmark = tm.mean(axis=1).values
        sd = run_stochastic_dominance(is_best_returns.values, benchmark, config)
        assert "first_order" in sd

        # Step 9: Parameter stability
        param_grid = [{"trial_id": i, "p1": i % 4, "p2": i % 5} for i in range(20)]
        mg = build_metric_grid(param_grid, full_metrics)
        plateau = plateau_detection(mg, config)
        assert "plateau_fraction" in plateau

        # Step 10: Verdict
        v = classify(
            pbo_result["pbo"],
            dsr["dsr"],
            plateau["plateau_fraction"],
            config,
        )
        assert v["verdict"] in {"ROBUST", "LIKELY ROBUST", "BORDERLINE", "LIKELY OVERFIT", "OVERFIT", "INSUFFICIENT DATA"}
        assert v["color"] in {"GREEN", "YELLOW", "RED", "GRAY"}

    def test_all_trial_metrics(self, synthetic_trial_matrix, config):
        """compute_all_metrics returns valid dicts for every trial."""
        for col in synthetic_trial_matrix.columns:
            m = compute_all_metrics(synthetic_trial_matrix[col], config)
            assert "sharpe" in m
            assert "max_drawdown" in m
            assert isinstance(m["sharpe"], float)

    def test_two_trial_minimum(self, config):
        """Pipeline works with exactly 2 trials (minimum for PBO)."""
        rng = np.random.RandomState(99)
        dates = pd.bdate_range("2020-01-01", periods=100)
        tm = pd.DataFrame({
            "0": rng.normal(0, 0.01, 100),
            "1": rng.normal(0, 0.01, 100),
        }, index=dates)

        cscv = run_cscv(tm, config)
        pbo = compute_pbo(cscv)
        assert 0.0 <= pbo["pbo"] <= 1.0
