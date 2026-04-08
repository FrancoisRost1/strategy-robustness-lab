"""Tests for src/cscv.py — partition, combination generation, and CSCV run."""

import math

import numpy as np
import pandas as pd
import pytest

from src.cscv import generate_combinations, partition_blocks, run_cscv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return {
        "ranking": {"annualization_factor": 252, "metric": "sharpe"},
        "data": {"risk_free_rate": 0.0},
        "cscv": {"n_partitions": 4, "max_combinations": None, "random_seed": 42},
    }


@pytest.fixture
def trial_matrix():
    """8 trials × 100 days of random returns."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    data = {str(i): rng.normal(0, 0.01, 100) for i in range(8)}
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Partition blocks
# ---------------------------------------------------------------------------

class TestPartitionBlocks:
    def test_equal_length_blocks(self, trial_matrix):
        blocks = partition_blocks(trial_matrix, 4)
        assert len(blocks) == 4
        lengths = [len(b) for b in blocks]
        assert len(set(lengths)) == 1  # all equal

    def test_truncation_from_start(self):
        """103 rows / 4 partitions → truncate 3 leading rows → 25 per block."""
        dates = pd.bdate_range("2020-01-01", periods=103)
        df = pd.DataFrame({"a": range(103)}, index=dates)
        blocks = partition_blocks(df, 4)
        assert len(blocks) == 4
        assert all(len(b) == 25 for b in blocks)
        # First 3 rows should be truncated — first block starts at row 3
        assert blocks[0].iloc[0]["a"] == 3

    def test_too_few_rows_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        df = pd.DataFrame({"a": range(3)}, index=dates)
        with pytest.raises(ValueError, match="Not enough data"):
            partition_blocks(df, 4)

    def test_blocks_cover_all_data(self, trial_matrix):
        """After truncation, concatenated blocks = trimmed input."""
        n = len(trial_matrix)
        s = 4
        blocks = partition_blocks(trial_matrix, s)
        remainder = n % s
        trimmed = trial_matrix.iloc[remainder:]
        rebuilt = pd.concat(blocks)
        pd.testing.assert_frame_equal(rebuilt, trimmed)


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

class TestGenerateCombinations:
    def test_count_matches_formula(self):
        """C(4,2) = 6 combinations."""
        combos = generate_combinations(4)
        assert len(combos) == math.comb(4, 2)

    def test_count_s8(self):
        """C(8,4) = 70."""
        combos = generate_combinations(8)
        assert len(combos) == math.comb(8, 4)

    def test_symmetric_split(self):
        """Each combo has S/2 IS indices and S/2 OOS indices."""
        combos = generate_combinations(6)
        for is_idx, oos_idx in combos:
            assert len(is_idx) == 3
            assert len(oos_idx) == 3
            assert set(is_idx) | set(oos_idx) == set(range(6))
            assert set(is_idx) & set(oos_idx) == set()

    def test_max_combinations_sampling(self):
        """Sampling caps the number of combinations."""
        all_combos = generate_combinations(8)
        sampled = generate_combinations(8, max_combinations=10, random_seed=42)
        assert len(sampled) == 10
        assert len(sampled) < len(all_combos)

    def test_seed_reproducibility(self):
        c1 = generate_combinations(8, max_combinations=10, random_seed=99)
        c2 = generate_combinations(8, max_combinations=10, random_seed=99)
        assert c1 == c2


# ---------------------------------------------------------------------------
# Run CSCV
# ---------------------------------------------------------------------------

class TestRunCSCV:
    def test_output_shape(self, trial_matrix, config):
        results = run_cscv(trial_matrix, config)
        expected_combos = math.comb(4, 2)
        assert len(results) == expected_combos

    def test_output_columns(self, trial_matrix, config):
        results = run_cscv(trial_matrix, config)
        expected_cols = {
            "combo_id", "is_indices", "oos_indices",
            "is_best_trial", "is_best_metric", "oos_metric_of_is_best",
            "oos_rank", "n_trials",
        }
        assert set(results.columns) == expected_cols

    def test_oos_rank_bounds(self, trial_matrix, config):
        """OOS rank must be between 1 and n_trials."""
        results = run_cscv(trial_matrix, config)
        n = trial_matrix.shape[1]
        assert (results["oos_rank"] >= 1).all()
        assert (results["oos_rank"] <= n).all()

    def test_n_trials_consistent(self, trial_matrix, config):
        results = run_cscv(trial_matrix, config)
        assert (results["n_trials"] == trial_matrix.shape[1]).all()

    def test_max_combinations_limits_output(self, trial_matrix, config):
        config["cscv"]["max_combinations"] = 3
        results = run_cscv(trial_matrix, config)
        assert len(results) == 3
