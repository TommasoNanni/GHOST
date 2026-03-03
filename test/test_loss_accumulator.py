"""
Unit tests for LossAccumulator in fusion/trainer.py

No GPU, no model loading, no file I/O.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from fusion.trainer import LossAccumulator


class TestLossAccumulator:

    def test_empty_compute_returns_empty_dict(self):
        acc = LossAccumulator()
        assert acc.compute() == {}

    def test_single_update_single_key(self):
        acc = LossAccumulator()
        acc.update({"loss": 2.5})
        stats = acc.compute()
        assert "loss" in stats
        assert stats["loss"]["mean"]   == pytest.approx(2.5)
        assert stats["loss"]["median"] == pytest.approx(2.5)
        assert stats["loss"]["min"]    == pytest.approx(2.5)
        assert stats["loss"]["max"]    == pytest.approx(2.5)
        assert stats["loss"]["std"]    == pytest.approx(0.0)

    def test_multiple_updates_mean_and_extremes(self):
        acc = LossAccumulator()
        acc.update({"loss": 1.0})
        acc.update({"loss": 3.0})
        acc.update({"loss": 5.0})
        stats = acc.compute()
        assert stats["loss"]["mean"]   == pytest.approx(3.0)
        assert stats["loss"]["median"] == pytest.approx(3.0)
        assert stats["loss"]["min"]    == pytest.approx(1.0)
        assert stats["loss"]["max"]    == pytest.approx(5.0)

    def test_std_correctness(self):
        acc = LossAccumulator()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            acc.update({"x": v})
        stats = acc.compute()
        expected_std = float(np.std(values))  # population std (ddof=0)
        assert stats["x"]["std"] == pytest.approx(expected_std, abs=1e-9)

    def test_multiple_keys_independent(self):
        acc = LossAccumulator()
        acc.update({"pose": 1.0, "shape": 10.0})
        acc.update({"pose": 3.0, "shape": 20.0})
        stats = acc.compute()
        assert stats["pose"]["mean"]  == pytest.approx(2.0)
        assert stats["shape"]["mean"] == pytest.approx(15.0)

    def test_keys_updated_different_number_of_times(self):
        """Keys can have different numbers of observations."""
        acc = LossAccumulator()
        acc.update({"a": 1.0, "b": 10.0})
        acc.update({"a": 3.0})  # b not updated this step
        stats = acc.compute()
        assert stats["a"]["mean"] == pytest.approx(2.0)
        assert stats["b"]["mean"] == pytest.approx(10.0)

    def test_reset_clears_all(self):
        acc = LossAccumulator()
        acc.update({"loss": 1.0, "reg": 0.5})
        acc.reset()
        assert acc.compute() == {}

    def test_reset_then_update_fresh_state(self):
        acc = LossAccumulator()
        acc.update({"loss": 100.0})
        acc.reset()
        acc.update({"loss": 5.0})
        stats = acc.compute()
        assert stats["loss"]["mean"] == pytest.approx(5.0)

    def test_float_coercion(self):
        """update() should coerce tensors or ints to float."""
        acc = LossAccumulator()
        acc.update({"loss": 3})   # int
        stats = acc.compute()
        assert isinstance(stats["loss"]["mean"], float)

    def test_values_order_does_not_affect_mean(self):
        """Order of updates should not affect mean."""
        acc1 = LossAccumulator()
        acc2 = LossAccumulator()
        for v in [1.0, 5.0, 3.0]:
            acc1.update({"x": v})
        for v in [3.0, 1.0, 5.0]:
            acc2.update({"x": v})
        assert acc1.compute()["x"]["mean"] == pytest.approx(acc2.compute()["x"]["mean"])

    def test_median_with_even_count(self):
        acc = LossAccumulator()
        for v in [1.0, 2.0, 3.0, 4.0]:
            acc.update({"x": v})
        stats = acc.compute()
        assert stats["x"]["median"] == pytest.approx(2.5)
