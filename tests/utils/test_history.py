"""
Unit tests for odyssnet.utils.history.

Covers:
- TrainingHistory: recording, retrieval, metrics listing, and plot generation.
"""

import pytest
import os
import tempfile


from odyssnet.utils.history import TrainingHistory


class TestTrainingHistory:
    def test_record_and_get(self):
        h = TrainingHistory()
        h.record(loss=0.5, lr=1e-3)
        h.record(loss=0.3, lr=1e-4)
        assert h.get("loss") == [0.5, 0.3]
        assert h.get("lr") == [1e-3, 1e-4]

    def test_get_missing_key_returns_empty(self):
        h = TrainingHistory()
        assert h.get("nonexistent") == []

    def test_metrics_lists_recorded_keys(self):
        h = TrainingHistory()
        h.record(loss=1.0, accuracy=0.9)
        assert set(h.metrics) == {"loss", "accuracy"}

    def test_empty_history_metrics(self):
        h = TrainingHistory()
        assert h.metrics == []

    def test_plot_saves_file(self):
        h = TrainingHistory()
        for i in range(10):
            h.record(loss=1.0 / (i + 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_plot.png")
            h.plot(save_path=path, title="Test")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_plot_empty_history_no_error(self, capsys):
        h = TrainingHistory()
        h.plot(title="Empty")
        captured = capsys.readouterr()
        assert "No metrics" in captured.out

    def test_record_converts_to_float(self):
        h = TrainingHistory()
        h.record(loss=1)  # int
        h.record(loss=0.5)  # float
        assert all(isinstance(v, float) for v in h.get("loss"))
