"""
Unit tests for odyssnet.utils.odyssstore.

Covers:
- save_checkpoint: file creation, content, extra_data
- load_checkpoint: model restoration, optimizer LR override, missing file
- transplant_weights: exact match, scale-up, scale-down, missing keys
- get_checkpoint_info: metadata extraction, missing file
"""

import pytest
import torch
import os

os.environ.setdefault("NO_BNB", "1")

from odyssnet import OdyssNet
from odyssnet.utils.odyssstore import (
    save_checkpoint,
    load_checkpoint,
    transplant_weights,
    get_checkpoint_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(n=4):
    return OdyssNet(num_neurons=n, input_ids=[0], output_ids=[n - 1], device="cpu")


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


# ===========================================================================
# save_checkpoint
# ===========================================================================

class TestSaveCheckpoint:
    def test_file_created(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "ckpt" / "test.pt")
        result = save_checkpoint(model, opt, epoch=1, loss=0.5, path=path)
        assert os.path.exists(path)

    def test_returns_path(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "ckpt" / "out.pt")
        returned = save_checkpoint(model, opt, epoch=3, loss=0.1, path=path)
        assert returned == path

    def test_checkpoint_contains_required_keys(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(model, opt, epoch=2, loss=0.25, path=path)
        ckpt = torch.load(path, map_location="cpu")
        assert "epoch" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "loss" in ckpt

    def test_epoch_and_loss_stored(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "ep.pt")
        save_checkpoint(model, opt, epoch=7, loss=1.23, path=path)
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt["epoch"] == 7
        assert ckpt["loss"] == pytest.approx(1.23)

    def test_extra_data_merged(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "extra.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path, extra_data={"tag": "v1"})
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt["tag"] == "v1"

    def test_parent_directory_created(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        deep_path = str(tmp_path / "a" / "b" / "c" / "ckpt.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=deep_path)
        assert os.path.exists(deep_path)

    def test_trainer_state_saved_and_restored(self, tmp_path):
        # convergence_skill_transfer.py passes trainer_state=trainer.state_dict()
        # to save_checkpoint, then load_checkpoint restores it via trainer=.
        os.environ["NO_BNB"] = "1"
        from odyssnet import OdyssNetTrainer

        model = _model()
        opt = _optimizer(model)
        trainer = OdyssNetTrainer(model, device="cpu", lr=1e-3)

        # Run a few steps so there is non-trivial trainer state
        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        for _ in range(3):
            trainer.train_batch(x, y, thinking_steps=2)

        path = str(tmp_path / "trainer_ckpt.pt")
        save_checkpoint(model, opt, epoch=3, loss=0.5, path=path,
                        trainer_state=trainer.state_dict())

        ckpt = torch.load(path, map_location="cpu")
        assert "trainer_state_dict" in ckpt
        assert ckpt["trainer_state_dict"]["step_count"] == trainer._step_count


# ===========================================================================
# load_checkpoint
# ===========================================================================

class TestLoadCheckpoint:
    def test_model_weights_restored(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "ckpt.pt")

        # Modify weights and save
        with torch.no_grad():
            model.W.data.fill_(0.123)
            model.W.fill_diagonal_(0.0)
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)

        # Load into fresh model
        model2 = _model()
        opt2 = _optimizer(model2)
        load_checkpoint(model2, opt2, path, device="cpu")

        assert torch.allclose(model2.W.data, model.W.data)

    def test_missing_file_raises(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, opt, str(tmp_path / "missing.pt"))

    def test_epoch_returned(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "ret.pt")
        save_checkpoint(model, opt, epoch=5, loss=0.5, path=path)
        model2 = _model()
        opt2 = _optimizer(model2)
        ckpt = load_checkpoint(model2, opt2, path)
        assert ckpt["epoch"] == 5

    def test_lr_override(self, tmp_path):
        model = _model()
        opt = _optimizer(model)  # lr=1e-3
        path = str(tmp_path / "lr.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)

        model2 = _model()
        opt2 = _optimizer(model2)
        load_checkpoint(model2, opt2, path, lr=5e-4)
        for group in opt2.param_groups:
            assert group["lr"] == pytest.approx(5e-4)

    def test_strict_false_ignores_extra_source_keys(self, tmp_path):
        """strict=False must not raise when the checkpoint has keys the target lacks."""
        # Save a model that has vocab projections
        model_with_vocab = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", vocab_size=8, vocab_mode="discrete",
        )
        opt = _optimizer(model_with_vocab)
        path = str(tmp_path / "vocab_ckpt.pt")
        save_checkpoint(model_with_vocab, opt, epoch=1, loss=0.0, path=path)

        # Load into a model without vocab — checkpoint has embed/decoder keys the
        # target doesn't; strict=False should not raise.
        model_plain = _model(n=4)
        opt2 = _optimizer(model_plain)
        load_checkpoint(model_plain, opt2, path, strict=False)

    def test_load_checkpoint_restores_trainer_state(self, tmp_path):
        # experiment_llm.py passes trainer= to load_checkpoint so that the
        # trainer's step_count, scheduler state etc. are fully restored.
        os.environ["NO_BNB"] = "1"
        from odyssnet import OdyssNetTrainer

        model = _model()
        opt = _optimizer(model)
        trainer = OdyssNetTrainer(model, device="cpu", lr=1e-3)

        x = torch.randn(4, 4)
        y = torch.randn(4, 1)
        for _ in range(4):
            trainer.train_batch(x, y, thinking_steps=2)

        path = str(tmp_path / "full_ckpt.pt")
        save_checkpoint(model, opt, epoch=4, loss=0.1, path=path,
                        trainer_state=trainer.state_dict())

        # Restore into fresh objects
        model2 = _model()
        opt2 = _optimizer(model2)
        trainer2 = OdyssNetTrainer(model2, device="cpu", lr=1e-3)
        load_checkpoint(model2, opt2, path, trainer=trainer2)

        assert trainer2._step_count == trainer._step_count


# ===========================================================================
# transplant_weights
# ===========================================================================

class TestTransplantWeights:
    def test_exact_size_match_full_copy(self, tmp_path):
        model_src = _model(n=4)
        opt = _optimizer(model_src)
        path = str(tmp_path / "src.pt")
        with torch.no_grad():
            model_src.W.data.fill_(0.5)
            model_src.W.fill_diagonal_(0.0)
        save_checkpoint(model_src, opt, epoch=1, loss=0.0, path=path)

        model_tgt = _model(n=4)
        stats = transplant_weights(model_tgt, path, device="cpu", verbose=False)

        assert stats["transplanted_params"] > 0
        # Off-diagonal W should be 0.5
        off = model_tgt.W.data.clone().fill_diagonal_(0.0)
        assert off.abs().max().item() == pytest.approx(0.5, abs=1e-5)

    def test_scale_up_preserves_overlap(self, tmp_path):
        model_small = _model(n=3)
        opt = _optimizer(model_small)
        path = str(tmp_path / "small.pt")
        with torch.no_grad():
            model_small.W.data.fill_(0.77)
            model_small.W.fill_diagonal_(0.0)
        save_checkpoint(model_small, opt, epoch=1, loss=0.0, path=path)

        model_large = _model(n=6)
        transplant_weights(model_large, path, device="cpu", verbose=False)

        # The top-left 3x3 block (off-diagonal) should be ~0.77
        block = model_large.W.data[:3, :3].clone()
        block.fill_diagonal_(0.0)
        assert block.abs().max().item() == pytest.approx(0.77, abs=1e-5)

    def test_scale_down_no_error(self, tmp_path):
        model_large = _model(n=6)
        opt = _optimizer(model_large)
        path = str(tmp_path / "large.pt")
        save_checkpoint(model_large, opt, epoch=1, loss=0.0, path=path)

        model_small = _model(n=3)
        stats = transplant_weights(model_small, path, device="cpu", verbose=False)
        assert stats["transplanted_params"] > 0

    def test_missing_file_raises(self, tmp_path):
        model = _model()
        with pytest.raises(FileNotFoundError):
            transplant_weights(model, str(tmp_path / "ghost.pt"))

    def test_stats_keys(self, tmp_path):
        model = _model(n=4)
        opt = _optimizer(model)
        path = str(tmp_path / "stat.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)
        model2 = _model(n=4)
        stats = transplant_weights(model2, path, verbose=False)
        for key in ("total_params", "transplanted_params", "new_params",
                    "keys_matched", "keys_resized", "keys_missing"):
            assert key in stats

    def test_transplant_total_equals_sum(self, tmp_path):
        model = _model(n=4)
        opt = _optimizer(model)
        path = str(tmp_path / "sum.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)
        model2 = _model(n=4)
        stats = transplant_weights(model2, path, verbose=False)
        assert stats["transplanted_params"] + stats["new_params"] == stats["total_params"]


# ===========================================================================
# get_checkpoint_info
# ===========================================================================

class TestGetCheckpointInfo:
    def test_returns_epoch_and_loss(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "info.pt")
        save_checkpoint(model, opt, epoch=11, loss=0.42, path=path)
        info = get_checkpoint_info(path)
        assert info["epoch"] == 11
        assert info["loss"] == pytest.approx(0.42)

    def test_returns_num_neurons(self, tmp_path):
        model = _model(n=6)
        opt = _optimizer(model)
        path = str(tmp_path / "nn.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)
        info = get_checkpoint_info(path)
        assert info["num_neurons"] == 6

    def test_returns_total_params(self, tmp_path):
        model = _model(n=4)
        opt = _optimizer(model)
        path = str(tmp_path / "tp.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)
        info = get_checkpoint_info(path)
        assert info["total_params"] > 0

    def test_returns_keys_list(self, tmp_path):
        model = _model()
        opt = _optimizer(model)
        path = str(tmp_path / "keys.pt")
        save_checkpoint(model, opt, epoch=1, loss=0.0, path=path)
        info = get_checkpoint_info(path)
        assert isinstance(info["keys"], list)
        assert "W" in info["keys"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_checkpoint_info(str(tmp_path / "nope.pt"))
