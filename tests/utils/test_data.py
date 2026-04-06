"""
Unit tests for odyssnet.utils.data.

Covers:
- set_seed: reproducibility across random, numpy, and torch
- prepare_input: 2-D, 1-D, 3-D (sequential) inputs and neuron mapping
- to_tensor: type coercions, device placement, dtype handling
"""

import pytest
import random
import numpy as np
import torch
import os


from odyssnet.utils.data import set_seed, prepare_input, to_tensor


# ===========================================================================
# set_seed
# ===========================================================================

class TestSetSeed:
    def test_torch_reproducible_after_seed(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_numpy_reproducible_after_seed(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_reproducible_after_seed(self):
        set_seed(42)
        a = [random.random() for _ in range(5)]
        set_seed(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_produce_different_sequences(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.allclose(a, b)

    def test_default_seed_is_42(self):
        set_seed()
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


# ===========================================================================
# prepare_input — 2-D (standard batch)
# ===========================================================================

class TestPrepareInput2D:
    def test_output_shape(self):
        feats = torch.randn(4, 2)
        x, bs = prepare_input(feats, [0, 1], 5, "cpu")
        assert x.shape == (4, 5)
        assert bs == 4

    def test_features_placed_at_correct_neurons(self):
        feats = torch.ones(3, 2)
        x, _ = prepare_input(feats, [1, 3], 5, "cpu")
        assert torch.all(x[:, 1] == 1.0)
        assert torch.all(x[:, 3] == 1.0)
        # Other neurons should be zero
        assert torch.all(x[:, 0] == 0.0)
        assert torch.all(x[:, 2] == 0.0)
        assert torch.all(x[:, 4] == 0.0)

    def test_non_tensor_input_converted(self):
        feats = [[1.0, 2.0], [3.0, 4.0]]
        x, bs = prepare_input(feats, [0, 1], 3, "cpu")
        assert isinstance(x, torch.Tensor)
        assert bs == 2

    def test_numpy_input_converted(self):
        feats = np.ones((3, 2), dtype=np.float32)
        x, bs = prepare_input(feats, [0, 1], 4, "cpu")
        assert x.shape == (3, 4)

    def test_extra_features_clipped_to_input_ids(self):
        # More features than input_ids: only first len(input_ids) are used
        feats = torch.ones(2, 5)
        x, _ = prepare_input(feats, [0, 1], 4, "cpu")
        # Only neurons 0 and 1 receive data; neuron 2 and 3 stay 0
        assert torch.all(x[:, 0] == 1.0)
        assert torch.all(x[:, 1] == 1.0)


# ===========================================================================
# prepare_input — 1-D (auto-unsqueeze)
# ===========================================================================

class TestPrepareInput1D:
    def test_1d_input_auto_unsqueeze(self):
        feats = torch.tensor([1.0, 2.0, 3.0])
        x, bs = prepare_input(feats, [0, 1, 2], 4, "cpu")
        # 1-D treated as (3,) -> unsqueeze -> (3, 1)
        assert bs == 3
        assert x.shape == (3, 4)

    def test_1d_single_feature_placed(self):
        feats = torch.tensor([5.0, 7.0])
        x, _ = prepare_input(feats, [2], 4, "cpu")
        assert torch.all(x[:, 2] == feats.unsqueeze(1)[:, 0])


# ===========================================================================
# prepare_input — 3-D (sequential / stream)
# ===========================================================================

class TestPrepareInput3D:
    def test_3d_output_shape(self):
        feats = torch.randn(2, 5, 3)  # (batch, steps, features)
        x, bs = prepare_input(feats, [0, 1, 2], 6, "cpu")
        assert x.shape == (2, 5, 6)
        assert bs == 2

    def test_3d_features_mapped_correctly(self):
        feats = torch.ones(2, 4, 2)
        x, _ = prepare_input(feats, [0, 3], 5, "cpu")
        assert torch.all(x[:, :, 0] == 1.0)
        assert torch.all(x[:, :, 3] == 1.0)
        assert torch.all(x[:, :, 1] == 0.0)


# ===========================================================================
# to_tensor
# ===========================================================================

class TestToTensor:
    def test_list_converted_to_tensor(self):
        t = to_tensor([1.0, 2.0, 3.0], "cpu")
        assert isinstance(t, torch.Tensor)

    def test_numpy_array_converted(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = to_tensor(arr, "cpu")
        assert isinstance(t, torch.Tensor)

    def test_tensor_passthrough(self):
        x = torch.tensor([1.0, 2.0])
        t = to_tensor(x, "cpu")
        assert t is x or torch.allclose(t, x)

    def test_float64_promoted_to_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        t = to_tensor(arr, "cpu")
        assert t.dtype == torch.float32

    def test_int_dtype_preserved(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        t = to_tensor(arr, "cpu")
        assert t.dtype == torch.int64

    def test_explicit_dtype_overrides(self):
        arr = [1.0, 2.0]
        t = to_tensor(arr, "cpu", dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_device_placement(self):
        arr = [1.0, 2.0]
        t = to_tensor(arr, "cpu")
        assert t.device.type == "cpu"

    def test_tensor_moved_to_device(self):
        x = torch.tensor([1.0, 2.0])
        t = to_tensor(x, "cpu")
        assert t.device.type == "cpu"

    def test_nested_list_converted(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        t = to_tensor(data, "cpu")
        assert t.shape == (2, 2)
