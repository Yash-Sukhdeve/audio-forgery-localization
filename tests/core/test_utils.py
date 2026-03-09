import torch
import numpy as np
from core.utils.seed import set_seed


class TestSetSeed:
    def test_torch_deterministic(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_numpy_deterministic(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        assert np.array_equal(a, b)
