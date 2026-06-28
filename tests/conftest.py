"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def small_prob_map():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (64, 64)).astype(np.float32)


@pytest.fixture
def small_binary_mask():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    return mask
