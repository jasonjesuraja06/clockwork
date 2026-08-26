import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def seed_all():
    def _seed(seed: int = 0) -> int:
        random.seed(seed)
        import numpy

        numpy.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        return seed

    _seed()
    return _seed


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    return tmp_path
