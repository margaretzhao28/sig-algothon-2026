"""Price data loading and time-series split utilities."""

from __future__ import annotations
from typing import Generator, Tuple
import os
import numpy as np
import pandas as pd

# ── Dataset selection ─────────────────────────────────────────────────────────
# Each year's data lives in its own subdirectory under DATA_ROOT
# (e.g. data/2024/prices.txt, data/2025/prices.txt). The repo-root prices.txt
# is a symlink to the active dataset, so eval.py and test_parity.py (which read
# "./prices.txt" directly) keep working unmodified — switching datasets is just
# repointing that symlink. ACTIVE_DATASET is the default for code that resolves
# paths via prices_path() instead of relying on the symlink.
DATA_ROOT: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ACTIVE_DATASET: str = "2026"


def prices_path(dataset: str | None = None) -> str:
    """Return the prices.txt path for `dataset` (default ACTIVE_DATASET)."""
    return os.path.join(DATA_ROOT, dataset or ACTIVE_DATASET, "prices.txt")


def load_prices(filepath: str = "prices.txt") -> np.ndarray:
    """Return price array of shape (nInst, nt), matching eval.py's loadPrices."""
    df = pd.read_csv(filepath, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def train_test_split(
    prices: np.ndarray,
    test_start: int,
    test_end: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split prices into train slice [:test_start] and test slice [test_start:test_end]."""
    if test_end is None:
        test_end = prices.shape[1]
    return prices[:, :test_start], prices[:, test_start:test_end]


def walk_forward_windows(
    nt: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    min_train: int | None = None,
) -> Generator[Tuple[int, int, int, int], None, None]:
    """
    Yield (train_start, train_end, test_start, test_end) index tuples for walk-forward CV.

    Parameters
    ----------
    nt        : total number of days
    train_size: number of days in each training window (0 = expanding window from 0)
    test_size : number of days in each test window
    step      : how many days to advance the window each iteration (default = test_size)
    min_train : minimum training days required before first test window (default = train_size)
    """
    if step is None:
        step = test_size
    if min_train is None:
        min_train = train_size if train_size > 0 else 1

    test_start = min_train
    while test_start + test_size <= nt:
        test_end = test_start + test_size
        if train_size > 0:
            train_start = max(0, test_start - train_size)
        else:
            train_start = 0
        yield train_start, test_start, test_start, test_end
        test_start += step
