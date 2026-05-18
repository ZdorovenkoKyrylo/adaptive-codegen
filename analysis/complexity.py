"""Computational complexity analysis helpers."""
import numpy as np


def flop_estimate(n_vars: int, n_cons: int, nnz: int) -> int:
    """Estimate FLOPs per solver iteration."""
    return 2 * nnz + int(n_vars ** 1.5) + 10 * n_cons * 10


def memory_estimate_bytes(n_vars: int, n_cons: int, nnz: int) -> int:
    B = 8  # float64
    return n_vars * B * 14 + nnz * (B + 4) + n_cons * B * 4
