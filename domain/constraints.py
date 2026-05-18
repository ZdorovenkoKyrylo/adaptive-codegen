from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class LinearConstraint:
    """Ax = b or Ax <= b"""
    A: np.ndarray
    b: np.ndarray
    equality: bool = True

    @property
    def sparsity_pattern(self) -> np.ndarray:
        return (np.abs(self.A) > 1e-12).astype(int)

    @property
    def nnz(self) -> int:
        return np.count_nonzero(self.A)


@dataclass
class QuadraticConstraint:
    """x^T P x + q^T x + r <= 0"""
    P: np.ndarray
    q: np.ndarray
    r: float

    def linearize_at(self, x0: np.ndarray) -> LinearConstraint:
        """First-order Taylor expansion for SCvx."""
        grad = 2 * self.P @ x0 + self.q
        const = x0.T @ self.P @ x0 + self.q.T @ x0 + self.r
        return LinearConstraint(
            A=grad.reshape(1, -1),
            b=np.array([-const + grad @ x0]),
            equality=False
        )


@dataclass
class SecondOrderConeConstraint:
    """||Ax + b|| <= c^T x + d"""
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: float

    @property
    def cone_dimension(self) -> int:
        return self.A.shape[0] + 1


@dataclass
class ObstacleAvoidanceConstraint:
    """||position - obstacle_center|| >= radius"""
    position_vars: List[str]
    center_param: str
    radius_param: str

    def generate_socp(self, position_indices: List[int],
                      center: np.ndarray, radius: float,
                      n_vars: int) -> SecondOrderConeConstraint:
        A = np.zeros((len(position_indices), n_vars))
        for i, idx in enumerate(position_indices):
            A[i, idx] = 1.0

        c = np.zeros(n_vars)

        return SecondOrderConeConstraint(
            A=-A,
            b=center,
            c=c,
            d=-radius
        )
