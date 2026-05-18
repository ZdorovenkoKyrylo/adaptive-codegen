"""NumPy backend — generates pure-Python/NumPy solver code."""
from typing import Dict
from codegen.backends.base import CodegenBackend, BackendRegistry
from codegen.core.ast import Module
from domain.problem import OptimizationProblem, ProblemAnalysis


class NumpyBackend(CodegenBackend):
    @property
    def name(self) -> str:
        return 'python_numpy'

    @property
    def file_extension(self) -> str:
        return '.py'

    def generate(self, problem: OptimizationProblem,
                 analysis: ProblemAnalysis) -> Dict[str, str]:
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        precision = 'np.float32' if analysis.recommended_precision == 'float32' else 'np.float64'

        source = f'''"""
Auto-generated NumPy solver for: {problem.name}
Variables: {n_vars}  |  Constraints: {n_cons}
"""
import numpy as np


N_VARS = {n_vars}
N_CONS = {n_cons}
MAX_ITER = {problem.max_iterations}
TOLERANCE = {problem.tolerance}
DTYPE = {precision}


def objective(x: np.ndarray, Q: np.ndarray, c: np.ndarray) -> float:
    return float(0.5 * x @ Q @ x + c @ x)


def gradient(x: np.ndarray, Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    return Q @ x + c


def solve(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray,
          x0: np.ndarray = None, step_size: float = 0.01,
          max_iter: int = MAX_ITER, tol: float = TOLERANCE):
    """Gradient descent solver."""
    x = np.zeros(N_VARS, dtype=DTYPE) if x0 is None else x0.copy()
    for it in range(max_iter):
        x_prev = x.copy()
        g = gradient(x, Q, c)
        x -= step_size * g
        if np.max(np.abs(x - x_prev)) < tol:
            return x, it + 1, True
    return x, max_iter, False
'''
        return {f"{problem.name}_solver.py": source}

    def build_ast(self, problem: OptimizationProblem,
                  analysis: ProblemAnalysis) -> Module:
        raise NotImplementedError


BackendRegistry.register(NumpyBackend())
