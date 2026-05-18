from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from domain.problem import OptimizationProblem, ProblemAnalysis, ConstraintType


class ProblemAnalyzer:
    """Analyzes optimization problem structure for code generation."""

    def analyze(self, problem: OptimizationProblem) -> ProblemAnalysis:

        total_vars = sum(v.size for v in problem.variables)
        total_constraints = len(problem.constraints)
        total_params = sum(
            p.shape[0] if len(p.shape) == 1 else int(np.prod(p.shape))
            for p in problem.parameters
        )

        sparsity_info = self._analyze_sparsity(problem, total_vars, total_constraints)

        has_linear_obj = problem.objective.objective_type.name == 'LINEAR'
        has_quad_cons = any(c.constraint_type == ConstraintType.QUADRATIC
                            for c in problem.constraints)
        has_cone = any(c.constraint_type == ConstraintType.SECOND_ORDER_CONE
                       for c in problem.constraints)
        has_obstacles = any(p.is_obstacle for p in problem.parameters)

        iteration_flops = self._estimate_iteration_flops(
            total_vars, total_constraints, sparsity_info['jacobian_nnz']
        )
        memory_bytes = self._estimate_memory(total_vars, total_constraints, sparsity_info)

        precision = self._recommend_precision(total_vars, memory_bytes)
        backend = self._recommend_backend(problem, total_vars, has_obstacles)
        parallelizable = total_constraints > 50 or total_vars > 200

        return ProblemAnalysis(
            total_variables=total_vars,
            total_constraints=total_constraints,
            total_parameters=total_params,
            constraint_jacobian_nnz=sparsity_info['jacobian_nnz'],
            hessian_nnz=sparsity_info['hessian_nnz'],
            sparsity_ratio=sparsity_info['sparsity_ratio'],
            iteration_flops=iteration_flops,
            memory_bytes=memory_bytes,
            has_linear_objective=has_linear_obj,
            has_quadratic_constraints=has_quad_cons,
            has_cone_constraints=has_cone,
            has_obstacle_constraints=has_obstacles,
            recommended_precision=precision,
            recommended_backend=backend,
            parallelizable=parallelizable
        )

    def _analyze_sparsity(self, problem: OptimizationProblem,
                          n_vars: int, n_cons: int) -> Dict:
        avg_vars_per_constraint = max(2, n_vars // 10)
        jacobian_nnz = n_cons * avg_vars_per_constraint
        hessian_nnz = n_vars * 3
        dense_jacobian = n_cons * n_vars
        sparsity_ratio = 1.0 - (jacobian_nnz / max(1, dense_jacobian))

        return {
            'jacobian_nnz': jacobian_nnz,
            'hessian_nnz': hessian_nnz,
            'sparsity_ratio': sparsity_ratio
        }

    def _estimate_iteration_flops(self, n_vars: int, n_cons: int,
                                   jacobian_nnz: int) -> int:
        matvec_flops = 2 * jacobian_nnz
        solve_flops = int(n_vars ** 1.5)
        linesearch_flops = 10 * (n_cons * 10)
        return matvec_flops + solve_flops + linesearch_flops

    def _estimate_memory(self, n_vars: int, n_cons: int,
                         sparsity_info: Dict) -> int:
        bytes_per_double = 8
        var_memory = n_vars * bytes_per_double * 4
        jacobian_memory = sparsity_info['jacobian_nnz'] * (bytes_per_double + 4)
        hessian_memory = sparsity_info['hessian_nnz'] * (bytes_per_double + 4)
        work_memory = n_vars * bytes_per_double * 10
        return var_memory + jacobian_memory + hessian_memory + work_memory

    def _recommend_precision(self, n_vars: int, memory_bytes: int) -> str:
        if n_vars < 100 and memory_bytes < 1024 * 10:
            return 'float32'
        elif n_vars < 500:
            return 'float32'
        else:
            return 'float64'

    def _recommend_backend(self, problem: OptimizationProblem,
                           n_vars: int, has_obstacles: bool) -> str:
        if n_vars < 50 and not has_obstacles:
            return 'c_embedded'
        elif n_vars < 200:
            return 'c_optimized'
        elif n_vars > 1000:
            return 'cuda'
        else:
            return 'c_optimized'
