from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum, auto
import numpy as np


class VariableType(Enum):
    CONTINUOUS = auto()
    INTEGER = auto()
    BINARY = auto()


class ConstraintType(Enum):
    LINEAR_EQUALITY = auto()
    LINEAR_INEQUALITY = auto()
    QUADRATIC = auto()
    SECOND_ORDER_CONE = auto()
    NONLINEAR_CONVEXIFIED = auto()


class ObjectiveType(Enum):
    QUADRATIC = auto()
    LINEAR = auto()
    CUSTOM = auto()


@dataclass
class Variable:
    name: str
    shape: tuple
    dtype: VariableType = VariableType.CONTINUOUS
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))


@dataclass
class Parameter:
    name: str
    shape: tuple
    default_value: Optional[np.ndarray] = None
    is_obstacle: bool = False


@dataclass
class Constraint:
    name: str
    constraint_type: ConstraintType
    variables: List[str]
    parameters: List[str]
    expression: str
    jacobian: Optional[str] = None
    hessian: Optional[str] = None


@dataclass
class Objective:
    objective_type: ObjectiveType
    expression: str
    gradient: Optional[str] = None
    hessian: Optional[str] = None


@dataclass
class OptimizationProblem:
    name: str
    variables: List[Variable]
    parameters: List[Parameter]
    constraints: List[Constraint]
    objective: Objective

    max_iterations: int = 100
    tolerance: float = 1e-6
    warm_start_enabled: bool = True
    adaptive_precision: bool = True

    is_qp: bool = False
    is_socp: bool = False
    sparsity_pattern: Optional[np.ndarray] = None

    def analyze(self) -> 'ProblemAnalysis':
        from codegen.core.analyzer import ProblemAnalyzer
        return ProblemAnalyzer().analyze(self)


@dataclass
class ProblemAnalysis:
    total_variables: int
    total_constraints: int
    total_parameters: int

    constraint_jacobian_nnz: int
    hessian_nnz: int
    sparsity_ratio: float

    iteration_flops: int
    memory_bytes: int

    has_linear_objective: bool
    has_quadratic_constraints: bool
    has_cone_constraints: bool
    has_obstacle_constraints: bool

    recommended_precision: str
    recommended_backend: str
    parallelizable: bool
