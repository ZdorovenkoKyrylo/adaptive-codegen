from abc import ABC, abstractmethod
from typing import Dict, Any
from domain.problem import OptimizationProblem, ProblemAnalysis
from codegen.core.ast import Module


class CodegenBackend(ABC):
    """Abstract base class for code generation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        pass

    @abstractmethod
    def generate(self, problem: OptimizationProblem,
                 analysis: ProblemAnalysis) -> Dict[str, str]:
        pass

    @abstractmethod
    def build_ast(self, problem: OptimizationProblem,
                  analysis: ProblemAnalysis) -> Module:
        pass

    def emit(self, module: Module) -> str:
        raise NotImplementedError


class BackendRegistry:
    """Registry for code generation backends."""

    _backends: Dict[str, 'CodegenBackend'] = {}

    @classmethod
    def register(cls, backend: 'CodegenBackend'):
        cls._backends[backend.name] = backend

    @classmethod
    def get(cls, name: str) -> 'CodegenBackend':
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: '{name}'. Available: {list(cls._backends.keys())}")
        return cls._backends[name]

    @classmethod
    def list_backends(cls) -> list:
        return list(cls._backends.keys())
