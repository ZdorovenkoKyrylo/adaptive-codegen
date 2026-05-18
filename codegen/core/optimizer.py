"""Code optimization passes (placeholder for future extension)."""


class OptimizationPass:
    """Base class for AST optimization passes."""

    def apply(self, module):
        raise NotImplementedError


class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at code-gen time."""

    def apply(self, module):
        # Future: walk AST and fold BinaryOp(Literal, op, Literal) → Literal
        return module


class DeadCodeEliminationPass(OptimizationPass):
    """Remove unreachable code paths."""

    def apply(self, module):
        return module


class Optimizer:
    def __init__(self, passes=None):
        self.passes = passes or [ConstantFoldingPass(), DeadCodeEliminationPass()]

    def optimize(self, module):
        for p in self.passes:
            module = p.apply(module)
        return module
