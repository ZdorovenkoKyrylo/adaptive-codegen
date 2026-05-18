"""Main code generation interface."""
from typing import Dict, Optional, List
from pathlib import Path
import json

from domain.problem import OptimizationProblem, ProblemAnalysis
from codegen.core.analyzer import ProblemAnalyzer
from codegen.backends.base import BackendRegistry

# Register all built-in backends by importing them
import codegen.backends.c_embedded   # noqa: F401
import codegen.backends.c_optimized  # noqa: F401
import codegen.backends.python_numpy  # noqa: F401
import codegen.backends.cuda         # noqa: F401


class CodeGenerator:
    """
    Main interface for generating solver code from problem specifications.

    Usage
    -----
    >>> gen = CodeGenerator()
    >>> files = gen.generate(problem)                    # auto backend
    >>> files = gen.generate(problem, backend='c_embedded')
    >>> gen.save(files, './generated/my_problem')
    """

    def __init__(self):
        self.analyzer = ProblemAnalyzer()

    def generate(self, problem: OptimizationProblem,
                 backend: Optional[str] = None,
                 options: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate solver code.

        Parameters
        ----------
        problem  : OptimizationProblem
        backend  : one of 'c_embedded', 'c_optimized', 'python_numpy', 'cuda'
                   (auto-selected when None)
        options  : reserved for future backend-specific options

        Returns
        -------
        dict mapping filename → file content
        """
        analysis = self.analyzer.analyze(problem)
        if backend is None:
            backend = analysis.recommended_backend

        backend_impl = BackendRegistry.get(backend)
        files = backend_impl.generate(problem, analysis)

        files['problem_info.json'] = self._generate_metadata(problem, analysis, backend)
        files.update(self._generate_build_files(problem, analysis, backend))
        return files

    def save(self, files: Dict[str, str], output_dir: str):
        """Write all generated files to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            fp = out / filename
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)
        return out

    def list_backends(self) -> List[str]:
        return BackendRegistry.list_backends()

    # ---------------------------------------------------------------- helpers
    def _generate_metadata(self, problem, analysis, backend) -> str:
        return json.dumps({
            'problem': {
                'name': problem.name,
                'variables': analysis.total_variables,
                'constraints': analysis.total_constraints,
                'parameters': analysis.total_parameters,
            },
            'analysis': {
                'jacobian_nnz': analysis.constraint_jacobian_nnz,
                'hessian_nnz': analysis.hessian_nnz,
                'sparsity_ratio': analysis.sparsity_ratio,
                'estimated_flops_per_iteration': analysis.iteration_flops,
                'estimated_memory_bytes': analysis.memory_bytes,
            },
            'generation': {
                'backend': backend,
                'precision': analysis.recommended_precision,
                'parallelizable': analysis.parallelizable,
            },
            'configuration': {
                'max_iterations': problem.max_iterations,
                'tolerance': problem.tolerance,
                'warm_start': problem.warm_start_enabled,
                'adaptive_precision': problem.adaptive_precision,
            },
        }, indent=2)

    def _generate_build_files(self, problem, analysis, backend) -> Dict[str, str]:
        if backend.startswith('c_'):
            return {
                'Makefile':       self._generate_makefile(problem, analysis),
                'CMakeLists.txt': self._generate_cmake(problem, analysis),
            }
        return {}

    def _generate_makefile(self, problem, analysis) -> str:
        return f'''# Auto-generated Makefile for {problem.name} solver
CC     = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra

# Embedded target example:
# CC     = arm-none-eabi-gcc
# CFLAGS = -O2 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16

SRCS = {problem.name}_solver.c {problem.name}_sparse.c {problem.name}_warmstart.c
OBJS = $(SRCS:.c=.o)
LIB  = lib{problem.name}_solver.a
TEST = test_{problem.name}

all: $(LIB)

$(LIB): $(OBJS)
\tar rcs $@ $^

%.o: %.c
\t$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST)
\t./$(TEST)

$(TEST): test_{problem.name}.c $(LIB)
\t$(CC) $(CFLAGS) $< -L. -l{problem.name}_solver -lm -o $@

clean:
\trm -f $(OBJS) $(LIB) $(TEST)

.PHONY: all test clean
'''

    def _generate_cmake(self, problem, analysis) -> str:
        return f'''cmake_minimum_required(VERSION 3.10)
project({problem.name}_solver C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -O3 -march=native -ffast-math")

add_library({problem.name}_solver STATIC
    {problem.name}_solver.c
    {problem.name}_sparse.c
    {problem.name}_warmstart.c
)
target_include_directories({problem.name}_solver PUBLIC ${{CMAKE_CURRENT_SOURCE_DIR}})

add_executable(test_{problem.name} test_{problem.name}.c)
target_link_libraries(test_{problem.name} {problem.name}_solver m)

add_executable(benchmark_{problem.name} benchmark_{problem.name}.c)
target_link_libraries(benchmark_{problem.name} {problem.name}_solver m)

enable_testing()
add_test(NAME {problem.name}_test COMMAND test_{problem.name})
'''
