from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import time
import tracemalloc
import numpy as np
import subprocess
import json


@dataclass
class BenchmarkConfig:
    problem_sizes:   List[int] = None
    obstacle_counts: List[int] = None
    num_runs:        int = 10
    warmup_runs:     int = 2
    timeout_seconds: float = 60.0

    def __post_init__(self):
        if self.problem_sizes is None:
            self.problem_sizes = [20, 50, 100, 200]
        if self.obstacle_counts is None:
            self.obstacle_counts = [5, 10, 25, 50, 100]


@dataclass
class BenchmarkResult:
    solver_name:   str
    problem_size:  int
    obstacle_count: int

    mean_runtime_ms: float
    std_runtime_ms:  float
    min_runtime_ms:  float
    max_runtime_ms:  float

    peak_memory_kb: float

    iterations:          int
    converged:           bool
    final_objective:     float
    constraint_violation: float

    flops_per_second: float = 0.0


class BenchmarkExecutor:
    """Executes benchmarks across multiple solvers and problem configurations."""

    def __init__(self, config: BenchmarkConfig = None):
        self.config  = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []

    def run_python_solver(self, solver, problem) -> BenchmarkResult:
        runtimes   = []
        peak_memory = 0

        for _ in range(self.config.warmup_runs):
            solver.solve(problem)

        for _ in range(self.config.num_runs):
            tracemalloc.start()
            t0     = time.perf_counter()
            result = solver.solve(problem)
            t1     = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            runtimes.append((t1 - t0) * 1000)
            peak_memory = max(peak_memory, peak)

        runtimes = np.array(runtimes)

        return BenchmarkResult(
            solver_name=solver.__class__.__name__,
            problem_size=len(problem.variables[0].shape),
            obstacle_count=sum(1 for p in problem.parameters if p.is_obstacle),
            mean_runtime_ms=float(np.mean(runtimes)),
            std_runtime_ms=float(np.std(runtimes)),
            min_runtime_ms=float(np.min(runtimes)),
            max_runtime_ms=float(np.max(runtimes)),
            peak_memory_kb=peak_memory / 1024,
            iterations=getattr(result, 'iterations', 0),
            converged=getattr(result, 'converged', True),
            final_objective=getattr(result, 'objective', 0.0),
            constraint_violation=0.0,
        )

    def run_c_solver(self, executable_path: str,
                     problem_data_path: str) -> BenchmarkResult:
        runtimes = []
        output   = {}

        for _ in range(self.config.num_runs):
            t0 = time.perf_counter()
            proc = subprocess.run(
                [executable_path, problem_data_path],
                capture_output=True, text=True,
                timeout=self.config.timeout_seconds,
            )
            t1 = time.perf_counter()
            runtimes.append((t1 - t0) * 1000)
            try:
                output = json.loads(proc.stdout)
            except json.JSONDecodeError:
                output = {}

        runtimes = np.array(runtimes)
        return BenchmarkResult(
            solver_name='C_Generated',
            problem_size=output.get('problem_size', 0),
            obstacle_count=output.get('obstacle_count', 0),
            mean_runtime_ms=float(np.mean(runtimes)),
            std_runtime_ms=float(np.std(runtimes)),
            min_runtime_ms=float(np.min(runtimes)),
            max_runtime_ms=float(np.max(runtimes)),
            peak_memory_kb=output.get('peak_memory_kb', 0),
            iterations=output.get('iterations', 0),
            converged=output.get('converged', False),
            final_objective=output.get('objective', 0.0),
            constraint_violation=output.get('constraint_violation', 0.0),
        )

    def run_comparison(self, solvers: Dict, problem_generator: Callable) -> List[BenchmarkResult]:
        results = []
        for n_points in self.config.problem_sizes:
            for n_obstacles in self.config.obstacle_counts:
                problem = problem_generator(n_points, n_obstacles)
                for name, solver in solvers.items():
                    try:
                        r = self.run_python_solver(solver, problem)
                        r.solver_name = name
                        results.append(r)
                    except Exception as e:
                        print(f"Solver {name} failed: {e}")
        self.results = results
        return results

    def generate_report(self) -> str:
        report = "# Benchmark Results\n\n"
        for solver in sorted(set(r.solver_name for r in self.results)):
            report += f"## {solver}\n\n"
            report += "| Problem Size | Obstacles | Runtime (ms) | Memory (KB) | Iterations |\n"
            report += "|-------------|-----------|--------------|-------------|------------|\n"
            for r in [x for x in self.results if x.solver_name == solver]:
                report += (f"| {r.problem_size} | {r.obstacle_count} | "
                           f"{r.mean_runtime_ms:.2f} ± {r.std_runtime_ms:.2f} | "
                           f"{r.peak_memory_kb:.1f} | {r.iterations} |\n")
            report += "\n"
        return report
