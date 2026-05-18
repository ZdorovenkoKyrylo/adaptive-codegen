"""
Complete example: generate and benchmark trajectory optimization solvers.

Run from the project root:
    python -m examples.trajectory_planning.main
or
    python examples/trajectory_planning/main.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from pathlib import Path

from domain.geometry import Workspace, TrajectorySpec, Obstacle
from codegen.generator import CodeGenerator
from codegen.transforms.trajectory import TrajectoryProblemGenerator
from benchmark.executor import BenchmarkExecutor, BenchmarkConfig, BenchmarkResult
from benchmark.comparison import SolverComparison


# ---------------------------------------------------------------------------
def create_test_workspace(n_obstacles: int) -> Workspace:
    np.random.seed(42)
    obstacles = [
        Obstacle(center=np.random.uniform(1, 9, size=2),
                 radius=np.random.uniform(0.3, 0.8))
        for _ in range(n_obstacles)
    ]
    return Workspace(
        bounds_min=np.array([0.0, 0.0]),
        bounds_max=np.array([10.0, 10.0]),
        obstacles=obstacles,
    )


def create_trajectory_spec(n_points: int) -> TrajectorySpec:
    return TrajectorySpec(
        n_points=n_points,
        state_dim=4,
        control_dim=2,
        dt=0.1,
        start_state=np.array([0.5, 0.5, 0.0, 0.0]),
        goal_state=np.array([9.5, 9.5, 0.0, 0.0]),
        state_bounds=(np.array([0.0, 0.0, -5.0, -5.0]),
                      np.array([10.0, 10.0, 5.0, 5.0])),
        control_bounds=(np.array([-3.0, -3.0]),
                        np.array([3.0, 3.0])),
    )


def generate_simulated_benchmarks(config: BenchmarkConfig):
    solvers = {
        'CVXGEN_style':   {'base_time': 8,  'time_exp': 1.5, 'base_mem': 50},
        'SCvx_ECOS':      {'base_time': 18, 'time_exp': 1.8, 'base_mem': 120},
        'Adaptive_Hybrid':{'base_time': 6,  'time_exp': 1.3, 'base_mem': 35},
    }
    results = []
    for size in config.problem_sizes:
        for obs in config.obstacle_counts:
            for name, p in solvers.items():
                tf = (size / 20) ** p['time_exp']
                of = 1 + 0.02 * obs
                rt = p['base_time'] * tf * of
                mem = p['base_mem'] * (size / 20) * (1 + 0.01 * obs)
                results.append(BenchmarkResult(
                    solver_name=name, problem_size=size, obstacle_count=obs,
                    mean_runtime_ms=rt, std_runtime_ms=rt * 0.1,
                    min_runtime_ms=rt * 0.9, max_runtime_ms=rt * 1.1,
                    peak_memory_kb=mem, iterations=int(20 + size * 0.3),
                    converged=True, final_objective=100.0, constraint_violation=1e-6,
                ))
    return results


# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Adaptive Embedded Convex Optimization Code Generator")
    print("=" * 60)

    output_dir = Path("generated")
    n_points   = 100
    n_obstacles = 8

    # 1. Create problem
    print("\n[1] Creating problem specification...")
    workspace   = create_test_workspace(n_obstacles)
    spec        = create_trajectory_spec(n_points)
    problem_gen = TrajectoryProblemGenerator()
    problem     = problem_gen.generate(spec, workspace)

    total_vars = sum(v.size for v in problem.variables)
    print(f"    Problem: {problem.name}")
    print(f"    Variables: {total_vars}")
    print(f"    Constraints: {len(problem.constraints)}")
    print(f"    Obstacles: {n_obstacles}")

    # 2. Analyse
    print("\n[2] Analysing problem structure...")
    generator = CodeGenerator()
    analysis  = generator.analyzer.analyze(problem)
    print(f"    Jacobian NNZ:        {analysis.constraint_jacobian_nnz}")
    print(f"    Sparsity ratio:      {analysis.sparsity_ratio:.2%}")
    print(f"    Estimated memory:    {analysis.memory_bytes / 1024:.1f} KB")
    print(f"    Recommended backend: {analysis.recommended_backend}")
    print(f"    Recommended prec:    {analysis.recommended_precision}")

    # 3. Generate code
    print("\n[3] Generating solver code...")
    for backend in ['c_embedded', 'c_optimized']:
        print(f"\n    Backend: {backend}")
        files = generator.generate(problem, backend=backend)
        generator.save(files, str(output_dir / backend))
        print(f"    Generated {len(files)} files:")
        for fn in sorted(files):
            print(f"      {fn}  ({len(files[fn])} bytes)")

    # 4. Show header snippet
    print("\n[4] Header snippet (c_embedded):")
    print("-" * 40)
    emb_files = generator.generate(problem, backend='c_embedded')
    hdr = emb_files.get(f"{problem.name}_solver.h", "")
    print(hdr[:800] + "\n  [... truncated ...]")

    # 5. Benchmarks
    print("\n[5] Running simulated benchmarks...")
    config  = BenchmarkConfig(problem_sizes=[20, 40, 80], obstacle_counts=[5, 10, 25], num_runs=5)
    sim_res = generate_simulated_benchmarks(config)
    cmp     = SolverComparison(sim_res)

    speedups = cmp.compute_speedup(baseline='SCvx_ECOS')
    print("\n    Speedup vs SCvx+ECOS:")
    for solver, sp in speedups.items():
        print(f"      {solver}: {sp:.2f}x")

    scaling = cmp.compute_scaling()
    print("\n    Complexity scaling (time ∝ n^exp):")
    for solver, (exp, _) in scaling.items():
        print(f"      {solver}: O(n^{exp:.2f})")

    # 6. Persist reports
    print("\n[6] Saving reports...")
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'benchmark_report.md').write_text(
        BenchmarkExecutor(config).generate_report()
        if False  # executor.results is empty without real solvers
        else "# Benchmark\n\nSee console output (simulated results).\n"
    )
    print(f"    Saved to: {output_dir.absolute()}")

    print("\n" + "=" * 60)
    print("Done!  Generated files are in:", output_dir.absolute())
    print("=" * 60)


if __name__ == '__main__':
    main()
