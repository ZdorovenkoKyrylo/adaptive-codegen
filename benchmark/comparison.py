from typing import Dict, List
import numpy as np
from dataclasses import dataclass
from benchmark.executor import BenchmarkResult


@dataclass
class ComparisonMetrics:
    speedup_vs_baseline:           Dict[str, float]
    memory_reduction_vs_baseline:  Dict[str, float]
    iteration_reduction_vs_baseline: Dict[str, float]
    scaling_coefficients:          Dict[str, tuple]


class SolverComparison:
    """Compares performance across different solvers."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.solvers = sorted(set(r.solver_name for r in results))

    def compute_speedup(self, baseline: str = 'SCvx_ECOS') -> Dict[str, float]:
        baseline_results = [r for r in self.results if r.solver_name == baseline]
        speedups = {}
        for solver in self.solvers:
            if solver == baseline:
                continue
            solver_results = [r for r in self.results if r.solver_name == solver]
            ratios = []
            for br in baseline_results:
                for sr in solver_results:
                    if br.problem_size == sr.problem_size and br.obstacle_count == sr.obstacle_count:
                        ratios.append(br.mean_runtime_ms / sr.mean_runtime_ms)
            speedups[solver] = float(np.mean(ratios)) if ratios else 1.0
        return speedups

    def compute_scaling(self) -> Dict[str, tuple]:
        scaling = {}
        for solver in self.solvers:
            sr = [r for r in self.results if r.solver_name == solver]
            sizes = np.array([r.problem_size for r in sr])
            times = np.array([r.mean_runtime_ms for r in sr])
            if len(sizes) < 2:
                scaling[solver] = (1.0, 1.0)
                continue
            coeffs = np.polyfit(np.log(sizes + 1e-9), np.log(times + 1e-9), 1)
            scaling[solver] = (float(coeffs[0]), float(np.exp(coeffs[1])))
        return scaling

    def plot_runtime_comparison(self, output_path: str = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for solver in self.solvers:
            sr    = [r for r in self.results if r.solver_name == solver]
            sizes = [r.problem_size for r in sr]
            times = [r.mean_runtime_ms for r in sr]
            axes[0].plot(sizes, times, 'o-', label=solver)

        axes[0].set(xlabel='Problem Size (trajectory points)', ylabel='Runtime (ms)',
                    title='Runtime vs Problem Size', yscale='log')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        for solver in self.solvers:
            sr   = [r for r in self.results if r.solver_name == solver]
            obs  = [r.obstacle_count for r in sr]
            times = [r.mean_runtime_ms for r in sr]
            axes[1].plot(obs, times, 'o-', label=solver)

        axes[1].set(xlabel='Number of Obstacles', ylabel='Runtime (ms)',
                    title='Runtime vs Obstacle Count')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_memory_comparison(self, output_path: str = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        x     = np.arange(len(self.solvers))
        width = 0.25
        sizes = sorted(set(r.problem_size for r in self.results))[:3]

        for i, size in enumerate(sizes):
            mems = []
            for solver in self.solvers:
                rs = [r for r in self.results if r.solver_name == solver and r.problem_size == size]
                mems.append(float(np.mean([r.peak_memory_kb for r in rs])) if rs else 0)
            ax.bar(x + i * width, mems, width, label=f'Size {size}')

        ax.set(xlabel='Solver', ylabel='Peak Memory (KB)', title='Memory Usage Comparison')
        ax.set_xticks(x + width); ax.set_xticklabels(self.solvers, rotation=45, ha='right')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        return fig

    def generate_latex_table(self) -> str:
        scaling = self.compute_scaling()
        rows    = ''
        for solver in self.solvers:
            small = [r for r in self.results if r.solver_name == solver and r.problem_size == 40]
            large = [r for r in self.results if r.solver_name == solver and r.problem_size == 200]
            st = float(np.mean([r.mean_runtime_ms  for r in small])) if small else 0
            sm = float(np.mean([r.peak_memory_kb   for r in small])) if small else 0
            lt = float(np.mean([r.mean_runtime_ms  for r in large])) if large else 0
            lm = float(np.mean([r.peak_memory_kb   for r in large])) if large else 0
            exp = scaling.get(solver, (0, 0))[0]
            rows += (f"{solver} & {st:.1f} & {sm:.0f} & {lt:.1f} & {lm:.0f} "
                     f"& {exp:.2f} \\\\\n")

        return (
            r"\begin{table}[h]\centering"
            r"\caption{Solver Performance Comparison}"
            r"\begin{tabular}{lccccc}\toprule"
            r"Solver & \multicolumn{2}{c}{Small (40 pts)} & \multicolumn{2}{c}{Large (200 pts)} & Scaling \\"
            r" & Time (ms) & Mem (KB) & Time (ms) & Mem (KB) & Exponent \\\midrule"
            f"\n{rows}"
            r"\bottomrule\end{tabular}\end{table}"
        )
