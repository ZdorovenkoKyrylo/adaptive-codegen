"""Performance metric helpers."""
import numpy as np
from typing import List
from benchmark.executor import BenchmarkResult


def summary_stats(results: List[BenchmarkResult]) -> dict:
    times = [r.mean_runtime_ms for r in results]
    return {
        'mean_ms':   float(np.mean(times)),
        'median_ms': float(np.median(times)),
        'p95_ms':    float(np.percentile(times, 95)),
        'min_ms':    float(np.min(times)),
        'max_ms':    float(np.max(times)),
        'converged_pct': float(np.mean([r.converged for r in results]) * 100),
    }
