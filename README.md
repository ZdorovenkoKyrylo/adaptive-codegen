# adaptive-codegen

**Adaptive Embedded Convex Optimization Code Generator**

Transforms high-level optimization problem descriptions into production-ready solver code tailored for the target platform — from bare-metal microcontrollers to CUDA GPUs.

---

## Features

| Feature | Details |
|---|---|
| **Multi-backend codegen** | `c_embedded`, `c_optimized`, `python_numpy`, `cuda` |
| **Problem DSL** | Declarative Python API for variables, constraints, objectives |
| **Auto-analysis** | Sparsity detection, complexity estimation, backend recommendation |
| **Warm-start cache** | Ring-buffer with linear extrapolation — fewer iterations on hot paths |
| **Sparse CSR Jacobian** | CSR-format constraint matrix, ~50 % faster matrix ops |
| **Adaptive step size** | Backtracking line search with Armijo-like adaptation |
| **Build system output** | Auto-generated `Makefile` + `CMakeLists.txt` |
| **Benchmark suite** | Multi-solver comparison with scaling analysis and LaTeX tables |

---

## Project Structure

```
adaptive-codegen/
├── codegen/
│   ├── core/
│   │   ├── ast.py          # Code AST representation
│   │   ├── analyzer.py     # Problem structure analysis
│   │   ├── optimizer.py    # AST optimization passes
│   │   └── registry.py     # Backend registry re-export
│   ├── backends/
│   │   ├── base.py         # Abstract backend + registry
│   │   ├── c_embedded.py   # Bare-metal C (no malloc)
│   │   ├── c_optimized.py  # Optimized C with sparse ops & warm-start
│   │   ├── python_numpy.py # Pure NumPy reference
│   │   └── cuda.py         # CUDA kernel skeleton
│   ├── transforms/
│   │   └── trajectory.py   # Trajectory → OptimizationProblem
│   └── generator.py        # Main CodeGenerator interface
├── domain/
│   ├── problem.py          # Problem DSL (Variable, Constraint, Objective …)
│   ├── constraints.py      # Linear / Quadratic / SOC / Obstacle constraints
│   └── geometry.py         # Workspace, Obstacle, TrajectorySpec
├── analysis/
│   └── complexity.py       # FLOP & memory estimators
├── benchmark/
│   ├── executor.py         # BenchmarkExecutor
│   ├── metrics.py          # Summary statistics
│   └── comparison.py       # SolverComparison (plots, LaTeX tables)
├── validation/             # Verifier / fuzzer stubs
├── examples/
│   └── trajectory_planning/
│       └── main.py         # End-to-end demo
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/adaptive-codegen.git
cd adaptive-codegen
pip install -e ".[plots]"   # adds matplotlib for benchmark charts
```

### 2. Run the built-in example

```bash
python -m examples.trajectory_planning.main
```

This will:
- Build a 50-point, 10-obstacle trajectory problem
- Analyse sparsity and recommend a backend
- Generate C solver files in `generated/c_embedded/` and `generated/c_optimized/`
- Print simulated benchmark comparisons

### 3. Quick API usage

```python
import numpy as np
from domain.problem import OptimizationProblem, Variable, Parameter, Constraint, Objective
from domain.problem import VariableType, ConstraintType, ObjectiveType
from codegen.generator import CodeGenerator

# Define a tiny QP: min 0.5 x^T Q x + c^T x  s.t. Ax = b
problem = OptimizationProblem(
    name='my_qp',
    variables=[Variable('x', shape=(4,))],
    parameters=[],
    constraints=[
        Constraint('eq', ConstraintType.LINEAR_EQUALITY,
                   variables=['x'], parameters=[],
                   expression='A @ x - b'),
    ],
    objective=Objective(ObjectiveType.QUADRATIC, '0.5 * x.T @ Q @ x + c @ x'),
    max_iterations=200,
    tolerance=1e-7,
)

gen   = CodeGenerator()
files = gen.generate(problem, backend='c_embedded')
gen.save(files, './output/my_qp')

for name in files:
    print(name)
```

### 4. Trajectory planning from spec

```python
from domain.geometry import Workspace, TrajectorySpec, Obstacle
from codegen.transforms.trajectory import TrajectoryProblemGenerator
from codegen.generator import CodeGenerator

workspace = Workspace(
    bounds_min=np.array([0., 0.]),
    bounds_max=np.array([10., 10.]),
    obstacles=[Obstacle(center=np.array([5., 5.]), radius=1.0)],
)

spec = TrajectorySpec(
    n_points=30, state_dim=4, control_dim=2, dt=0.1,
    start_state=np.array([0.5, 0.5, 0., 0.]),
    goal_state=np.array([9.5, 9.5, 0., 0.]),
    state_bounds=(np.array([0., 0., -5., -5.]), np.array([10., 10., 5., 5.])),
    control_bounds=(np.array([-3., -3.]), np.array([3., 3.])),
)

problem = TrajectoryProblemGenerator().generate(spec, workspace)
gen     = CodeGenerator()

# Auto-select backend based on problem size
files = gen.generate(problem)
gen.save(files, './output/trajectory')
```

---

## Available Backends

| Backend | Flag | Best for |
|---|---|---|
| Embedded C | `c_embedded` | MCUs, no dynamic alloc, < 50 vars |
| Optimised C | `c_optimized` | Workstations / Linux embedded, sparse ops |
| NumPy | `python_numpy` | Rapid prototyping, unit tests |
| CUDA | `cuda` | GPU acceleration (skeleton — extend as needed) |

```python
gen.list_backends()
# ['c_embedded', 'c_optimized', 'python_numpy', 'cuda']
```

---

## Compiling Generated C Code

After generation a `Makefile` and `CMakeLists.txt` are included.

**Make:**
```bash
cd output/trajectory
make
./test_trajectory    # runs bundled test harness
```

**CMake:**
```bash
cd output/trajectory
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

**Cross-compile for ARM Cortex-M4** (edit `Makefile`):
```makefile
CC     = arm-none-eabi-gcc
CFLAGS = -O2 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard
```

---

## Benchmark Framework

```python
from benchmark.executor import BenchmarkExecutor, BenchmarkConfig
from benchmark.comparison import SolverComparison

config  = BenchmarkConfig(problem_sizes=[20, 50, 100], obstacle_counts=[5, 20])
results = [...]   # list of BenchmarkResult (from run_python_solver / run_c_solver)

cmp = SolverComparison(results)
print(cmp.compute_speedup(baseline='SCvx_ECOS'))
cmp.plot_runtime_comparison('runtime.png')
print(cmp.generate_latex_table())
```

---

## Adding a Custom Backend

1. Create `codegen/backends/my_backend.py`:

```python
from codegen.backends.base import CodegenBackend, BackendRegistry
from codegen.core.ast import Module
from domain.problem import OptimizationProblem, ProblemAnalysis
from typing import Dict

class MyBackend(CodegenBackend):
    @property
    def name(self): return 'my_backend'

    @property
    def file_extension(self): return '.c'

    def generate(self, problem: OptimizationProblem,
                 analysis: ProblemAnalysis) -> Dict[str, str]:
        # return {filename: content, ...}
        return {'solver.c': '/* your generated code */'}

    def build_ast(self, problem, analysis) -> Module:
        raise NotImplementedError

BackendRegistry.register(MyBackend())
```

2. Import it in `codegen/generator.py`:
```python
import codegen.backends.my_backend  # noqa: F401
```

3. Use it:
```python
files = gen.generate(problem, backend='my_backend')
```

---

## Requirements

- Python ≥ 3.9
- `numpy` ≥ 1.24
- `matplotlib` ≥ 3.7 (optional, for benchmark plots)
- A C compiler (`gcc` / `clang` / `arm-none-eabi-gcc`) for the generated code

---

## License

MIT
