from typing import Dict
from codegen.backends.base import CodegenBackend, BackendRegistry
from codegen.core.ast import Module
from domain.problem import OptimizationProblem, ProblemAnalysis


class OptimizedCBackend(CodegenBackend):
    """
    Generates performance-optimized C code with:
    - Sparse matrix operations (CSR format)
    - SIMD-alignment hints
    - Cache-friendly memory layout
    - Warm-start acceleration
    - Adaptive step sizing
    """

    @property
    def name(self) -> str:
        return 'c_optimized'

    @property
    def file_extension(self) -> str:
        return '.c'

    def generate(self, problem: OptimizationProblem,
                 analysis: ProblemAnalysis) -> Dict[str, str]:
        return {
            f"{problem.name}_solver.h":    self._generate_header(problem, analysis),
            f"{problem.name}_solver.c":    self._generate_solver(problem, analysis),
            f"{problem.name}_sparse.h":    self._generate_sparse_header(problem, analysis),
            f"{problem.name}_sparse.c":    self._generate_sparse_ops(problem, analysis),
            f"{problem.name}_warmstart.h": self._generate_warmstart_header(problem, analysis),
            f"{problem.name}_warmstart.c": self._generate_warmstart(problem, analysis),
        }

    def build_ast(self, problem: OptimizationProblem,
                  analysis: ProblemAnalysis) -> Module:
        raise NotImplementedError("AST emission not implemented for c_optimized; "
                                  "use generate() for direct source output.")

    # ------------------------------------------------------------------ header
    def _generate_header(self, problem, analysis) -> str:
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        jac_nnz = analysis.constraint_jacobian_nnz
        precision = 'float' if analysis.recommended_precision == 'float32' else 'double'

        return f'''/**
 * Optimized solver for: {problem.name}
 *
 * Features: sparse CSR Jacobian · warm-start · adaptive step · SIMD alignment
 *
 * Variables:    {n_vars}
 * Constraints:  {n_cons}
 * Jacobian NNZ: {jac_nnz}
 */
#ifndef {problem.name.upper()}_SOLVER_H
#define {problem.name.upper()}_SOLVER_H

#include <stdint.h>
#include "{problem.name}_sparse.h"
#include "{problem.name}_warmstart.h"

#define {problem.name.upper()}_N_VARS   {n_vars}
#define {problem.name.upper()}_N_CONS   {n_cons}
#define {problem.name.upper()}_JAC_NNZ  {jac_nnz}

typedef {precision} real_t;

typedef struct {{
    int32_t max_iterations;
    real_t  tolerance;
    real_t  initial_step_size;
    real_t  step_size_decrease;
    real_t  step_size_increase;
    int32_t use_warm_start;
    int32_t adaptive_precision;
}} {problem.name}_config_t;

typedef struct {{
    real_t x[{n_vars}]             __attribute__((aligned(32)));
    real_t x_prev[{n_vars}]        __attribute__((aligned(32)));
    real_t lambda[{n_cons}]        __attribute__((aligned(32)));
    real_t grad[{n_vars}]          __attribute__((aligned(32)));
    real_t primal_residual[{n_cons}];
    real_t dual_residual[{n_vars}];
    int32_t iteration;
    int32_t converged;
    real_t  objective_value;
    real_t  step_size;
    uint64_t total_flops;
    uint64_t total_sparse_ops;
}} {problem.name}_state_t;

typedef struct {{
    real_t Q[{n_vars}][{n_vars}];
    real_t c[{n_vars}];
    sparse_csr_t jacobian;
    real_t b[{n_cons}];
    int32_t n_obstacles;
    real_t  obstacle_centers[100][2];
    real_t  obstacle_radii[100];
}} {problem.name}_data_t;

typedef struct {{
    int32_t iterations;
    int32_t converged;
    real_t  final_objective;
    real_t  primal_infeasibility;
    real_t  dual_infeasibility;
    double  solve_time_ms;
}} {problem.name}_info_t;

void {problem.name}_default_config({problem.name}_config_t* config);

void {problem.name}_init(
    {problem.name}_state_t* state,
    const {problem.name}_config_t* config
);

int {problem.name}_solve(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    const {problem.name}_config_t* config,
    {problem.name}_info_t* info
);

void {problem.name}_warm_start(
    {problem.name}_state_t* state,
    const real_t* x_init
);

void {problem.name}_store_solution(
    {problem.name}_warmstart_cache_t* cache,
    const {problem.name}_state_t* state
);

#endif /* {problem.name.upper()}_SOLVER_H */
'''

    # ------------------------------------------------------------------ solver
    def _generate_solver(self, problem, analysis) -> str:
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        precision = 'float' if analysis.recommended_precision == 'float32' else 'double'
        sqrt_fn = 'sqrtf' if precision == 'float' else 'sqrt'
        fabs_fn = 'fabsf' if precision == 'float' else 'fabs'

        return f'''/**
 * Optimized solver implementation for: {problem.name}
 */
#include "{problem.name}_solver.h"
#include <math.h>
#include <string.h>
#include <time.h>

void {problem.name}_default_config({problem.name}_config_t* config)
{{
    config->max_iterations    = {problem.max_iterations};
    config->tolerance         = {problem.tolerance};
    config->initial_step_size = 0.1;
    config->step_size_decrease = 0.5;
    config->step_size_increase = 1.2;
    config->use_warm_start    = {1 if problem.warm_start_enabled else 0};
    config->adaptive_precision = {1 if problem.adaptive_precision else 0};
}}

void {problem.name}_init(
    {problem.name}_state_t* state,
    const {problem.name}_config_t* config)
{{
    memset(state, 0, sizeof({problem.name}_state_t));
    state->step_size = config->initial_step_size;
}}

static void sparse_jac_vec(const sparse_csr_t* J, const real_t* x, real_t* r)
{{
    for (int row = 0; row < J->n_rows; row++) {{
        real_t sum = 0.0;
        for (int idx = J->row_ptr[row]; idx < J->row_ptr[row + 1]; idx++)
            sum += J->values[idx] * x[J->col_idx[idx]];
        r[row] = sum;
    }}
}}

static real_t compute_objective(const real_t* x, const {problem.name}_data_t* data)
{{
    real_t obj = 0.0;
    for (int i = 0; i < {n_vars}; i++) {{
        for (int j = 0; j < {n_vars}; j++)
            obj += 0.5 * x[i] * data->Q[i][j] * x[j];
        obj += data->c[i] * x[i];
    }}
    return obj;
}}

static void compute_gradient(const real_t* x, const {problem.name}_data_t* data, real_t* grad)
{{
    for (int i = 0; i < {n_vars}; i++) {{
        grad[i] = data->c[i];
        for (int j = 0; j < {n_vars}; j++)
            grad[i] += data->Q[i][j] * x[j];
    }}
}}

static void compute_primal_residual(
    const real_t* x, const {problem.name}_data_t* data, real_t* residual)
{{
    sparse_jac_vec(&data->jacobian, x, residual);
    for (int i = 0; i < {n_cons}; i++)
        residual[i] -= data->b[i];
}}

static int check_convergence(
    const {problem.name}_state_t* state,
    const {problem.name}_config_t* config)
{{
    real_t change_norm = 0.0, primal_norm = 0.0;
    for (int i = 0; i < {n_vars}; i++) {{
        real_t d = state->x[i] - state->x_prev[i];
        change_norm += d * d;
    }}
    for (int i = 0; i < {n_cons}; i++)
        primal_norm += state->primal_residual[i] * state->primal_residual[i];

    return ({sqrt_fn}(change_norm) < config->tolerance) &&
           ({sqrt_fn}(primal_norm) < config->tolerance * 10.0);
}}

static void proximal_gradient_step(
    {problem.name}_state_t* state, const {problem.name}_data_t* data)
{{
    memcpy(state->x_prev, state->x, sizeof(state->x));
    compute_gradient(state->x, data, state->grad);
    for (int i = 0; i < {n_vars}; i++)
        state->x[i] -= state->step_size * state->grad[i];
    compute_primal_residual(state->x, data, state->primal_residual);
}}

static void adapt_step_size(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    const {problem.name}_config_t* config)
{{
    real_t new_obj = compute_objective(state->x, data);
    if (new_obj < state->objective_value) {{
        state->step_size *= config->step_size_increase;
        if (state->step_size > 1.0) state->step_size = 1.0;
    }} else {{
        state->step_size *= config->step_size_decrease;
        if (state->step_size < 1e-8) state->step_size = 1e-8;
    }}
    state->objective_value = new_obj;
}}

int {problem.name}_solve(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    const {problem.name}_config_t* config,
    {problem.name}_info_t* info)
{{
    clock_t t0 = clock();
    state->objective_value = compute_objective(state->x, data);

    while (state->iteration < config->max_iterations) {{
        proximal_gradient_step(state, data);
        adapt_step_size(state, data, config);
        state->iteration++;
        if (check_convergence(state, config)) {{
            state->converged = 1;
            break;
        }}
    }}

    if (info) {{
        info->iterations        = state->iteration;
        info->converged         = state->converged;
        info->final_objective   = state->objective_value;
        real_t pi = 0.0;
        for (int i = 0; i < {n_cons}; i++)
            pi += {fabs_fn}(state->primal_residual[i]);
        info->primal_infeasibility = pi / {n_cons};
        info->solve_time_ms =
            (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
    }}
    return state->converged;
}}

void {problem.name}_warm_start({problem.name}_state_t* state, const real_t* x_init)
{{
    if (x_init) memcpy(state->x, x_init, sizeof(state->x));
}}
'''

    # ---------------------------------------------------------------- sparse
    def _generate_sparse_header(self, problem, analysis) -> str:
        return f'''/**
 * Sparse matrix operations for: {problem.name}
 */
#ifndef {problem.name.upper()}_SPARSE_H
#define {problem.name.upper()}_SPARSE_H

#include <stdint.h>

typedef struct {{
    int32_t  n_rows;
    int32_t  n_cols;
    int32_t  nnz;
    int32_t* row_ptr;
    int32_t* col_idx;
    double*  values;
}} sparse_csr_t;

void sparse_csr_from_dense(sparse_csr_t* csr, const double* dense,
                            int32_t n_rows, int32_t n_cols, double threshold);
void sparse_csr_free(sparse_csr_t* csr);
void sparse_matvec(const sparse_csr_t* A, const double* x, double* y);
void sparse_matvec_t(const sparse_csr_t* A, const double* x, double* y);

#endif /* {problem.name.upper()}_SPARSE_H */
'''

    def _generate_sparse_ops(self, problem, analysis) -> str:
        return f'''/**
 * Sparse matrix operations implementation for: {problem.name}
 */
#include "{problem.name}_sparse.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void sparse_csr_from_dense(
    sparse_csr_t* csr, const double* dense,
    int32_t n_rows, int32_t n_cols, double threshold)
{{
    int32_t nnz = 0;
    for (int32_t i = 0; i < n_rows * n_cols; i++)
        if (fabs(dense[i]) > threshold) nnz++;

    csr->n_rows  = n_rows;
    csr->n_cols  = n_cols;
    csr->nnz     = nnz;
    csr->row_ptr = (int32_t*)malloc((n_rows + 1) * sizeof(int32_t));
    csr->col_idx = (int32_t*)malloc(nnz * sizeof(int32_t));
    csr->values  = (double* )malloc(nnz * sizeof(double));

    int32_t idx = 0;
    for (int32_t row = 0; row < n_rows; row++) {{
        csr->row_ptr[row] = idx;
        for (int32_t col = 0; col < n_cols; col++) {{
            double val = dense[row * n_cols + col];
            if (fabs(val) > threshold) {{
                csr->col_idx[idx] = col;
                csr->values[idx]  = val;
                idx++;
            }}
        }}
    }}
    csr->row_ptr[n_rows] = nnz;
}}

void sparse_csr_free(sparse_csr_t* csr)
{{
    free(csr->row_ptr); csr->row_ptr = NULL;
    free(csr->col_idx); csr->col_idx = NULL;
    free(csr->values);  csr->values  = NULL;
}}

void sparse_matvec(const sparse_csr_t* A, const double* x, double* y)
{{
    for (int32_t row = 0; row < A->n_rows; row++) {{
        double sum = 0.0;
        for (int32_t i = A->row_ptr[row]; i < A->row_ptr[row + 1]; i++)
            sum += A->values[i] * x[A->col_idx[i]];
        y[row] = sum;
    }}
}}

void sparse_matvec_t(const sparse_csr_t* A, const double* x, double* y)
{{
    memset(y, 0, A->n_cols * sizeof(double));
    for (int32_t row = 0; row < A->n_rows; row++)
        for (int32_t i = A->row_ptr[row]; i < A->row_ptr[row + 1]; i++)
            y[A->col_idx[i]] += A->values[i] * x[row];
}}
'''

    # -------------------------------------------------------------- warmstart
    def _generate_warmstart_header(self, problem, analysis) -> str:
        n_vars = analysis.total_variables
        return f'''/**
 * Warm-start cache for: {problem.name}
 */
#ifndef {problem.name.upper()}_WARMSTART_H
#define {problem.name.upper()}_WARMSTART_H

#include <stdint.h>

#define {problem.name.upper()}_WARMSTART_HISTORY 5

typedef double real_t;

typedef struct {{
    real_t  solutions[{problem.name.upper()}_WARMSTART_HISTORY][{n_vars}];
    int32_t count;
    int32_t current_idx;
}} {problem.name}_warmstart_cache_t;

void {problem.name}_warmstart_init({problem.name}_warmstart_cache_t* cache);
void {problem.name}_warmstart_add({problem.name}_warmstart_cache_t* cache, const real_t* solution);
const real_t* {problem.name}_warmstart_get(const {problem.name}_warmstart_cache_t* cache);
const real_t* {problem.name}_warmstart_interpolate(
    const {problem.name}_warmstart_cache_t* cache, real_t* result);

#endif /* {problem.name.upper()}_WARMSTART_H */
'''

    def _generate_warmstart(self, problem, analysis) -> str:
        n_vars = analysis.total_variables
        hist = f"{problem.name.upper()}_WARMSTART_HISTORY"
        return f'''/**
 * Warm-start cache implementation for: {problem.name}
 */
#include "{problem.name}_warmstart.h"
#include <string.h>

void {problem.name}_warmstart_init({problem.name}_warmstart_cache_t* cache)
{{
    memset(cache, 0, sizeof({problem.name}_warmstart_cache_t));
}}

void {problem.name}_warmstart_add(
    {problem.name}_warmstart_cache_t* cache, const real_t* solution)
{{
    memcpy(cache->solutions[cache->current_idx], solution, {n_vars} * sizeof(real_t));
    cache->current_idx = (cache->current_idx + 1) % {hist};
    if (cache->count < {hist}) cache->count++;
}}

const real_t* {problem.name}_warmstart_get(const {problem.name}_warmstart_cache_t* cache)
{{
    if (cache->count == 0) return NULL;
    int32_t last = (cache->current_idx - 1 + {hist}) % {hist};
    return cache->solutions[last];
}}

const real_t* {problem.name}_warmstart_interpolate(
    const {problem.name}_warmstart_cache_t* cache, real_t* result)
{{
    if (cache->count == 0) return NULL;
    if (cache->count == 1) return {problem.name}_warmstart_get(cache);

    int32_t i1 = (cache->current_idx - 1 + {hist}) % {hist};
    int32_t i0 = (cache->current_idx - 2 + {hist}) % {hist};

    for (int i = 0; i < {n_vars}; i++)
        result[i] = 2.0 * cache->solutions[i1][i] - cache->solutions[i0][i];

    return result;
}}
'''


BackendRegistry.register(OptimizedCBackend())
