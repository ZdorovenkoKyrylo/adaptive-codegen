from typing import Dict, List
from codegen.backends.base import CodegenBackend, BackendRegistry
from codegen.core.ast import *
from domain.problem import OptimizationProblem, ProblemAnalysis


class EmbeddedCBackend(CodegenBackend):
    """
    Generates embedded-compatible C code.

    Features:
    - No dynamic memory allocation
    - Fixed-size arrays
    - Deterministic execution
    - Minimal dependencies (only <math.h> and <string.h>)
    """

    @property
    def name(self) -> str:
        return 'c_embedded'

    @property
    def file_extension(self) -> str:
        return '.c'

    def generate(self, problem: OptimizationProblem,
                 analysis: ProblemAnalysis) -> Dict[str, str]:
        ast = self.build_ast(problem, analysis)
        header = self._generate_header(problem, analysis)
        source = self._generate_source(problem, analysis, ast)
        return {
            f"{problem.name}_solver.h": header,
            f"{problem.name}_solver.c": source,
        }

    def build_ast(self, problem: OptimizationProblem,
                  analysis: ProblemAnalysis) -> Module:
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        dtype = DataType.FLOAT32 if analysis.recommended_precision == 'float32' \
            else DataType.FLOAT64

        module = Module(
            name=f"{problem.name}_solver",
            includes=['<math.h>', '<string.h>'],
            structs=[],
            globals=[],
            functions=[]
        )

        state_struct = StructDef(
            name=f"{problem.name}_state",
            fields=[
                TypedVariable('x', dtype, (n_vars,)),
                TypedVariable('x_prev', dtype, (n_vars,)),
                TypedVariable('gradient', dtype, (n_vars,)),
                TypedVariable('residual', dtype, (n_cons,)),
                TypedVariable('iteration', DataType.INT32),
                TypedVariable('converged', DataType.INT32),
            ]
        )
        module.structs.append(state_struct)

        data_struct = StructDef(
            name=f"{problem.name}_data",
            fields=self._generate_data_fields(problem, dtype)
        )
        module.structs.append(data_struct)

        module.functions.extend(
            self._generate_solver_functions(problem, analysis, dtype)
        )
        return module

    def _generate_data_fields(self, problem: OptimizationProblem,
                               dtype: DataType) -> List[TypedVariable]:
        fields = []
        for param in problem.parameters:
            shape = (param.shape[0],) if len(param.shape) == 1 else param.shape
            fields.append(TypedVariable(param.name, dtype, shape))
        return fields

    def _generate_solver_functions(self, problem, analysis, dtype):
        return [
            self._gen_init_function(problem, analysis, dtype),
            self._gen_objective_function(problem, analysis, dtype),
            self._gen_gradient_function(problem, analysis, dtype),
            self._gen_constraint_function(problem, analysis, dtype),
            self._gen_iteration_function(problem, analysis, dtype),
            self._gen_solve_function(problem, analysis, dtype),
        ]

    def _gen_init_function(self, problem, analysis, dtype):
        n_vars = analysis.total_variables
        body = [
            Assignment(target=VariableRef('state->iteration'), value=Literal(0)),
            Assignment(target=VariableRef('state->converged'), value=Literal(0)),
        ]
        if problem.warm_start_enabled:
            body.append(ForLoop(
                iterator='i', start=Literal(0), end=Literal(n_vars), step=Literal(1),
                body=[Assignment(
                    target=VariableRef('state->x', [VariableRef('i')]),
                    value=VariableRef('warm_start', [VariableRef('i')])
                )]
            ))
        return FunctionDef(
            name=f"{problem.name}_init",
            return_type=DataType.VOID,
            parameters=[
                TypedVariable('state', dtype),
                TypedVariable('warm_start', dtype, (n_vars,)),
            ],
            body=body
        )

    def _gen_objective_function(self, problem, analysis, dtype):
        n_vars = analysis.total_variables
        body = [Assignment(target=VariableRef('result'), value=Literal(0.0))]
        body.append(ForLoop(
            iterator='i', start=Literal(0), end=Literal(n_vars), step=Literal(1),
            body=[ForLoop(
                iterator='j', start=Literal(0), end=Literal(n_vars), step=Literal(1),
                body=[Assignment(
                    target=VariableRef('result'),
                    value=BinaryOp(
                        left=VariableRef('result'), op='+',
                        right=BinaryOp(
                            left=BinaryOp(left=Literal(0.5), op='*',
                                          right=VariableRef('x', [VariableRef('i')])),
                            op='*',
                            right=BinaryOp(
                                left=VariableRef('data->Q', [VariableRef('i'), VariableRef('j')]),
                                op='*',
                                right=VariableRef('x', [VariableRef('j')])
                            )
                        )
                    )
                )]
            )]
        ))
        body.append(Return(VariableRef('result')))
        return FunctionDef(
            name=f"{problem.name}_objective",
            return_type=dtype,
            parameters=[
                TypedVariable('x', dtype, (n_vars,)),
                TypedVariable('data', dtype),
            ],
            body=body,
            is_inline=True
        )

    def _gen_gradient_function(self, problem, analysis, dtype):
        n_vars = analysis.total_variables
        body = [ForLoop(
            iterator='i', start=Literal(0), end=Literal(n_vars), step=Literal(1),
            body=[
                Assignment(
                    target=VariableRef('grad', [VariableRef('i')]),
                    value=VariableRef('data->c', [VariableRef('i')])
                ),
                ForLoop(
                    iterator='j', start=Literal(0), end=Literal(n_vars), step=Literal(1),
                    body=[Assignment(
                        target=VariableRef('grad', [VariableRef('i')]),
                        value=BinaryOp(
                            left=VariableRef('grad', [VariableRef('i')]), op='+',
                            right=BinaryOp(
                                left=VariableRef('data->Q', [VariableRef('i'), VariableRef('j')]),
                                op='*',
                                right=VariableRef('x', [VariableRef('j')])
                            )
                        )
                    )]
                )
            ]
        )]
        return FunctionDef(
            name=f"{problem.name}_gradient",
            return_type=DataType.VOID,
            parameters=[
                TypedVariable('x', dtype, (n_vars,)),
                TypedVariable('grad', dtype, (n_vars,)),
                TypedVariable('data', dtype),
            ],
            body=body
        )

    def _gen_constraint_function(self, problem, analysis, dtype):
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        body = [ForLoop(
            iterator='c', start=Literal(0), end=Literal(n_cons), step=Literal(1),
            body=[Assignment(
                target=VariableRef('residual', [VariableRef('c')]),
                value=FunctionCall(
                    name=f"eval_constraint_{problem.name}",
                    arguments=[VariableRef('c'), VariableRef('x'), VariableRef('data')]
                )
            )]
        )]
        return FunctionDef(
            name=f"{problem.name}_constraints",
            return_type=DataType.VOID,
            parameters=[
                TypedVariable('x', dtype, (n_vars,)),
                TypedVariable('residual', dtype, (n_cons,)),
                TypedVariable('data', dtype),
            ],
            body=body
        )

    def _gen_iteration_function(self, problem, analysis, dtype):
        n_vars = analysis.total_variables
        body = [
            FunctionCall(
                name=f"{problem.name}_gradient",
                arguments=[VariableRef('state->x'), VariableRef('state->gradient'), VariableRef('data')]
            ),
            ForLoop(
                iterator='i', start=Literal(0), end=Literal(n_vars), step=Literal(1),
                body=[
                    Assignment(
                        target=VariableRef('state->x_prev', [VariableRef('i')]),
                        value=VariableRef('state->x', [VariableRef('i')])
                    ),
                    Assignment(
                        target=VariableRef('state->x', [VariableRef('i')]),
                        value=BinaryOp(
                            left=VariableRef('state->x', [VariableRef('i')]), op='-',
                            right=BinaryOp(
                                left=VariableRef('step_size'), op='*',
                                right=VariableRef('state->gradient', [VariableRef('i')])
                            )
                        )
                    )
                ]
            ),
            Assignment(
                target=VariableRef('state->iteration'),
                value=BinaryOp(left=VariableRef('state->iteration'), op='+', right=Literal(1))
            )
        ]
        return FunctionDef(
            name=f"{problem.name}_iterate",
            return_type=DataType.VOID,
            parameters=[
                TypedVariable('state', dtype),
                TypedVariable('data', dtype),
                TypedVariable('step_size', dtype, (1,)),
            ],
            body=body
        )

    def _gen_solve_function(self, problem, analysis, dtype):
        body = [
            WhileLoop(
                condition=BinaryOp(
                    left=BinaryOp(
                        left=VariableRef('state->iteration'), op='<',
                        right=Literal(problem.max_iterations)
                    ),
                    op='&&',
                    right=UnaryOp('!', VariableRef('state->converged'))
                ),
                body=[
                    FunctionCall(
                        name=f"{problem.name}_iterate",
                        arguments=[VariableRef('state'), VariableRef('data'), VariableRef('step_size')]
                    ),
                    Assignment(
                        target=VariableRef('state->converged'),
                        value=FunctionCall(
                            name=f"{problem.name}_check_convergence",
                            arguments=[VariableRef('state'), VariableRef('tolerance')]
                        )
                    )
                ]
            ),
            Return(VariableRef('state->iteration'))
        ]
        return FunctionDef(
            name=f"{problem.name}_solve",
            return_type=DataType.INT32,
            parameters=[
                TypedVariable('state', dtype),
                TypedVariable('data', dtype),
                TypedVariable('step_size', dtype, (1,)),
                TypedVariable('tolerance', dtype, (1,)),
            ],
            body=body
        )

    def _generate_header(self, problem: OptimizationProblem,
                         analysis: ProblemAnalysis) -> str:
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        precision = 'float' if analysis.recommended_precision == 'float32' else 'double'

        return f'''/**
 * Auto-generated embedded solver for: {problem.name}
 *
 * Variables:   {n_vars}
 * Constraints: {n_cons}
 * Precision:   {precision}
 *
 * Generated by Adaptive Embedded Optimizer
 * No dynamic memory allocation — safe for bare-metal targets.
 */
#ifndef {problem.name.upper()}_SOLVER_H
#define {problem.name.upper()}_SOLVER_H

#include <stdint.h>

#define {problem.name.upper()}_N_VARS   {n_vars}
#define {problem.name.upper()}_N_CONS   {n_cons}
#define {problem.name.upper()}_MAX_ITER {problem.max_iterations}
#define {problem.name.upper()}_TOLERANCE {problem.tolerance}

typedef {precision} real_t;

typedef struct {{
    real_t x[{n_vars}];
    real_t x_prev[{n_vars}];
    real_t gradient[{n_vars}];
    real_t residual[{n_cons}];
    int32_t iteration;
    int32_t converged;
}} {problem.name}_state_t;

typedef struct {{
    real_t Q[{n_vars}][{n_vars}];
    real_t c[{n_vars}];
    real_t A[{n_cons}][{n_vars}];
    real_t b[{n_cons}];
}} {problem.name}_data_t;

void  {problem.name}_init(
    {problem.name}_state_t* state,
    const real_t* warm_start
);

real_t {problem.name}_objective(
    const real_t* x,
    const {problem.name}_data_t* data
);

void {problem.name}_gradient(
    const real_t* x,
    real_t* grad,
    const {problem.name}_data_t* data
);

void {problem.name}_constraints(
    const real_t* x,
    real_t* residual,
    const {problem.name}_data_t* data
);

void {problem.name}_iterate(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    real_t step_size
);

int {problem.name}_check_convergence(
    const {problem.name}_state_t* state,
    real_t tolerance
);

int {problem.name}_solve(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    real_t step_size,
    real_t tolerance
);

#endif /* {problem.name.upper()}_SOLVER_H */
'''

    def _generate_source(self, problem: OptimizationProblem,
                         analysis: ProblemAnalysis,
                         ast: Module) -> str:
        n_vars = analysis.total_variables
        n_cons = analysis.total_constraints
        precision = 'float' if analysis.recommended_precision == 'float32' else 'double'
        fabs_fn = 'fabsf' if precision == 'float' else 'fabs'

        return f'''/**
 * Auto-generated embedded solver for: {problem.name}
 * Generated by Adaptive Embedded Optimizer
 */
#include "{problem.name}_solver.h"
#include <math.h>
#include <string.h>

/* ============ Initialization ============ */
void {problem.name}_init(
    {problem.name}_state_t* state,
    const real_t* warm_start)
{{
    state->iteration = 0;
    state->converged = 0;

    if (warm_start != NULL) {{
        memcpy(state->x, warm_start, sizeof(state->x));
    }} else {{
        memset(state->x, 0, sizeof(state->x));
    }}
    memset(state->x_prev,   0, sizeof(state->x_prev));
    memset(state->gradient, 0, sizeof(state->gradient));
    memset(state->residual, 0, sizeof(state->residual));
}}

/* ============ Objective Function ============ */
real_t {problem.name}_objective(
    const real_t* x,
    const {problem.name}_data_t* data)
{{
    real_t result = 0.0;

    for (int i = 0; i < {n_vars}; i++) {{
        for (int j = 0; j < {n_vars}; j++) {{
            result += 0.5 * x[i] * data->Q[i][j] * x[j];
        }}
        result += data->c[i] * x[i];
    }}
    return result;
}}

/* ============ Gradient Computation ============ */
void {problem.name}_gradient(
    const real_t* x,
    real_t* grad,
    const {problem.name}_data_t* data)
{{
    for (int i = 0; i < {n_vars}; i++) {{
        grad[i] = data->c[i];
        for (int j = 0; j < {n_vars}; j++) {{
            grad[i] += data->Q[i][j] * x[j];
        }}
    }}
}}

/* ============ Constraint Evaluation ============ */
void {problem.name}_constraints(
    const real_t* x,
    real_t* residual,
    const {problem.name}_data_t* data)
{{
    for (int c = 0; c < {n_cons}; c++) {{
        residual[c] = -data->b[c];
        for (int j = 0; j < {n_vars}; j++) {{
            residual[c] += data->A[c][j] * x[j];
        }}
    }}
}}

/* ============ Convergence Check ============ */
int {problem.name}_check_convergence(
    const {problem.name}_state_t* state,
    real_t tolerance)
{{
    real_t max_change = 0.0;
    for (int i = 0; i < {n_vars}; i++) {{
        real_t diff = {fabs_fn}(state->x[i] - state->x_prev[i]);
        if (diff > max_change) max_change = diff;
    }}
    return max_change < tolerance;
}}

/* ============ Single Iteration ============ */
void {problem.name}_iterate(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    real_t step_size)
{{
    memcpy(state->x_prev, state->x, sizeof(state->x));
    {problem.name}_gradient(state->x, state->gradient, data);

    for (int i = 0; i < {n_vars}; i++) {{
        state->x[i] -= step_size * state->gradient[i];
    }}
    state->iteration++;
}}

/* ============ Main Solver ============ */
int {problem.name}_solve(
    {problem.name}_state_t* state,
    const {problem.name}_data_t* data,
    real_t step_size,
    real_t tolerance)
{{
    while (state->iteration < {problem.max_iterations} && !state->converged) {{
        {problem.name}_iterate(state, data, step_size);
        state->converged = {problem.name}_check_convergence(state, tolerance);
    }}
    return state->iteration;
}}
'''


BackendRegistry.register(EmbeddedCBackend())
