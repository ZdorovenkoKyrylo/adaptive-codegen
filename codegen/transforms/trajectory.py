from typing import List
import numpy as np
from domain.problem import (
    OptimizationProblem, Variable, Parameter, Constraint, Objective,
    VariableType, ConstraintType, ObjectiveType,
)
from domain.geometry import Workspace, TrajectorySpec


class TrajectoryProblemGenerator:
    """
    Generates trajectory optimization problems from high-level specifications.
    """

    def generate(self, spec: TrajectorySpec,
                 workspace: Workspace) -> OptimizationProblem:
        variables  = self._create_variables(spec)
        parameters = self._create_parameters(workspace)

        constraints = []
        constraints.extend(self._boundary_constraints(spec))
        constraints.extend(self._dynamics_constraints(spec))
        constraints.extend(self._bound_constraints(spec))
        constraints.extend(self._obstacle_constraints(spec, workspace))

        objective = self._create_objective(spec)

        return OptimizationProblem(
            name='trajectory',
            variables=variables,
            parameters=parameters,
            constraints=constraints,
            objective=objective,
            max_iterations=100,
            tolerance=1e-6,
            warm_start_enabled=True,
            adaptive_precision=True,
        )

    def _create_variables(self, spec: TrajectorySpec) -> List[Variable]:
        n = spec.n_points
        variables = []
        for dim in ['x', 'y', 'vx', 'vy'][:spec.state_dim]:
            variables.append(Variable(name=f'state_{dim}', shape=(n,),
                                      dtype=VariableType.CONTINUOUS))
        for dim in ['ax', 'ay'][:spec.control_dim]:
            variables.append(Variable(name=f'control_{dim}', shape=(n - 1,),
                                      dtype=VariableType.CONTINUOUS))
        return variables

    def _create_parameters(self, workspace: Workspace) -> List[Parameter]:
        params = []
        for i, obs in enumerate(workspace.obstacles):
            params.append(Parameter(
                name=f'obstacle_{i}_center', shape=(2,),
                default_value=obs.center, is_obstacle=True))
            params.append(Parameter(
                name=f'obstacle_{i}_radius', shape=(1,),
                default_value=np.array([obs.radius]), is_obstacle=True))
        return params

    def _boundary_constraints(self, spec: TrajectorySpec) -> List[Constraint]:
        cs = []
        for i, dim in enumerate(['x', 'y', 'vx', 'vy'][:spec.state_dim]):
            cs.append(Constraint(
                name=f'initial_{dim}',
                constraint_type=ConstraintType.LINEAR_EQUALITY,
                variables=[f'state_{dim}'], parameters=[],
                expression=f'state_{dim}[0] - {spec.start_state[i]}'))
            cs.append(Constraint(
                name=f'final_{dim}',
                constraint_type=ConstraintType.LINEAR_EQUALITY,
                variables=[f'state_{dim}'], parameters=[],
                expression=f'state_{dim}[-1] - {spec.goal_state[i]}'))
        return cs

    def _dynamics_constraints(self, spec: TrajectorySpec) -> List[Constraint]:
        dt = spec.dt
        return [
            Constraint(
                name='dynamics_x', constraint_type=ConstraintType.LINEAR_EQUALITY,
                variables=['state_x', 'state_vx', 'control_ax'], parameters=[],
                expression=f'state_x[1:] - state_x[:-1] - state_vx[:-1]*{dt} - 0.5*control_ax*{dt**2}'),
            Constraint(
                name='dynamics_y', constraint_type=ConstraintType.LINEAR_EQUALITY,
                variables=['state_y', 'state_vy', 'control_ay'], parameters=[],
                expression=f'state_y[1:] - state_y[:-1] - state_vy[:-1]*{dt} - 0.5*control_ay*{dt**2}'),
            Constraint(
                name='dynamics_vx', constraint_type=ConstraintType.LINEAR_EQUALITY,
                variables=['state_vx', 'control_ax'], parameters=[],
                expression=f'state_vx[1:] - state_vx[:-1] - control_ax*{dt}'),
            Constraint(
                name='dynamics_vy', constraint_type=ConstraintType.LINEAR_EQUALITY,
                variables=['state_vy', 'control_ay'], parameters=[],
                expression=f'state_vy[1:] - state_vy[:-1] - control_ay*{dt}'),
        ]

    def _bound_constraints(self, spec: TrajectorySpec) -> List[Constraint]:
        cs = []
        for i, dim in enumerate(['x', 'y', 'vx', 'vy'][:spec.state_dim]):
            cs.append(Constraint(
                name=f'{dim}_lb', constraint_type=ConstraintType.LINEAR_INEQUALITY,
                variables=[f'state_{dim}'], parameters=[],
                expression=f'{spec.state_bounds[0][i]} - state_{dim}'))
            cs.append(Constraint(
                name=f'{dim}_ub', constraint_type=ConstraintType.LINEAR_INEQUALITY,
                variables=[f'state_{dim}'], parameters=[],
                expression=f'state_{dim} - {spec.state_bounds[1][i]}'))
        for i, dim in enumerate(['ax', 'ay'][:spec.control_dim]):
            cs.append(Constraint(
                name=f'{dim}_lb', constraint_type=ConstraintType.LINEAR_INEQUALITY,
                variables=[f'control_{dim}'], parameters=[],
                expression=f'{spec.control_bounds[0][i]} - control_{dim}'))
            cs.append(Constraint(
                name=f'{dim}_ub', constraint_type=ConstraintType.LINEAR_INEQUALITY,
                variables=[f'control_{dim}'], parameters=[],
                expression=f'control_{dim} - {spec.control_bounds[1][i]}'))
        return cs

    def _obstacle_constraints(self, spec: TrajectorySpec,
                               workspace: Workspace) -> List[Constraint]:
        cs = []
        for obs_idx, obs in enumerate(workspace.obstacles):
            cs.append(Constraint(
                name=f'obstacle_{obs_idx}_avoidance',
                constraint_type=ConstraintType.SECOND_ORDER_CONE,
                variables=['state_x', 'state_y'],
                parameters=[f'obstacle_{obs_idx}_center', f'obstacle_{obs_idx}_radius'],
                expression=(
                    f'obstacle_{obs_idx}_radius - '
                    f'norm(stack(state_x - obstacle_{obs_idx}_center[0], '
                    f'state_y - obstacle_{obs_idx}_center[1]))'
                ),
            ))
        return cs

    def _create_objective(self, spec: TrajectorySpec) -> Objective:
        return Objective(
            objective_type=ObjectiveType.QUADRATIC,
            expression='sum(control_ax**2 + control_ay**2)',
            gradient='[2*control_ax, 2*control_ay]',
            hessian='diag([2, 2, ...])',
        )
