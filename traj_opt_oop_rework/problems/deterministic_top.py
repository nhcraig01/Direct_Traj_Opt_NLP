"""DeterministicTOP - 06_deterministic_top_composition.puml.

The simplest concrete TrajectoryOptimizationProblem: no GainParameterization,
no FeedbackControlMethod, no ErrorDynamics/ErrorState. Intended as the first
end-to-end implementation target and the benchmark against legacy
Problem_Type == "deterministic" results.
"""

import jax
import jax.numpy as jnp

from traj_opt_oop_rework.constraints.constraint import (
    BoundaryConstraint, ControlNormConstraint, DetCollisionAvoidanceConstraint, MatchPointConstraint, MeshConstraint,
)
from traj_opt_oop_rework.data_structures import ErrorState, OptimizationVariable, TrajectoryState
from traj_opt_oop_rework.dynamics.equations_of_motion import CR3BPDynamics
from traj_opt_oop_rework.problem_definition import ProblemDefinition
from traj_opt_oop_rework.problems.trajectory_optimization_problem import TrajectoryOptimizationProblem
from traj_opt_oop_rework.propagators.propagator import StatePropagator


class DeterministicTOP(TrajectoryOptimizationProblem):
    def __init__(self, problem_def: ProblemDefinition):
        propagator = StatePropagator(problem_def, CR3BPDynamics(problem_def))
        constraints_list = [
            BoundaryConstraint(problem_def, "initial"),
            BoundaryConstraint(problem_def, "final"),
            MatchPointConstraint(problem_def),
            ControlNormConstraint(problem_def),
        ]
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            constraints_list.append(MeshConstraint(problem_def))
        if problem_def.toggles.det_col_avoid and not problem_def.toggles.stat_col_avoid:
            constraints_list.append(DetCollisionAvoidanceConstraint(problem_def))
        super().__init__(problem_def, propagator, constraints_list)

    def variables(self) -> list[OptimizationVariable]:
        dims = self._problem_def.dims
        bc = self._problem_def.boundary_conditions

        U_size = dims.control_dim * dims.N_arcs

        variables_list = [
            OptimizationVariable(
                name='U_arc_hst', size=U_size,
                lower=-jnp.ones(U_size), upper=jnp.ones(U_size), value=jnp.zeros(U_size),
                guess_fn=lambda key: jnp.sqrt(1e-3) * jax.random.normal(key, shape=(U_size,)),
            ),
            OptimizationVariable(
                name='X0', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.X0_init, jnp.array([1.0])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.X0_interp.evaluate(jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max)),
                    jnp.array([1.0]),
                ]),
            ),
            OptimizationVariable(
                name='Xf', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.Xf_init, jnp.array([0.95])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.Xf_interp.evaluate(jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max)),
                    jnp.array([0.95]),
                ]),
            ),
            OptimizationVariable(
                name='alpha', size=1,
                lower=jnp.asarray(bc.alpha_min), upper=jnp.asarray(bc.alpha_max), value=jnp.asarray(bc.alpha_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max),
            ),
            OptimizationVariable(
                name='beta', size=1,
                lower=jnp.asarray(bc.beta_min), upper=jnp.asarray(bc.beta_max), value=jnp.asarray(bc.beta_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max),
            ),
        ]

        if self._problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            variables_list.append(OptimizationVariable(
                name='t_node_bound', size=dims.N_nodes,
                lower=jnp.zeros(dims.N_nodes), upper=jnp.full(dims.N_nodes, bc.t_node_bound[-1]),
                value=bc.t_node_bound, guess_fn=lambda key: bc.t_node_bound,
            ))

        return variables_list

    def objective(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> float:
        dims = self._problem_def.dims

        eps = 1e-12
        U_arc_hst = inputs['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        control_norms = jnp.sqrt(U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps)

        arc_length_hst = jnp.diff(traj_state.t_node_bound)
        J = control_norms @ arc_length_hst

        if self._problem_def.toggles.adaptive_mesh_type.lower() != 'fixed':
            t_node_bound = traj_state.t_node_bound
            dt_even = (t_node_bound[-1] - t_node_bound[0]) / dims.N_arcs
            rel_dt_err = arc_length_hst / dt_even - 1.0
            J = J + 1e-6 * (rel_dt_err @ rel_dt_err)

        return J

    def evaluate(self, inputs: dict) -> dict:
        traj_state, error_dynamics = self.propagate(inputs)  # error_dynamics is always None - StatePropagator never produces it
        error_state = None  # DeterministicTOP never computes an ErrorState

        output = {'o': self.objective(inputs, traj_state, error_state)}
        for c in self.constraints():
            if c.linearities:
                continue  # linear constraints (e.g. c_t_node_bound) are evaluated by pyoptsparse via their jac, not vals()
            output[c.name] = c.evaluate(inputs, traj_state, error_state)

        return output
