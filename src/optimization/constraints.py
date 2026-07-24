"""Constraint hierarchy - 04_constraints.puml.

Constraints do NOT contribute optimization variables - every variable is
declared in one place by TOP.variables() (see 05_top_overview.puml).

evaluate() always takes (problem_def, sol_data) - sol_data is inputs
updated in place by each pipeline stage as it runs (see
trajectory_optimization_problem.py); problem_def is passed explicitly
rather than relying on self._problem_def, so every constraint's evaluate()
has the exact same two-parameter signature regardless of what it happens
to need at construction time (mirrors ErrorPropagator.propagate_cov()).
Constructors still take problem_def to size/bound themselves, but don't
need to keep it around afterward - presence/absence of a sol_data key
replaces what used to be an error_state is None check.
"""

import jax.numpy as jnp
from jax import Array

from src.problem.problem_definition import ProblemDefinition
from src.stochastic.error_propagator import ErrorPropagator
from src.utils.math_utils import adaptive_mesh_con_terms, col_avoid_vmap


class Constraint:
    """+name, +size, +lower, +upper, +linearities."""

    def __init__(self, name: str, size: int, lower: Array, upper: Array, linearities: dict | None = None):
        self.name = name
        self.size = size
        self.lower = lower
        self.upper = upper
        self.linearities = linearities if linearities is not None else {}

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        raise NotImplementedError


class BoundaryConstraint(Constraint):
    """c_X0 / c_Xf - one instance per boundary, constructed twice by build_problem()."""

    def __init__(self, problem_def: ProblemDefinition, boundary: str):
        self._boundary = boundary
        bc = problem_def.boundary_conditions

        if boundary == "initial":
            name, size, orbit_family = "c_X0", problem_def.dims.state_dim, bc.X0_interp
        elif boundary == "final":
            name, size, orbit_family = "c_Xf", problem_def.dims.state_dim - 1, bc.Xf_interp
        else:
            raise ValueError(f"boundary must be 'initial' or 'final', got {boundary!r}")

        self._orbit_family = orbit_family
        super().__init__(name=name, size=size, lower=jnp.zeros(size), upper=jnp.zeros(size))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        if self._boundary == "initial":
            X0 = sol_data['X0']
            alpha = sol_data['alpha']
            return (X0[:7] - jnp.concatenate([self._orbit_family.evaluate(alpha).flatten(), jnp.array([1.0])])).flatten()
        else:
            Xf = sol_data['Xf']
            beta = sol_data['beta']
            return (Xf[:6] - self._orbit_family.evaluate(beta)).flatten()


class MatchPointConstraint(Constraint):
    """c_X_mp - identical for DeterministicTOP and StochasticTOP; only reads X_hst."""

    def __init__(self, problem_def: ProblemDefinition):
        size = problem_def.dims.state_dim
        super().__init__(name="c_X_mp", size=size, lower=jnp.zeros(size), upper=jnp.zeros(size))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        bc = problem_def.boundary_conditions
        X_hst = sol_data['X_hst']
        return (X_hst[bc.indx_f[-1], -1, :7] - X_hst[bc.indx_b[-1], 0, :7]).flatten()


class ControlNormConstraint(Constraint):
    """c_Us. Always includes control_norms; if an error_propagator is given
    (StochasticTOP only), also adds its stochastic_control_term()."""

    def __init__(self, problem_def: ProblemDefinition, error_propagator: ErrorPropagator | None = None):
        self._error_propagator = error_propagator
        size = problem_def.dims.N_arcs
        super().__init__(name="c_Us", size=size, lower=-jnp.inf * jnp.ones(size), upper=jnp.ones(size))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        eps = 1e-12
        U_arc_hst = sol_data['U_arc_hst'].reshape(problem_def.dims.N_arcs, problem_def.dims.control_dim)
        control_norms = jnp.sqrt(U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps)

        if self._error_propagator is None:
            return control_norms.flatten()

        stochastic_term = self._error_propagator.stochastic_control_term(problem_def, sol_data)
        return (control_norms + stochastic_term).flatten()
    
class ControlNormRegConstraint(Constraint):
    """c_Us. Always includes control_norms; if an error_propagator is given
    (StochasticTOP only), also adds its stochastic_control_term()."""

    def __init__(self, problem_def: ProblemDefinition, error_propagator: ErrorPropagator | None = None):
        self._error_propagator = error_propagator
        size = problem_def.dims.N_arcs
        super().__init__(name="c_Us", size=size, lower=-jnp.inf * jnp.ones(size), upper=jnp.ones(size))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        U_reg_arc_hst = sol_data['U_reg_arc_hst'].reshape(problem_def.dims.N_arcs, problem_def.dims.control_dim)
        control_norms = jnp.sum(U_reg_arc_hst ** 2, axis=1)

        if self._error_propagator is None:
            return control_norms.flatten()

        stochastic_term = self._error_propagator.stochastic_control_term(problem_def, sol_data)
        return (control_norms + stochastic_term).flatten()


class MeshConstraint(Constraint):
    """c_t_node_bound - linear constraint for the adaptive_fixedtof mesh.

    Pins t_node_bound[0] = 0 and t_node_bound[-1] = tf (fixed total time of
    flight) and bounds inter-node spacing to [dt_min, dt_max]. Purely linear
    in t_node_bound, so evaluate() is never called - TOP.evaluate() skips
    constraints with non-empty `linearities`.
    """

    def __init__(self, problem_def: ProblemDefinition):
        bc = problem_def.boundary_conditions

        mesh_con_terms = adaptive_mesh_con_terms(bc.t_node_bound)
        size = problem_def.dims.N_nodes + 1
        super().__init__(
            name="c_t_node_bound", size=size,
            lower=mesh_con_terms['lower'], upper=mesh_con_terms['upper'],
            linearities={'t_node_bound': mesh_con_terms['jac']},
        )


class CovarianceConstraint(Constraint):
    """c_P_Xf - terminal covariance constraint (StochasticTOP only).

    Pure bookkeeping - name/size/bounds are stable regardless of which
    ErrorPropagator produced error_state (raw covariance today, a future
    square-root or non-Gaussian representation later); evaluate() just
    forwards to terminal_constraint_value() on whichever ErrorPropagator
    instance StochasticTOP actually used to produce error_state, with zero
    reconstruction here.
    """

    def __init__(self, error_propagator: ErrorPropagator):
        self._error_propagator = error_propagator
        super().__init__(name="c_P_Xf", size=1, lower=-jnp.inf * jnp.ones(1), upper=jnp.zeros(1))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        return self._error_propagator.terminal_constraint_value(problem_def, sol_data)


class DetCollisionAvoidanceConstraint(Constraint):
    """c_det_col_avoid - deterministic collision avoidance at each trajectory
    node except the final one (size N_arcs*N_subarcs)."""

    def __init__(self, problem_def: ProblemDefinition):
        size = problem_def.dims.N_arcs * problem_def.dims.N_subarcs
        super().__init__(name="c_det_col_avoid", size=size, lower=-jnp.inf * jnp.ones(size), upper=jnp.zeros(size))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        ca = problem_def.collision_avoidance
        X_hst = sol_data['X_hst']

        N_nodes_minus_1 = dims.N_arcs * dims.N_subarcs
        X_node_hst = jnp.zeros((N_nodes_minus_1 + 1, 7))
        X_node_hst = X_node_hst.at[:-1, :].set(X_hst[:, :-1, :7].reshape(N_nodes_minus_1, 7))
        X_node_hst = X_node_hst.at[-1, :].set(X_hst[-1, -1, :7])

        dyn_args = {'r_obj': ca.r_obj, 'd_safe': ca.d_safe}
        return col_avoid_vmap(X_node_hst[:-1, :7], dyn_args).flatten()
