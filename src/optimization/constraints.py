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

from .control_representation import ControlRepresentation
from src.problem.problem_definition import ProblemDefinition
from src.stochastic.error_propagator import ErrorPropagator
from src.utils.math_utils import col_avoid_vmap


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
    """c_Us. Representation-generic (see ControlRepresentation in
    control_representation.py): the deterministic-only nominal
    control_norm_term when stochastic is False (plain DeterministicTOP),
    otherwise the representation's full stochastic_constraint_term -
    cartesian adds a TCM bound to the nominal norm, regularized replaces it
    entirely with a unified quantile bound (the mean command is already
    folded in there, so there's no separate nominal term to add)."""

    def __init__(self, problem_def: ProblemDefinition, control_repr: ControlRepresentation,
                 stochastic: bool = False):
        self._control_repr = control_repr
        self._stochastic = stochastic
        size = problem_def.dims.N_arcs
        super().__init__(name="c_Us", size=size, lower=-jnp.inf * jnp.ones(size), upper=jnp.ones(size))

    def evaluate(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        if not self._stochastic:
            return self._control_repr.control_norm_term(problem_def, sol_data).flatten()

        return self._control_repr.stochastic_constraint_term(problem_def, sol_data).flatten()

class MeshConstraint(Constraint):
    """c_arc_length - total time of flight, for the adaptive_fixedtof mesh.

    One row: sum(arc_length_hst) == tf. With arc durations as the optimization
    variable, node monotonicity is just each duration's own lower bound, so the
    bidiagonal difference rows the node-time parameterization needed are gone
    (42 rows on 41 variables -> 1 row on 40). Purely linear, so evaluate() is
    never called - TOP.evaluate() skips constraints with non-empty
    `linearities`.

    Only added for adaptive_fixedtof: when the mesh is fixed the durations are
    pinned by their own bounds, and this row would then be linearly dependent
    on those - a degenerate active set.
    """

    def __init__(self, problem_def: ProblemDefinition):
        bc = problem_def.boundary_conditions
        tf = bc.t_node_bound[-1] - bc.t_node_bound[0]
        super().__init__(
            name="c_arc_length", size=1,
            lower=jnp.atleast_1d(tf), upper=jnp.atleast_1d(tf),
            linearities={'arc_length_hst': jnp.ones((1, problem_def.dims.N_arcs))},
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
