"""Constraint hierarchy - 04_constraints.puml.

Constraints do NOT contribute optimization variables - every variable is
declared in one place by TOP.variables() (see 05_top_overview.puml).
"""

import jax.numpy as jnp
from jax import Array

from Lib.math import adaptive_mesh_con_terms, col_avoid_vmap, mat_lmax_vmap
from traj_opt_oop_rework.data_structures import ErrorState, TrajectoryState
from traj_opt_oop_rework.problem_definition import ProblemDefinition


class Constraint:
    """+name, +size, +lower, +upper, +linearities."""

    def __init__(self, name: str, size: int, lower: Array, upper: Array, linearities: dict | None = None):
        self.name = name
        self.size = size
        self.lower = lower
        self.upper = upper
        self.linearities = linearities if linearities is not None else {}

    def evaluate(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> Array:
        raise NotImplementedError


class BoundaryConstraint(Constraint):
    """c_X0 / c_Xf - one instance per boundary, constructed twice by build_problem()."""

    def __init__(self, problem_def: ProblemDefinition, boundary: str):
        self._problem_def = problem_def
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

    def evaluate(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> Array:
        if self._boundary == "initial":
            X0 = inputs['X0']
            alpha = inputs['alpha']
            return (X0[:7] - jnp.concatenate([self._orbit_family.evaluate(alpha).flatten(), jnp.array([1.0])])).flatten()
        else:
            Xf = inputs['Xf']
            beta = inputs['beta']
            return (Xf[:6] - self._orbit_family.evaluate(beta)).flatten()


class MatchPointConstraint(Constraint):
    """c_X_mp - identical for DeterministicTOP and StochasticTOP; only reads traj_state."""

    def __init__(self, problem_def: ProblemDefinition):
        self._problem_def = problem_def
        size = problem_def.dims.state_dim
        super().__init__(name="c_X_mp", size=size, lower=jnp.zeros(size), upper=jnp.zeros(size))

    def evaluate(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> Array:
        bc = self._problem_def.boundary_conditions
        return (traj_state.X_hst[bc.indx_f[-1], -1, :7] - traj_state.X_hst[bc.indx_b[-1], 0, :7]).flatten()


class ControlNormConstraint(Constraint):
    """c_Us. Always includes control_norms; if error_state is not None, also adds
    control_max_eig from error_state.P_U_arc_hst."""

    def __init__(self, problem_def: ProblemDefinition):
        self._problem_def = problem_def
        size = problem_def.dims.N_arcs
        super().__init__(name="c_Us", size=size, lower=-jnp.inf * jnp.ones(size), upper=jnp.ones(size))

    def evaluate(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> Array:
        eps = 1e-12
        U_arc_hst = inputs['U_arc_hst'].reshape(self._problem_def.dims.N_arcs, self._problem_def.dims.control_dim)
        control_norms = jnp.sqrt(U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps)

        if error_state is None:
            return control_norms.flatten()

        control_max_eig = self._problem_def.uncertainty.mx_tcm_bound * jnp.sqrt(mat_lmax_vmap(error_state.P_U_arc_hst))
        return (control_norms + control_max_eig).flatten()


class MeshConstraint(Constraint):
    """c_t_node_bound - linear constraint for the adaptive_fixedtof mesh.

    Pins t_node_bound[0] = 0 and t_node_bound[-1] = tf (fixed total time of
    flight) and bounds inter-node spacing to [dt_min, dt_max]. Purely linear
    in t_node_bound, so evaluate() is never called - TOP.evaluate() skips
    constraints with non-empty `linearities`.
    """

    def __init__(self, problem_def: ProblemDefinition):
        self._problem_def = problem_def
        bc = problem_def.boundary_conditions

        mesh_con_terms = adaptive_mesh_con_terms(bc.t_node_bound)
        size = problem_def.dims.N_nodes + 1
        super().__init__(
            name="c_t_node_bound", size=size,
            lower=mesh_con_terms['lower'], upper=mesh_con_terms['upper'],
            linearities={'t_node_bound': mesh_con_terms['jac']},
        )


class DetCollisionAvoidanceConstraint(Constraint):
    """c_det_col_avoid - deterministic collision avoidance at each trajectory
    node except the final one (size N_arcs*N_subarcs)."""

    def __init__(self, problem_def: ProblemDefinition):
        self._problem_def = problem_def
        size = problem_def.dims.N_arcs * problem_def.dims.N_subarcs
        super().__init__(name="c_det_col_avoid", size=size, lower=-jnp.inf * jnp.ones(size), upper=jnp.zeros(size))

    def evaluate(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> Array:
        dims = self._problem_def.dims
        ca = self._problem_def.collision_avoidance
        X_hst = traj_state.X_hst

        N_nodes_minus_1 = dims.N_arcs * dims.N_subarcs
        X_node_hst = jnp.zeros((N_nodes_minus_1 + 1, 7))
        X_node_hst = X_node_hst.at[:-1, :].set(X_hst[:, :-1, :7].reshape(N_nodes_minus_1, 7))
        X_node_hst = X_node_hst.at[-1, :].set(X_hst[-1, -1, :7])

        dyn_args = {'r_obj': ca.r_obj, 'd_safe': ca.d_safe}
        return col_avoid_vmap(X_node_hst[:-1, :7], dyn_args).flatten()
