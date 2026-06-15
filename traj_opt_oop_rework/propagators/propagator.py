"""Propagator hierarchy - 03_propagator.puml.

Propagator <|-- StatePropagator (used by DeterministicTOP)
Propagator <|-- SensitivityPropagator (used by StochasticTOP, not yet implemented)
"""

import diffrax as dfx
import jax
import jax.numpy as jnp
from jax import Array

from traj_opt_oop_rework.data_structures import TrajectoryState
from traj_opt_oop_rework.dynamics.equations_of_motion import EquationsOfMotion
from traj_opt_oop_rework.problem_definition import ProblemDefinition


class Propagator:
    """-propagate_arc(X0, U, t0, t1, prop_length) : array"""

    def __init__(self, problem_def: ProblemDefinition, eom: EquationsOfMotion):
        self._problem_def = problem_def
        self._eom = eom
        self._term = dfx.ODETerm(eom.eom)
        self._solver = dfx.Dopri8()
        self._stepsize_controller = dfx.PIDController(
            rtol=problem_def.integration.r_tol, atol=problem_def.integration.a_tol,
        )

    def _propagate_arc(self, X0: Array, U: Array, t0: float, t1: float, prop_length: int) -> Array:
        """Integrate one arc, saving `prop_length` states evenly spaced in time over [t0, t1]."""
        save_t = dfx.SaveAt(ts=jnp.linspace(t0, t1, prop_length))
        sol = dfx.diffeqsolve(
            self._term,
            self._solver,
            t0,
            t1,
            None,
            X0,
            args=U,
            stepsize_controller=self._stepsize_controller,
            adjoint=dfx.ForwardMode(),
            saveat=save_t,
            max_steps=16 ** 5,
        )
        return sol.ys

    def propagate(self, inputs: dict, problem_def: ProblemDefinition):
        raise NotImplementedError


class StatePropagator(Propagator):
    """Pure state propagation - forward/backward to the midpoint, no sensitivities."""

    def propagate(self, inputs: dict, problem_def: ProblemDefinition) -> tuple[TrajectoryState, None]:
        dims = problem_def.dims
        bc = problem_def.boundary_conditions

        X0 = inputs['X0']
        Xf = inputs['Xf']
        U_arc_hst = inputs['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            t_node_bound = inputs['t_node_bound']
        else:
            t_node_bound = bc.t_node_bound

        X_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, dims.state_dim))

        def fwd_body(i, carry):
            X0_true_f, X_hst = carry
            X_arc = self._propagate_arc(X0_true_f, U_arc_hst[i, :], t_node_bound[i], t_node_bound[i + 1], dims.arc_length_opt)
            X_hst = X_hst.at[i, :, :].set(X_arc)
            return X_arc[-1, :], X_hst

        X0_true_f, X_hst = jax.lax.fori_loop(0, len(bc.indx_f), fwd_body, (X0, X_hst))

        def bwd_body(ii, carry):
            X0_true_b, X_hst = carry
            i = bc.indx_b[ii]
            X_arc_b = self._propagate_arc(X0_true_b, U_arc_hst[i, :], t_node_bound[i + 1], t_node_bound[i], dims.arc_length_opt)
            X_arc_f = jnp.flipud(X_arc_b)
            X_hst = X_hst.at[i, :, :].set(X_arc_f)
            return X_arc_b[-1, :], X_hst

        X0_true_b, X_hst = jax.lax.fori_loop(0, len(bc.indx_b), bwd_body, (Xf, X_hst))

        t_hst = jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, dims.arc_length_opt))(t_node_bound[:-1], t_node_bound[1:])

        return TrajectoryState(X_hst=X_hst, t_hst=t_hst, t_node_bound=t_node_bound), None
