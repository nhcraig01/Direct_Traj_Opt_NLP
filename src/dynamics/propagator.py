"""Propagator hierarchy - 03_propagator.puml.

Propagator <|-- StatePropagator (used by DeterministicTOP)
Propagator <|-- StateSensitivityPropagator (used by StochasticTOP)
"""

import diffrax as dfx
import jax
import jax.numpy as jnp
from jax import Array

from .equations_of_motion import EquationsOfMotion
from src.problem.problem_definition import ProblemDefinition


def AB_rev2fwd(A_rev, B_rev):
    """Convert reverse-time arc sensitivities to forward-equivalent ones."""
    A_fwd = jnp.linalg.inv(A_rev)
    B_fwd = -A_fwd @ B_rev
    return A_fwd, B_fwd


AB_rev2fwd_vmap = jax.vmap(AB_rev2fwd, in_axes=(0, 0))


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

    def propagate(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        raise NotImplementedError


class StatePropagator(Propagator):
    """Pure state propagation - forward/backward to the midpoint, no sensitivities."""

    def propagate(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        bc = problem_def.boundary_conditions

        X0 = sol_data['X0']
        Xf = sol_data['Xf']
        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            t_node_bound = sol_data['t_node_bound']
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

        sol_data['X_hst'] = X_hst
        sol_data['t_hst'] = t_hst
        sol_data['t_node_bound'] = t_node_bound
        return sol_data


class StateSensitivityPropagator(Propagator):
    """State + first-order sensitivity propagation for StochasticTOP.

    A self-contained replacement for StatePropagator: propagates state
    identically (forward/backward to midpoint, same diffrax setup) but also
    computes A_hst/B_hst/A_arc_hst/B_arc_hst via jax.jacfwd of the endpoint
    propagator.

    Kept self-contained so that a future simultaneous state+sensitivity
    integration can be dropped in by changing only this class.
    """

    def __init__(self, problem_def: ProblemDefinition, eom: EquationsOfMotion):
        super().__init__(problem_def, eom)

        # Endpoint-only propagator: returns final state as (state_dim,) vector.
        # Used as the target for jax.jacfwd to get arc-level A (dX1/dX0) and
        # B (dX1/dU) matrices. Defined as a plain closure so jacfwd can be
        # applied cleanly without capturing `self` in the differentiated path.
        term = self._term
        solver = self._solver
        sc = self._stepsize_controller

        def _propagate_final(X0, U, t0, t1):
            sol = dfx.diffeqsolve(
                term, solver, t0, t1, None, X0,
                args=U, stepsize_controller=sc,
                adjoint=dfx.ForwardMode(),
                saveat=dfx.SaveAt(t1=True),
                max_steps=16 ** 5,
            )
            return sol.ys[-1].flatten()

        self._propagate_final = _propagate_final
        self._arc_dX0 = jax.jacfwd(_propagate_final, argnums=0)  # (state_dim, state_dim)
        self._arc_dU = jax.jacfwd(_propagate_final, argnums=1)   # (state_dim, control_dim)
        self._arc_dX0_vmap = jax.vmap(self._arc_dX0, in_axes=(0, None, 0, 0))
        self._arc_dU_vmap = jax.vmap(self._arc_dU, in_axes=(0, None, 0, 0))

    def propagate(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        bc = problem_def.boundary_conditions

        X0 = sol_data['X0']
        Xf = sol_data['Xf']
        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            t_node_bound = sol_data['t_node_bound']
        else:
            t_node_bound = bc.t_node_bound

        # Initialize all histories to zero
        X_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, dims.state_dim))
        A_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs, dims.state_dim, dims.state_dim))
        B_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs, dims.state_dim, dims.control_dim))
        A_arc_hst = jnp.zeros((dims.N_arcs, dims.state_dim, dims.state_dim))
        B_arc_hst = jnp.zeros((dims.N_arcs, dims.state_dim, dims.control_dim))

        def fwd_body(i, carry):
            X0_f, X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst = carry

            # State propagation
            X_arc = self._propagate_arc(X0_f, U_arc_hst[i], t_node_bound[i], t_node_bound[i + 1], dims.arc_length_opt)
            X_hst = X_hst.at[i, :, :].set(X_arc)

            # Arc-level state-transition and input-sensitivity matrices
            A_i = self._arc_dX0(X0_f, U_arc_hst[i], t_node_bound[i], t_node_bound[i + 1])
            B_i = self._arc_dU(X0_f, U_arc_hst[i], t_node_bound[i], t_node_bound[i + 1])
            A_arc_hst = A_arc_hst.at[i, :, :].set(A_i)
            B_arc_hst = B_arc_hst.at[i, :, :].set(B_i)

            # Sub-arc sensitivities
            t_arc = jnp.linspace(t_node_bound[i], t_node_bound[i + 1], dims.arc_length_opt)
            if dims.N_subarcs > 1:
                A_js = self._arc_dX0_vmap(X_arc[:-1], U_arc_hst[i], t_arc[:-1], t_arc[1:])
                B_js = self._arc_dU_vmap(X_arc[:-1], U_arc_hst[i], t_arc[:-1], t_arc[1:])
            else:
                A_js = A_i[None, :, :]
                B_js = B_i[None, :, :]
            A_hst = A_hst.at[i, :, :, :].set(A_js)
            B_hst = B_hst.at[i, :, :, :].set(B_js)

            return X_arc[-1], X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst

        _, X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst = jax.lax.fori_loop(
            0, len(bc.indx_f), fwd_body,
            (X0, X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst),
        )

        def bwd_body(ii, carry):
            X0_b, X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst = carry
            i = bc.indx_b[ii]

            # State propagation in reverse time, then flip to forward order
            X_arc_b = self._propagate_arc(X0_b, U_arc_hst[i], t_node_bound[i + 1], t_node_bound[i], dims.arc_length_opt)
            X_arc_f = jnp.flipud(X_arc_b)
            X_hst = X_hst.at[i, :, :].set(X_arc_f)

            # Arc-level sensitivities in reverse time, converted to forward-equivalent
            A_i_b = self._arc_dX0(X0_b, U_arc_hst[i], t_node_bound[i + 1], t_node_bound[i])
            B_i_b = self._arc_dU(X0_b, U_arc_hst[i], t_node_bound[i + 1], t_node_bound[i])
            A_i_f, B_i_f = AB_rev2fwd(A_i_b, B_i_b)
            A_arc_hst = A_arc_hst.at[i, :, :].set(A_i_f)
            B_arc_hst = B_arc_hst.at[i, :, :].set(B_i_f)

            # Sub-arc sensitivities (backward time, then converted + flipped)
            t_arc_b = jnp.linspace(t_node_bound[i + 1], t_node_bound[i], dims.arc_length_opt)
            if dims.N_subarcs > 1:
                A_js_b = self._arc_dX0_vmap(X_arc_b[:-1], U_arc_hst[i], t_arc_b[:-1], t_arc_b[1:])
                B_js_b = self._arc_dU_vmap(X_arc_b[:-1], U_arc_hst[i], t_arc_b[:-1], t_arc_b[1:])
                A_js_f, B_js_f = AB_rev2fwd_vmap(A_js_b, B_js_b)
                A_js_f = jnp.flipud(A_js_f)
                B_js_f = jnp.flipud(B_js_f)
            else:
                A_js_f = A_i_f[None, :, :]
                B_js_f = B_i_f[None, :, :]
            A_hst = A_hst.at[i, :, :, :].set(A_js_f)
            B_hst = B_hst.at[i, :, :, :].set(B_js_f)

            return X_arc_b[-1], X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst

        _, X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst = jax.lax.fori_loop(
            0, len(bc.indx_b), bwd_body,
            (Xf, X_hst, A_hst, B_hst, A_arc_hst, B_arc_hst),
        )

        t_hst = jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, dims.arc_length_opt))(t_node_bound[:-1], t_node_bound[1:])

        # Post-insertion coast STM (zero control, fixed duration tf_T) - used
        # by ErrorPropagator.terminal_constraint_value() for the c_P_Xf-style
        # terminal-covariance constraint. Same endpoint-jacfwd machinery as
        # the regular arcs, just one more call.
        A_postinsert = self._arc_dX0(
            Xf, jnp.zeros(dims.control_dim), t_node_bound[-1], t_node_bound[-1] + bc.tf_T,
        )

        sol_data['X_hst'] = X_hst
        sol_data['t_hst'] = t_hst
        sol_data['t_node_bound'] = t_node_bound
        sol_data['A_hst'] = A_hst
        sol_data['B_hst'] = B_hst
        sol_data['A_arc_hst'] = A_arc_hst
        sol_data['B_arc_hst'] = B_arc_hst
        sol_data['A_postinsert'] = A_postinsert
        return sol_data
