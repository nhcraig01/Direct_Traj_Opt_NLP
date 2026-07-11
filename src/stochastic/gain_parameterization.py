"""Gain parameterization hierarchy - 07_gain_parameterization.puml.

GainParameterization <|-- LQRArcGains   (Gain_Type == 'arc_lqr')
GainParameterization <|-- LQRFullTrajGains (Gain_Type == 'fulltraj_lqr')

Each subclass owns its optimization variable (gain_weights, size N_arcs*2)
and its compute_gains() implementation. compute_gains() takes sol_data
(already carrying A_arc_hst/B_arc_hst from StateSensitivityPropagator) and
problem_def, and returns sol_data with K_arc_hst added.

StochasticTOP merges gain_param.variables() into its own variables() list
and calls gain_param.compute_gains() between propagation and covariance
propagation (see 09_stochastic_top_composition.puml).
"""

import jax
import jax.numpy as jnp

from src.problem.data_structures import OptimizationVariable
from src.problem.problem_definition import ProblemDefinition


class GainParameterization:
    """+variables() : list[OptimizationVariable]
    +compute_gains(problem_def, sol_data) : dict
    """

    def variables(self) -> list[OptimizationVariable]:
        raise NotImplementedError

    def compute_gains(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        raise NotImplementedError


class LQRArcGains(GainParameterization):
    """Per-arc LQR gains ('arc_lqr').

    gain_weights: (N_arcs*2,) — one [xi_r, xi_v] pair per arc, each
    independently parameterizing the position/velocity weight block in the
    LQR cost. Each arc's K is solved independently via a one-step matrix
    inverse (no backward Riccati sweep).
    """

    def __init__(self, problem_def: ProblemDefinition):
        self._problem_def = problem_def

    def variables(self) -> list[OptimizationVariable]:
        dims = self._problem_def.dims
        size = dims.N_arcs * 2
        default_value = 1e-4 * jnp.ones(size)
        return [
            OptimizationVariable(
                name='gain_weights', size=size,
                lower=jnp.full(size, 1e-6), upper=jnp.full(size, jnp.inf),
                value=default_value, guess_fn=lambda key: default_value,
            ),
        ]

    def compute_gains(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        eps = 1e-12

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        gain_weights = sol_data['gain_weights'].reshape(dims.N_arcs, 2)
        control_norms = jnp.sqrt(
            U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps
        )

        P_targ_inv = problem_def.uncertainty.P_XT_targ_inv

        def xi2K(xi_k, A_k, B_k, U_mag):
            P_trg_rr_inv = P_targ_inv[:3, :3]
            P_trg_vv_inv = P_targ_inv[3:6, 3:6]
            weights = jax.scipy.linalg.block_diag(xi_k[0] * P_trg_rr_inv, xi_k[1] * P_trg_vv_inv)
            K_k = -jnp.linalg.inv(jnp.eye(3) + B_k[:6, :].T @ weights @ B_k[:6, :]) @ B_k[:6, :].T @ weights @ A_k[:6, :6]
            return jnp.hstack([K_k, jnp.zeros((3, 1))])

        xi2K_vmap = jax.vmap(xi2K, in_axes=(0, 0, 0, 0))
        K_arc_hst = xi2K_vmap(
            gain_weights,
            sol_data['A_arc_hst'],
            sol_data['B_arc_hst'],
            control_norms,
        )
        sol_data['K_arc_hst'] = K_arc_hst
        return sol_data


class LQRFullTrajGains(GainParameterization):
    """Full-trajectory LQR gains via backward Riccati sweep ('fulltraj_lqr').

    gain_weights: (N_arcs*2,) — one [xi_r, xi_v] pair per arc, shared as
    state-cost weights in a discrete-time LQR Riccati recursion sweeping
    backward from the final arc. All arcs contribute to the terminal cost
    and feed forward into earlier K values.
    """

    def __init__(self, problem_def: ProblemDefinition):
        self._problem_def = problem_def

    def variables(self) -> list[OptimizationVariable]:
        dims = self._problem_def.dims
        size = dims.N_arcs * 2
        default_value = 1e-5 * jnp.ones(size)
        return [
            OptimizationVariable(
                name='gain_weights', size=size,
                lower=jnp.full(size, 1e-6), upper=jnp.full(size, jnp.inf),
                value=default_value, guess_fn=lambda key: default_value,
            ),
        ]

    def compute_gains(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        eps = 1e-12

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        gain_weights = sol_data['gain_weights'].reshape(dims.N_arcs, 2)
        control_norms = jnp.sqrt(
            U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps
        )

        P_targ_inv = problem_def.uncertainty.P_XT_targ_inv
        xi_r = gain_weights[:, 0]
        xi_v = gain_weights[:, 1]

        A_rv = sol_data['A_arc_hst'][:, :6, :6]
        B_rv = sol_data['B_arc_hst'][:, :6, :]

        R_i = jnp.eye(dims.control_dim)
        P_trg_rr_inv = P_targ_inv[:3, :3]
        P_trg_vv_inv = P_targ_inv[3:6, 3:6]

        # Terminal state-cost matrix (same formula as all intermediate stages)
        Q_N = jax.scipy.linalg.block_diag(xi_r[-1] * P_trg_rr_inv, xi_v[-1] * P_trg_vv_inv) / 10

        K_arc_hst = jnp.zeros((dims.N_arcs, dims.control_dim, dims.state_dim))
        index_back = jnp.arange(dims.N_arcs - 1, -1, -1)

        def iterate_K(ii, carry):
            K_arc_hst, S_i1 = carry
            i = index_back[ii]

            A_i = A_rv[i]
            B_i = B_rv[i]

            Q_i = jax.scipy.linalg.block_diag(xi_r[i] * P_trg_rr_inv, xi_v[i] * P_trg_vv_inv) / 10

            M_i = R_i + B_i.T @ S_i1 @ B_i
            tmp = jnp.linalg.solve(M_i, B_i.T @ S_i1)

            K_i = -tmp @ A_i
            K_arc_hst = K_arc_hst.at[i, :, :6].set(K_i)

            S_i = A_i.T @ (S_i1 - S_i1 @ B_i @ tmp) @ A_i + Q_i
            return K_arc_hst, S_i

        K_arc_hst, _ = jax.lax.fori_loop(0, dims.N_arcs, iterate_K, (K_arc_hst, Q_N))
        sol_data['K_arc_hst'] = K_arc_hst
        return sol_data
