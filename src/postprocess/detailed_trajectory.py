"""Detailed deterministic trajectory + covariance generator (OOP rewrite of the
legacy Lib.dynamics.sim_Det_traj, plus the orbit histories from prepare_sol).

Given a solved case (problem_def + its StochasticTOP/DeterministicTOP + the
optimizer xStar), this replays the nominal trajectory at *detailed* resolution
(N_save points per subarc), linearizes it (A/B), computes the feedback gains,
propagates the state-error covariance (P for true_state, the 14x14 augmented
Paug for estimated_state) along it, and derives the control-norm bounds, dV
statistics and terminal target covariance. It reuses the same rework primitives
the MC runners use - `Propagator._propagate_arc`/`_arc_dX0_vmap`, the gain
parameterization, and the `ErrorPropagator` subarc cov/EKF steps - so there is
no duplicated math, just detailed-resolution orchestration.

Returns a nested dict {Name, Orb0, Orbf, Det} carrying everything the results
writer (src/io.py) needs; the field names inside 'Det' mirror the legacy
sim_Det_traj contract so parity can be checked directly.
"""

import jax
import jax.numpy as jnp

from src.stochastic.control_noise import gates2Gexe
from src.stochastic.error_propagator import TrueStateCovPropagator, EstimatedStateCovPropagator
from src.utils.math_utils import cart2sph_vmap, mat_lmax


def _orbit_history(prop, X, t_hst):
    """Propagate an orbit from state X over its period at the given save times."""
    ys = prop._propagate_arc(X, jnp.zeros(3), 0.0, float(t_hst[-1]), t_hst.shape[0])
    return ys, t_hst


def generate_detailed_data(problem_def, top, xStar):
    dims = problem_def.dims
    unc = problem_def.uncertainty
    bc = problem_def.boundary_conditions
    n, m = dims.state_dim, dims.control_dim
    N_arcs, N_save = dims.N_arcs, dims.N_save
    arc_len = dims.arc_length_det          # detailed points per arc (incl. bounds)
    transfer_len = dims.transfer_length_det
    post_len = dims.post_insert_length
    length = dims.length

    stochastic = problem_def.toggles.problem_type.lower() == 'stochastic_gauss_zoh'
    estimated = problem_def.toggles.feedback_control_type.lower() == 'estimated_state'

    # Coerce to jax - xStar loaded from sol.h5 is numpy, and the gain fori_loop
    # indexes it with a traced loop variable (fine under the optimizer's jit,
    # but not eager on numpy arrays).
    xStar = {k: jnp.asarray(v) for k, v in xStar.items()}

    prop = top._propagator
    X0, Xf = xStar['X0'], xStar['Xf']
    U_arc_hst = xStar['U_arc_hst'].reshape(N_arcs, m)
    if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
        t_node = xStar['t_node_bound']
    else:
        t_node = bc.t_node_bound

    # ----- Phase 1: detailed state + linearization per arc ----------------- #
    X_hst = jnp.zeros((length, n)).at[0].set(X0)
    t_hst = jnp.zeros((length,)).at[0].set(t_node[0])
    U_hst = jnp.zeros((length, m))
    X_node_hst = jnp.zeros((N_arcs + 1, n)).at[0].set(X0)
    A_hst = jnp.zeros((length - 1, n, n))
    B_hst = jnp.zeros((length - 1, n, m))
    A_arc_hst = jnp.zeros((N_arcs, n, n))
    B_arc_hst = jnp.zeros((N_arcs, n, m))

    X0_arc = X0
    for k in range(N_arcs):
        i0, i_f = k * (arc_len - 1), (k + 1) * (arc_len - 1)
        U_arc = U_arc_hst[k]
        X_arc = prop._propagate_arc(X0_arc, U_arc, t_node[k], t_node[k + 1], arc_len)
        t_arc = jnp.linspace(t_node[k], t_node[k + 1], arc_len)

        X_hst = X_hst.at[i0:i_f + 1].set(X_arc)
        t_hst = t_hst.at[i0:i_f + 1].set(t_arc)
        U_hst = U_hst.at[i0:i_f].set(jnp.tile(U_arc, (arc_len - 1, 1)))
        X_node_hst = X_node_hst.at[k + 1].set(X_arc[-1])

        if stochastic:
            A_arc_hst = A_arc_hst.at[k].set(prop._arc_dX0(X0_arc, U_arc, t_node[k], t_node[k + 1]))
            B_arc_hst = B_arc_hst.at[k].set(prop._arc_dU(X0_arc, U_arc, t_node[k], t_node[k + 1]))
            A_hst = A_hst.at[i0:i_f].set(prop._arc_dX0_vmap(X_arc[:-1], U_arc, t_arc[:-1], t_arc[1:]))
            B_hst = B_hst.at[i0:i_f].set(prop._arc_dU_vmap(X_arc[:-1], U_arc, t_arc[:-1], t_arc[1:]))
        X0_arc = X_arc[-1]

    # ----- Phase 1b: post-insertion coast (state + sensitivities) ---------- #
    pi0 = transfer_len - 1
    X0_pi, t0_pi = X_hst[pi0], t_hst[pi0]
    X_pi = prop._propagate_arc(X0_pi, jnp.zeros(m), t0_pi, t0_pi + bc.tf_T, post_len)
    t_pi = jnp.linspace(t0_pi, t0_pi + bc.tf_T, post_len)
    X_hst = X_hst.at[pi0 + 1:].set(X_pi[1:])
    t_hst = t_hst.at[pi0 + 1:].set(t_pi[1:])
    if stochastic:
        A_hst = A_hst.at[pi0:].set(prop._arc_dX0_vmap(X_pi[:-1], jnp.zeros(m), t_pi[:-1], t_pi[1:]))
        B_hst = B_hst.at[pi0:].set(prop._arc_dU_vmap(X_pi[:-1], jnp.zeros(m), t_pi[:-1], t_pi[1:]))

    # ----- orbit histories + dV_mean (always) ------------------------------ #
    orb0_X, orb0_t = _orbit_history(prop, X0, bc.Orb0_t_hst)
    orbf_X, orbf_t = _orbit_history(prop, Xf, bc.Orbf_t_hst)

    U_hst_sph = cart2sph_vmap(U_hst)
    dt_hst = jnp.diff(t_hst)
    Vs, U_acc = problem_def.Sys['Vs'], problem_def.spacecraft.U_Acc_min_nd
    dV_mean = jnp.sum(jnp.linalg.norm(U_hst[:-1], axis=1) * U_acc / X_hst[:-1, -1] * dt_hst) * Vs

    det = {'X_hst': X_hst, 'X_node_hst': X_node_hst, 'U_hst': U_hst, 'U_hst_sph': U_hst_sph,
           'U_arc_hst': U_arc_hst, 't_hst': t_hst, 't_node_hst': t_node, 'dV_mean': dV_mean,
           'length_transfer': transfer_len, 'length_arc': arc_len}
    out = {'Name': problem_def.name, 'Orb0': {'X_hst': orb0_X, 't_hst': orb0_t},
           'Orbf': {'X_hst': orbf_X, 't_hst': orbf_t}, 'Det': det}
    if not stochastic:
        return out

    # ----- Phase 2: gains ------------------------------------------------- #
    gain_sol = top._gain_param.compute_gains(problem_def, {
        'U_arc_hst': xStar['U_arc_hst'], 'gain_weights': xStar['gain_weights'],
        'A_arc_hst': A_arc_hst, 'B_arc_hst': B_arc_hst,
    })
    K_arc_hst = gain_sol['K_arc_hst']
    gain_weights = xStar['gain_weights'].reshape(N_arcs, 2)

    # ----- Phase 3: covariance along the detailed trajectory --------------- #
    P_hst = jnp.zeros((length, n, n))
    K_hst = jnp.zeros((length - 1, m, n))
    gain_weights_hst = jnp.zeros((length, 2))
    P_u_hst = jnp.zeros((length - 1, m, m))
    TCM_norm_bound_hst = jnp.zeros((length,))
    TCM_norm_dV_hst = jnp.zeros((length,))
    U_norm_bound_hst = jnp.zeros((length,))
    U_norm_dV_hst = jnp.zeros((length,))
    if estimated:
        Paug_hst = jnp.zeros((length, 2 * n, 2 * n))
        L_hst = jnp.zeros((length, n, top._error_propagator._meas.n_meas))
        H_hst = jnp.zeros((length, top._error_propagator._meas.n_meas, n))
        P_v_hst = jnp.zeros((length, top._error_propagator._meas.n_meas, top._error_propagator._meas.n_meas))
        Paug_hst = Paug_hst.at[0].set(top._error_propagator.init_error_state(problem_def))
        meas = top._error_propagator._meas
        ekf_step = EstimatedStateCovPropagator.subarc_ekf_step
    else:
        P_hst = P_hst.at[0].set(unc.Phat_0)
        cov_step = TrueStateCovPropagator.subarc_cov_step

    for k in range(N_arcs):
        i0, i_f = k * (arc_len - 1), (k + 1) * (arc_len - 1)
        K_arc, U_arc = K_arc_hst[k], U_arc_hst[k]
        P_exe_arc, _ = gates2Gexe(U_arc, unc.gates)
        P_u_arc = unc.G_stoch + P_exe_arc

        # control-dispersion covariance (Phat block for estimated) -> bounds
        if estimated:
            P0_ctrl = Paug_hst[i0][:n, :n]
        else:
            P0_ctrl = P_hst[i0]
        P_u_ctrl = K_arc @ P0_ctrl @ K_arc.T
        tcm_bound = unc.mx_tcm_bound * jnp.sqrt(mat_lmax(P_u_ctrl))
        tcm_dV = unc.mx_dV_bound * jnp.sqrt(mat_lmax(P_u_ctrl))
        u_norm = jnp.linalg.norm(U_arc)

        gain_weights_hst = gain_weights_hst.at[i0:i_f].set(jnp.tile(gain_weights[k], (arc_len - 1, 1)))
        K_hst = K_hst.at[i0:i_f].set(jnp.tile(K_arc, (arc_len - 1, 1, 1)))
        P_u_hst = P_u_hst.at[i0:i_f].set(jnp.tile(P_u_ctrl, (arc_len - 1, 1, 1)))
        TCM_norm_bound_hst = TCM_norm_bound_hst.at[i0:i_f].set(tcm_bound)
        TCM_norm_dV_hst = TCM_norm_dV_hst.at[i0:i_f].set(tcm_dV)
        U_norm_bound_hst = U_norm_bound_hst.at[i0:i_f].set(u_norm + tcm_bound)
        U_norm_dV_hst = U_norm_dV_hst.at[i0:i_f].set(u_norm + tcm_dV)

        if estimated:
            H_arc = meas.H_vmap(X_hst[i0:i_f + 1])
            P_v_arc = meas.P_v_vmap(X_hst[i0:i_f + 1])
            H_hst = H_hst.at[i0:i_f + 1].set(H_arc)
            P_v_hst = P_v_hst.at[i0:i_f + 1].set(P_v_arc)
            Paug0_arc = Paug_hst[i0]
            tau, gam = Paug0_arc, jnp.zeros((2 * n, m))
            for j in range(arc_len - 1):
                upd = jnp.where((j + 1) % (N_save - 1) == 0, 1.0, 0.0)
                Paug1, tau, gam, L1 = ekf_step(
                    A_hst[i0 + j], B_hst[i0 + j], K_arc, P_u_arc, Paug0_arc,
                    H_hst[i0 + j + 1], P_v_hst[i0 + j + 1], Paug_hst[i0 + j], tau, gam, upd)
                Paug_hst = Paug_hst.at[i0 + j + 1].set(Paug1)
                L_hst = L_hst.at[i0 + j + 1].set(L1)
        else:
            P0_arc = P_hst[i0]
            tau, gam = P0_arc, jnp.zeros((n, m))
            for j in range(arc_len - 1):
                P1, tau, gam = cov_step(A_hst[i0 + j], B_hst[i0 + j], K_arc, P_u_arc,
                                        P0_arc, P_hst[i0 + j], tau, gam)
                P_hst = P_hst.at[i0 + j + 1].set(P1)

    # ----- Phase 3b: post-insertion covariance (K=0, no noise/update) ------ #
    if estimated:
        H_pi = meas.H_vmap(X_pi)
        P_v_pi = meas.P_v_vmap(X_pi)
        H_hst = H_hst.at[pi0:].set(H_pi)
        P_v_hst = P_v_hst.at[pi0:].set(P_v_pi)
        Paug0_pi = Paug_hst[pi0]
        tau, gam = Paug0_pi, jnp.zeros((2 * n, m))
        for j in range(post_len - 1):
            Paug1, tau, gam, _ = ekf_step(
                A_hst[pi0 + j], B_hst[pi0 + j], jnp.zeros((m, n)), jnp.zeros((m, m)), Paug0_pi,
                H_hst[pi0 + j + 1], P_v_hst[pi0 + j + 1], Paug_hst[pi0 + j], tau, gam, 0.0)
            Paug_hst = Paug_hst.at[pi0 + j + 1].set(Paug1)
        Phat_hst = Paug_hst[:, :n, :n]
        Ptild_hst = Paug_hst[:, n:, n:]
        Phattild_hst = Paug_hst[:, :n, n:]
        P_hst = Phat_hst - Phattild_hst - Phattild_hst.swapaxes(-1, -2) + Ptild_hst
    else:
        P0_pi = P_hst[pi0]
        tau, gam = P0_pi, jnp.zeros((n, m))
        for j in range(post_len - 1):
            P1, tau, gam = cov_step(A_hst[pi0 + j], B_hst[pi0 + j], jnp.zeros((m, n)),
                                    jnp.zeros((m, m)), P0_pi, P_hst[pi0 + j], tau, gam)
            P_hst = P_hst.at[pi0 + j + 1].set(P1)

    # ----- Phase 4: terminal target covariance ----------------------------- #
    Af = prop._arc_dX0(Xf, jnp.zeros(m), t0_pi, t0_pi + bc.tf_T)
    Af_inv = jnp.linalg.inv(Af)
    P_XT_targ = unc.P_XT_targ
    P_Xf_targ = Af_inv @ P_XT_targ @ Af_inv.T
    P_Targ_hst = jnp.zeros((post_len, n, n)).at[0].set(P_Xf_targ)
    if estimated:
        P_Targ_hst = P_Targ_hst.at[-1].set(P_XT_targ)
    else:
        tau, gam = P_Xf_targ, jnp.zeros((n, m))
        for j in range(post_len - 1):
            P1, tau, gam = cov_step(A_hst[pi0 + j], B_hst[pi0 + j], jnp.zeros((m, n)),
                                    jnp.zeros((m, m)), P_Xf_targ, P_Targ_hst[j], tau, gam)
            P_Targ_hst = P_Targ_hst.at[j + 1].set(P1)

    # ----- dV statistics + assemble ---------------------------------------- #
    dV_stat = jnp.sum(TCM_norm_dV_hst[:-1] * U_acc / X_hst[:-1, -1] * dt_hst) * Vs
    dV_bound = jnp.sum(U_norm_dV_hst[:-1] * U_acc / X_hst[:-1, -1] * dt_hst) * Vs

    det.update({'TCM_norm_dV_hst': TCM_norm_dV_hst, 'TCM_norm_bound_hst': TCM_norm_bound_hst,
                'U_norm_dV_hst': U_norm_dV_hst, 'U_norm_bound_hst': U_norm_bound_hst,
                'dV_stat': dV_stat, 'dV_bound': dV_bound, 'A_hst': A_hst, 'B_hst': B_hst,
                'K_hst': K_hst, 'gain_weights_hst': gain_weights_hst, 'K_arc_hst': K_arc_hst,
                'P_hst': P_hst, 'P_u_hst': P_u_hst, 'P_Xf_targ': P_Xf_targ,
                'P_XT_targ': P_XT_targ, 'P_Targ_hst': P_Targ_hst})
    if estimated:
        det.update({'Phat_hst': Phat_hst, 'Ptild_hst': Ptild_hst, 'Phattild_hst': Phattild_hst,
                    'H_hst': H_hst, 'L_hst': L_hst, 'P_v_hst': P_v_hst})
    return out
