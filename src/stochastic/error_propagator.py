"""ErrorPropagator hierarchy.

Three methods, all called internally by each subclass rather than
orchestrated by StochasticTOP (keeps StochasticTOP.evaluate() branch-free):
  - init_error_state(): the internal initial condition fed into
    propagate_cov() - shape/meaning is private to the subclass (e.g.
    Phat_0 for TrueStateCovPropagator vs. the augmented Phat_0/Ptild_0
    block for a future EstimatedStateCovPropagator).
  - propagate_cov(): the main propagation loop, adding whatever covariance
    fields (P_hst, P_U_arc_hst, ...) this subclass produces to sol_data.
  - terminal_constraint_value(): the c_P_Xf-style "is the final error
    within the target bound" constraint value. Lives here (rather than as
    a standalone Constraint operating generically on sol_data) because
    different representations may need genuinely different formulations -
    e.g. a future square-root propagator computing this directly from
    S_hst rather than reconstructing P_hst, for numerical conditioning. A
    StochasticTOP-level Constraint subclass still owns the name/size/bounds
    bookkeeping and just forwards sol_data here with zero reconstruction.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .control_noise import gates2Gexe_vmap
from .measurement_model import MeasurementModel
from src.problem.problem_definition import ProblemDefinition
from src.utils.math_utils import mat_lmax, mat_lmax_vmap, smooth_val2val


class ErrorPropagator:
    """+init_error_state(problem_def) : array
    +propagate_cov(problem_def, sol_data) : dict
    +terminal_constraint_value(problem_def, sol_data) : array
    +stochastic_control_term(problem_def, sol_data) : array

    All subclasses share a uniform __init__(problem_def) so StochasticTOP can
    dispatch them by a single {name: cls} lookup with no per-type argument
    plumbing. Subclasses that need extra state (e.g. EstimatedStateCovPropagator
    building its MeasurementModel) construct it from problem_def alone; the
    ones that don't simply inherit this no-op.
    """

    def __init__(self, problem_def: ProblemDefinition):
        pass

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        raise NotImplementedError

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        raise NotImplementedError

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        raise NotImplementedError

    def stochastic_control_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        """Per-arc addition to the mean control norm (length N_arcs) -
        consumed by ControlNormConstraint and StochasticTOP.objective().
        Lives here rather than computed generically from P_U_arc_hst
        because representations differ in what's cheapest/best-conditioned
        to start from (e.g. a square-root propagator working directly from
        a control-gain sqrt factor rather than squaring it into a
        covariance first).
        """
        raise NotImplementedError


class TrueStateCovPropagator(ErrorPropagator):
    """True-state feedback ErrorPropagator ('true_state').

    propagate_cov() is per-arc/per-subarc fori_loop orchestration only -
    the actual covariance recursion is subarc_cov_step(), a standalone
    @staticmethod with no loop structure baked in. That split matters for
    reuse: legacy code calls the exact same subarc recursion
    (cov_propagators['subArc']) from objective_and_constraints (this
    propagator's job), sim_Det_traj (detailed-plotting granularity), and
    single_MC_trial - three different loop granularities, one shared
    formula. subarc_cov_step() preserves that: it can be called directly,
    with no TrueStateCovPropagator instance required, at whatever
    granularity a future solution-prep/MC-trial routine needs (including
    the post-insertion coast, by passing K_arc/G_exe_arc/G_stoch as zeros).

    Always uses the general tau/gam cross-correlation recursion, even for
    N_subarcs == 1 (legacy's cheaper single-step shortcut is algebraically
    identical there, since tau_0 = P0_arc and gam_0 = 0 makes the general
    formula collapse to the same thing) - one formula, no branching.
    """

    @staticmethod
    def subarc_cov_step(
        A_j: Array, B_j: Array, K_arc: Array, P_u_arc: Array, P0_arc: Array,
        P_j: Array, tau_j: Array, gam_j: Array,
    ) -> tuple[Array, Array, Array]:
        """One sub-arc covariance step: (P_j, tau_j, gam_j) -> (P_j1, tau_j1, gam_j1).

        Pure function of its arguments - no instance state, no loop
        structure - so it's reusable verbatim at any granularity/context.
        """
        tau_gam_term = A_j @ (tau_j @ K_arc.T + gam_j) @ B_j.T
        P_j1 = A_j @ P_j @ A_j.T + B_j @ (P_u_arc + K_arc @ P0_arc @ K_arc.T) @ B_j.T + tau_gam_term + tau_gam_term.T
        tau_j1 = A_j @ tau_j + B_j @ K_arc @ P0_arc
        gam_j1 = A_j @ gam_j + B_j @ P_u_arc
        return P_j1, tau_j1, gam_j1

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        return problem_def.uncertainty.Phat_0

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        unc = problem_def.uncertainty

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        P_exe_arc_hst, _ = gates2Gexe_vmap(U_arc_hst, unc.gates)
        K_arc_hst = sol_data['K_arc_hst']
        A_hst = sol_data['A_hst']
        B_hst = sol_data['B_hst']

        P_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, dims.state_dim, dims.state_dim))
        P_U_arc_hst = jnp.zeros((dims.N_arcs, dims.control_dim, dims.control_dim))

        def arc_body(i, carry):
            P0_arc, P_hst, P_U_arc_hst = carry
            P_hst = P_hst.at[i, 0, :, :].set(P0_arc)

            K_arc = K_arc_hst[i]
            P_U_arc_hst = P_U_arc_hst.at[i, :, :].set(K_arc @ P0_arc @ K_arc.T)
            P_u_arc = unc.G_stoch + P_exe_arc_hst[i]

            def subarc_body(j, carry2):
                P_js, tau_j, gam_j = carry2
                P_j1, tau_j1, gam_j1 = self.subarc_cov_step(
                    A_hst[i, j], B_hst[i, j], K_arc, P_u_arc, P0_arc, P_js[j], tau_j, gam_j,
                )
                P_js = P_js.at[j + 1, :, :].set(P_j1)
                return P_js, tau_j1, gam_j1

            tau_0 = P0_arc
            gam_0 = jnp.zeros((dims.state_dim, dims.control_dim))
            P_js, _, _ = jax.lax.fori_loop(0, dims.N_subarcs, subarc_body, (P_hst[i], tau_0, gam_0))
            P_hst = P_hst.at[i, :, :, :].set(P_js)

            return P_hst[i, -1, :, :], P_hst, P_U_arc_hst

        P0_init = self.init_error_state(problem_def)
        _, P_hst, P_U_arc_hst = jax.lax.fori_loop(0, dims.N_arcs, arc_body, (P0_init, P_hst, P_U_arc_hst))

        sol_data['P_hst'] = P_hst
        sol_data['P_U_arc_hst'] = P_U_arc_hst
        return sol_data

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        unc = problem_def.uncertainty

        A_postinsert = sol_data['A_postinsert']
        P_Xf_full = sol_data['P_hst'][-1, -1, :, :]

        S_Xf_targ_inv = unc.S_XT_targ_inv @ A_postinsert
        tmp_P_Xf_con_val = S_Xf_targ_inv @ P_Xf_full @ S_Xf_targ_inv.T - jnp.eye(dims.state_dim)
        return jnp.log10(mat_lmax(tmp_P_Xf_con_val) + 1)

    def stochastic_control_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        return problem_def.uncertainty.mx_tcm_bound * jnp.sqrt(mat_lmax_vmap(sol_data['P_U_arc_hst']))


class TrueStateSqrtCovPropagator(ErrorPropagator):
    """Square-root form of true-state feedback covariance propagation."""

    @staticmethod
    def subarc_sqrt_step(A_j: Array, B_j: Array, K_arc: Array, Saug_j: Array) -> Array:
        """One subarc step for z_j = [x_j; x0_arc; w_arc].

        w_arc (this arc's frozen process+execution noise draw) is already
        part of the augmented state rather than re-injected each step, so
        this is a pure linear map - no new noise to combine, hence no QR
        recompression here. That only happens once per arc, in
        propagate_cov, when the marginal block gets pulled back out.
        """
        n, m = A_j.shape[0], B_j.shape[1]
        A_aug = jnp.block([
            [A_j,               B_j @ K_arc,       B_j],
            [jnp.zeros((n, n)), jnp.eye(n),        jnp.zeros((n, m))],
            [jnp.zeros((m, n)), jnp.zeros((m, n)), jnp.eye(m)],
        ])
        return A_aug @ Saug_j

    @staticmethod
    def _augment_arc_initial(S0_arc: Array, S_w_arc: Array) -> Array:
        """Build an arc's initial augmented sqrt for z = [x_j; x0_arc; w_arc]
        from the marginal state sqrt S0_arc and this arc's frozen-noise sqrt
        S_w_arc (S_w_arc @ S_w_arc.T == G_stoch + G_exe_arc for this arc).

        x_j and its frozen arc-initial copy x0_arc are the SAME random
        variable at the start of the arc (full correlation, not
        independence, hence S0_arc repeated rather than block_diag'd);
        w_arc is independent of both, entering only through the third
        row block.
        """
        n = S0_arc.shape[0]
        zeros_top = jnp.zeros((n, S_w_arc.shape[1]))
        zeros_bot = jnp.zeros((S_w_arc.shape[0], n))
        return jnp.block([[S0_arc, zeros_top],
                           [S0_arc, zeros_top],
                           [zeros_bot, S_w_arc]])

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        S_0 = jnp.linalg.cholesky(problem_def.uncertainty.Phat_0, upper=False)
        return S_0

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        unc = problem_def.uncertainty

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        _, S_exe_arc_hst = gates2Gexe_vmap(U_arc_hst, unc.gates)
        K_arc_hst = sol_data['K_arc_hst']
        A_hst = sol_data['A_hst']
        B_hst = sol_data['B_hst']

        # G_stoch is constant across the trajectory, so its square root is
        # computed once here rather than inside the loop.
        S_stoch = jnp.linalg.cholesky(unc.G_stoch, upper=False)

        n = dims.state_dim
        S_w_width = S_stoch.shape[1] + S_exe_arc_hst.shape[-1]
        aug_rows = 2 * n + dims.control_dim
        aug_cols = n + S_w_width
        Saug_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, aug_rows, aug_cols))
        # K_arc @ S0_arc, not squared into a covariance - kept in sqrt scale
        # since that's the whole point of this propagator (see
        # stochastic_control_term(), which needs exactly this).
        KS_arc_hst = jnp.zeros((dims.N_arcs, dims.control_dim, n))

        def arc_body(i, carry):
            S0_arc, Saug_hst, KS_arc_hst = carry

            K_arc = K_arc_hst[i]
            KS = K_arc @ S0_arc
            KS_arc_hst = KS_arc_hst.at[i, :, :].set(KS)

            # S_w_arc @ S_w_arc.T == G_stoch + G_exe_arc == P_u_arc for this
            # arc - reproduced directly from the sqrt pieces already in
            # hand, no new Cholesky beyond the constant S_stoch above.
            S_w_arc = jnp.block([[S_stoch, S_exe_arc_hst[i]]])
            Saug0_arc = self._augment_arc_initial(S0_arc, S_w_arc)
            Saug_hst = Saug_hst.at[i, 0, :, :].set(Saug0_arc)

            def subarc_body(j, Saug_js):
                Saug_j1 = self.subarc_sqrt_step(A_hst[i, j], B_hst[i, j], K_arc, Saug_js[j])
                Saug_js = Saug_js.at[j + 1, :, :].set(Saug_j1)
                return Saug_js

            Saug_js = jax.lax.fori_loop(0, dims.N_subarcs, subarc_body, Saug_hst[i])
            Saug_hst = Saug_hst.at[i, :, :, :].set(Saug_js)

            # Retriangularize: this arc's marginal sqrt of Cov(x_{N_subarcs})
            # is wide (rows of Saug_js[-1], all columns) since Saug is no
            # longer triangular - compress it back to a compact (n,n)
            # square root here, once per arc, before it becomes the next
            # arc's S0. Without this the augmented width would grow every
            # arc instead of staying fixed at aug_cols.
            S_marginal_wide = Saug_js[-1, :n, :]
            S0_arc_next = jnp.linalg.qr(S_marginal_wide.T, mode='r').T
            return S0_arc_next, Saug_hst, KS_arc_hst

        S0_init = self.init_error_state(problem_def)
        _, Saug_hst, KS_arc_hst = jax.lax.fori_loop(0, dims.N_arcs, arc_body, (S0_init, Saug_hst, KS_arc_hst))

        sol_data['Saug_hst'] = Saug_hst
        sol_data['KS_arc_hst'] = KS_arc_hst
        return sol_data

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n = dims.state_dim

        A_postinsert = sol_data['A_postinsert']
        S_Xf_full = sol_data['Saug_hst'][-1, -1, :n, :]  # rows only - Saug isn't triangular here

        S_Xf_targ_inv = unc.S_XT_targ_inv @ A_postinsert
        M = S_Xf_targ_inv @ S_Xf_full

        # lam_max(M@M.T - I) + 1 == lam_max(M@M.T) == sigma_max(M)**2, so
        # going straight to the largest singular value of M (valid for any
        # shape, no need to square into M@M.T first) gives the same
        # feasibility boundary with half the log-scale magnitude.
        sigma_max = jnp.linalg.norm(M, ord=2)
        return jnp.log10(sigma_max)

    def stochastic_control_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        # Largest singular value of K_arc @ S0_arc directly, rather than
        # squaring it into a covariance and taking sqrt(max eigenvalue) -
        # avoids the square/un-square round trip that hurts scale, the
        # whole reason for using the sqrt form in the first place.
        sigma_max = jax.vmap(lambda KS: jnp.linalg.norm(KS, ord=2))(sol_data['KS_arc_hst'])
        return problem_def.uncertainty.mx_tcm_bound * sigma_max


class EstimatedStateCovPropagator(ErrorPropagator):
    """Estimated-state feedback ErrorPropagator ('estimated_state').

    Feedback acts on the EKF-estimated state rather than the true state, so
    the error dynamics carry BOTH the estimate-dispersion covariance (Phat)
    and the estimation-error covariance (Ptild) plus their cross-correlation.
    These are stacked into a 14x14 augmented covariance
    Paug = [[Phat, Phat~], [Phat~.T, Ptild]] propagated through each subarc
    with an interleaved EKF measurement update (underweighted Joseph form).

    Like TrueStateCovPropagator, the actual recursion lives in a standalone
    @staticmethod (subarc_ekf_step) with no loop structure baked in, so it
    can be reused verbatim at other granularities (detailed plotting, MC
    trials, post-insertion coast) - the Kalman gains it produces (L_arc_hst)
    are exactly what a per-trial MC EKF needs.

    The MeasurementModel is built from problem_def alone at construction, so
    nothing upstream has to thread it in - see the class docstring on
    ErrorPropagator.
    """

    def __init__(self, problem_def: ProblemDefinition):
        self._meas = MeasurementModel.build(problem_def)

    @staticmethod
    def subarc_ekf_step(
        A_j: Array, B_j: Array, K_arc: Array, P_u_arc: Array, Paug0_arc: Array,
        H_j1: Array, P_v_j1: Array, Paug_j: Array, tau_j: Array, gam_j: Array,
        update: float = 1.0,
    ) -> tuple[Array, Array, Array, Array]:
        """One subarc: augmented time-propagation + EKF measurement update.

        (Paug_j, tau_j, gam_j) -> (Paug_j1, tau_j1, gam_j1), also returning
        the Kalman gain L_j1 used at node j+1. Pure function of its arguments
        - no instance state, no loop structure.

        `update` gates the measurement update (matches legacy update_js): 1.0
        applies it (the optimization default, updating at every subarc node),
        0.0 makes the step pure time-propagation with L_j1 == 0 (used by the
        MC per-trial EKF, which propagates the covariance across a subarc's
        detailed micro-steps and only updates at the subarc end).
        """
        n, m = A_j.shape[0], B_j.shape[1]

        A_aug = jnp.block([[A_j, jnp.zeros((n, n))],
                           [jnp.zeros((n, n)), A_j]])
        B_aug = jnp.block([[B_j @ K_arc, jnp.zeros((n, n))],
                           [jnp.zeros((n, n)), jnp.zeros((n, n))]])
        C_aug = jnp.block([[jnp.zeros((n, m))],
                           [B_j]])

        # Time propagation of the augmented covariance + P0_arc/process-noise
        # cross-correlation carriers (tau/gam), same structure as the
        # true-state tau/gam recursion but in augmented (2n) space.
        tmp_a = A_aug @ tau_j @ B_aug.T
        tmp_b = A_aug @ gam_j @ C_aug.T
        Paug_j1m = (A_aug @ Paug_j @ A_aug.T + tmp_a + tmp_a.T
                    + B_aug @ Paug0_arc @ B_aug.T - tmp_b - tmp_b.T
                    + C_aug @ P_u_arc @ C_aug.T)
        tau_j1m = A_aug @ tau_j + B_aug @ Paug0_arc
        gam_j1m = A_aug @ gam_j - C_aug @ P_u_arc

        # EKF measurement update on the estimation-error block, with adaptive
        # underweighting (p_apply) that softens the update when the predicted
        # residual covariance is small relative to measurement noise.
        Ptild_j1m = Paug_j1m[n:, n:]
        p_test = 5 / 6
        val_under = (jnp.linalg.trace(H_j1 @ Ptild_j1m @ H_j1.T)
                     - p_test / (1 - p_test) * jnp.linalg.trace(P_v_j1))
        p_apply = smooth_val2val(val_under, a=100, val1=p_test, val2=1.0)

        C_j1 = Ptild_j1m @ H_j1.T
        W_j1 = (1 / p_apply) * H_j1 @ Ptild_j1m @ H_j1.T + P_v_j1
        L_j1 = update * jax.scipy.linalg.solve(W_j1.T, C_j1.T).T

        D_aug = jnp.block([[jnp.eye(n), -L_j1 @ H_j1],
                           [jnp.zeros((n, n)), jnp.eye(n) - L_j1 @ H_j1]])
        F_aug = jnp.block([[L_j1],
                           [L_j1]])
        Paug_j1p = D_aug @ Paug_j1m @ D_aug.T + F_aug @ P_v_j1 @ F_aug.T
        tau_j1 = D_aug @ tau_j1m
        gam_j1 = D_aug @ gam_j1m
        return Paug_j1p, tau_j1, gam_j1, L_j1

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        unc = problem_def.uncertainty
        n = problem_def.dims.state_dim
        return jnp.block([[unc.Phat_0, jnp.zeros((n, n))],
                          [jnp.zeros((n, n)), unc.Ptild_0]])

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n = dims.state_dim

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        P_exe_arc_hst, _ = gates2Gexe_vmap(U_arc_hst, unc.gates)
        K_arc_hst = sol_data['K_arc_hst']
        A_hst = sol_data['A_hst']
        B_hst = sol_data['B_hst']
        X_hst = sol_data['X_hst']  # (N_arcs, N_subarcs+1, n)

        # Measurement Jacobians / noise covariances at every subarc node,
        # evaluated at the nominal states (vmap over arcs, then over nodes).
        H_hst = jax.vmap(self._meas.H_vmap)(X_hst)        # (N_arcs, N_subarcs+1, n_meas, n)
        P_v_hst = jax.vmap(self._meas.P_v_vmap)(X_hst)    # (N_arcs, N_subarcs+1, n_meas, n_meas)

        n_meas = self._meas.n_meas
        Paug_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, 2 * n, 2 * n))
        P_U_arc_hst = jnp.zeros((dims.N_arcs, dims.control_dim, dims.control_dim))
        L_arc_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, n, n_meas))

        def arc_body(i, carry):
            Paug0_arc, Paug_hst, P_U_arc_hst, L_arc_hst = carry
            Paug_hst = Paug_hst.at[i, 0, :, :].set(Paug0_arc)

            K_arc = K_arc_hst[i]
            # Control-dispersion covariance uses the estimate block (Phat),
            # since feedback acts on the estimated state.
            P_U_arc_hst = P_U_arc_hst.at[i, :, :].set(K_arc @ Paug0_arc[:n, :n] @ K_arc.T)
            P_u_arc = unc.G_stoch + P_exe_arc_hst[i]

            def subarc_body(j, carry2):
                Paug_js, L_js, tau_j, gam_j = carry2
                Paug_j1, tau_j1, gam_j1, L_j1 = self.subarc_ekf_step(
                    A_hst[i, j], B_hst[i, j], K_arc, P_u_arc, Paug0_arc,
                    H_hst[i, j + 1], P_v_hst[i, j + 1], Paug_js[j], tau_j, gam_j,
                )
                Paug_js = Paug_js.at[j + 1, :, :].set(Paug_j1)
                L_js = L_js.at[j + 1, :, :].set(L_j1)
                return Paug_js, L_js, tau_j1, gam_j1

            tau_0 = Paug0_arc
            gam_0 = jnp.zeros((2 * n, dims.control_dim))
            Paug_js, L_js, _, _ = jax.lax.fori_loop(
                0, dims.N_subarcs, subarc_body, (Paug_hst[i], L_arc_hst[i], tau_0, gam_0),
            )
            Paug_hst = Paug_hst.at[i, :, :, :].set(Paug_js)
            L_arc_hst = L_arc_hst.at[i, :, :, :].set(L_js)

            return Paug_js[-1], Paug_hst, P_U_arc_hst, L_arc_hst

        Paug0_init = self.init_error_state(problem_def)
        _, Paug_hst, P_U_arc_hst, L_arc_hst = jax.lax.fori_loop(
            0, dims.N_arcs, arc_body, (Paug0_init, Paug_hst, P_U_arc_hst, L_arc_hst),
        )

        sol_data['Paug_hst'] = Paug_hst
        sol_data['P_U_arc_hst'] = P_U_arc_hst
        sol_data['L_arc_hst'] = L_arc_hst   # Kalman gains, for MC-trial EKF reuse
        sol_data['H_hst'] = H_hst
        sol_data['P_v_hst'] = P_v_hst
        return sol_data

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n = dims.state_dim

        A_postinsert = sol_data['A_postinsert']
        Paug_Xf = sol_data['Paug_hst'][-1, -1, :, :]
        Phat_f = Paug_Xf[:n, :n]
        Ptild_f = Paug_Xf[n:, n:]
        Phattild_f = Paug_Xf[:n, n:]
        # True-state deviation covariance from the augmented blocks (same
        # reconstruction legacy uses for c_P_Xf).
        P_Xf_full = Phat_f - Phattild_f - Phattild_f.T + Ptild_f

        S_Xf_targ_inv = unc.S_XT_targ_inv @ A_postinsert
        tmp_P_Xf_con_val = S_Xf_targ_inv @ P_Xf_full @ S_Xf_targ_inv.T - jnp.eye(n)
        return jnp.log10(mat_lmax(tmp_P_Xf_con_val) + 1)

    def stochastic_control_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        return problem_def.uncertainty.mx_tcm_bound * jnp.sqrt(mat_lmax_vmap(sol_data['P_U_arc_hst']))
