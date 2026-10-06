"""ErrorPropagator hierarchy.

Three methods, all called internally by each subclass rather than
orchestrated by StochasticTOP (keeps StochasticTOP.evaluate() branch-free):
  - init_error_state(): the internal initial condition fed into
    propagate_cov() - shape/meaning is private to the subclass (e.g.
    Phat_0 for TrueStateCovPropagator vs. the augmented Phat_0/Ptild_0
    block for a future EstimatedStateCovPropagator).
  - propagate_cov(): the main propagation loop, adding whatever covariance
    fields (P_hst, P_control_arc_hst, ...) this subclass produces to sol_data.
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
from src.utils.math_utils import mat_lmax, smooth_val2val


def _lifted_transition_matrices(A_arc_hst: Array, B_arc_hst: Array,
                                Hq_arc_hst: Array, K_arc_hst: Array) -> tuple[Array, Array]:
    """Trajectory-wide ("lifted") transition matrices A_big, B_big for

        X = [dx_0; dx_1; ...; dx_N] = A_big @ dx_0 + B_big @ U
        U = [du_0; ...; du_{N-1}]     (one frozen noise draw per arc, ZOH)

    A_big stacks [I; Acl_0; Acl_1 Acl_0; ...] and B_big is block lower
    triangular with block (i, j) = Acl_{i-1}...Acl_{j+1} Phi_u,j - the
    transition from arc j's control to state i. Closed loop per arc is
    Acl_i = Phi_x,i + Phi_u,i Hq_i K_i.

    One step per arc is exact for true-state feedback: with no measurement
    update there is nothing forcing sub-arc granularity, and the control is
    ZOH (one command, one execution-error draw per arc), so the arc-level
    sensitivities A_arc_hst/B_arc_hst carry the whole story. Both are built
    by one forward sweep - each row is the previous row pushed through Acl,
    with the new arc's Phi_u written into its own column block.
    """
    N, n = A_arc_hst.shape[0], A_arc_hst.shape[1]
    m = B_arc_hst.shape[2]
    Acl = A_arc_hst + B_arc_hst @ Hq_arc_hst @ K_arc_hst        # (N, n, n)

    A_big = jnp.zeros(((N + 1) * n, n)).at[:n, :].set(jnp.eye(n))
    B_big = jnp.zeros(((N + 1) * n, N * m))

    def row_body(i, carry):
        A_row, B_row, A_big, B_big = carry
        A_row = Acl[i] @ A_row
        B_row = jax.lax.dynamic_update_slice(Acl[i] @ B_row, B_arc_hst[i], (0, i * m))
        A_big = jax.lax.dynamic_update_slice(A_big, A_row, ((i + 1) * n, 0))
        B_big = jax.lax.dynamic_update_slice(B_big, B_row, ((i + 1) * n, 0))
        return A_row, B_row, A_big, B_big

    _, _, A_big, B_big = jax.lax.fori_loop(
        0, N, row_body, (jnp.eye(n), jnp.zeros((n, N * m)), A_big, B_big))
    return A_big, B_big


def _lifted_gain_matrix(K_arc_hst: Array) -> Array:
    """K_big = blkdiag(K_0, ..., K_{N-1}), shape (N*m, N*n).

    dQ = [dq_0; ...; dq_{N-1}] = K_big @ X[:N*n], i.e. it acts on every
    state block EXCEPT the last - dx_N has no control after it, so the final
    block row/column of the lifted state covariance is dropped before
    applying this. (dx_N is what terminal_constraint_value uses instead.)

    Note Hq does NOT appear here - dq is the deviation in the control
    variable's own units, which is what the control-norm constraint and the
    Patnaik fit both want."""
    N, m, n = K_arc_hst.shape
    K_big = jnp.zeros((N * m, N * n))

    def body(i, K_big):
        return jax.lax.dynamic_update_slice(K_big, K_arc_hst[i], (i * m, i * n))

    return jax.lax.fori_loop(0, N, body, K_big)


class ErrorPropagator:
    """+init_error_state(problem_def) : array
    +propagate_cov(problem_def, sol_data) : dict
    +terminal_constraint_value(problem_def, sol_data) : array

    All subclasses share a uniform __init__(problem_def) so StochasticTOP can
    dispatch them by a single {name: cls} lookup with no per-type argument
    plumbing. Subclasses that need extra state (e.g. EstimatedStateCovPropagator
    building its MeasurementModel) construct it from problem_def alone; the
    ones that don't simply inherit this no-op.

    Control-dispersion contract: propagate_cov() stashes the per-arc
    covariance of the feedback command (K_arc applied to the arc-initial
    state covariance) in whichever form is natural for the subclass -
    full-covariance classes write P_control_arc_hst (N_arcs, m, m),
    square-root classes write S_control_arc_hst (N_arcs, m, k) with
    S @ S.T == P. ControlRepresentation consumes whichever is present and
    forms its bound accordingly, so nothing here needs to know about
    control representation or quantile levels.
    """

    def __init__(self, problem_def: ProblemDefinition):
        pass

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        raise NotImplementedError

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        raise NotImplementedError

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
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
        A_j: Array, B_j: Array, Hq_arc: Array, K_arc: Array, P_u_arc: Array, P0_arc: Array,
        P_j: Array, tau_j: Array, gam_j: Array,
    ) -> tuple[Array, Array, Array]:
        """One sub-arc covariance step: (P_j, tau_j, gam_j) -> (P_j1, tau_j1, gam_j1).

        Pure function of its arguments - no instance state, no loop
        structure - so it's reusable verbatim at any granularity/context.

        Hq_arc (d(U_cart)/d(U_repr), identity for cartesian control) only
        applies where B_j pairs with the feedback gain K_arc - the closed
        loop command is Hq_arc @ K_arc @ dx, not K_arc @ dx directly. It does
        NOT apply where B_j pairs with P_u_arc (process/execution-error
        noise, injected directly in cartesian units regardless of control
        representation) - see control_representation.py. tau_j (feedback
        cross-correlation) and gam_j (noise cross-correlation) were already
        separate accumulators; P_j1/tau_gam_term combined the two under one
        B_j @ (...) sandwich, so those need to split into a raw-B_j noise
        term and a (B_j @ Hq_arc) feedback term rather than one shared B_j.
        """
        B_fb_j = B_j @ Hq_arc
        tau_gam_term = A_j @ tau_j @ K_arc.T @ B_fb_j.T + A_j @ gam_j @ B_j.T
        P_j1 = (A_j @ P_j @ A_j.T + B_j @ P_u_arc @ B_j.T
                + B_fb_j @ (K_arc @ P0_arc @ K_arc.T) @ B_fb_j.T + tau_gam_term + tau_gam_term.T)
        tau_j1 = A_j @ tau_j + B_fb_j @ K_arc @ P0_arc
        gam_j1 = A_j @ gam_j + B_j @ P_u_arc
        return P_j1, tau_j1, gam_j1

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        return problem_def.uncertainty.Phat_0

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        unc = problem_def.uncertainty

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        P_exe_arc_hst, _ = gates2Gexe_vmap(U_arc_hst, unc.gates)

        A_big, B_big = _lifted_transition_matrices(
            sol_data['A_arc_hst'], sol_data['B_arc_hst'],
            sol_data['Hq_arc_hst'], sol_data['K_arc_hst'],
        )
        K_big = _lifted_gain_matrix(sol_data['K_arc_hst'])

        # Block diagonal of each arc's process + execution noise. Cross-arc
        # blocks are zero under the usual independent-draws assumption; the
        # cross-correlations that matter downstream come from the shared
        # trajectory, and B_big carries those.
        P_U = jax.scipy.linalg.block_diag(*(unc.G_stoch + P_exe_arc_hst))
        P_x0 = self.init_error_state(problem_def)

        P_X = A_big @ P_x0 @ A_big.T + B_big @ P_U @ B_big.T
        # Drop the final block row/column (dx_N has no control after it)
        # before mapping states to feedback commands.
        n_ctrl = dims.N_arcs * dims.state_dim
        sol_data['P_X'] = P_X
        sol_data['P_Q'] = K_big @ P_X[:n_ctrl, :n_ctrl] @ K_big.T
        return sol_data

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n = dims.state_dim

        A_postinsert = sol_data['A_postinsert']
        P_Xf_full = sol_data['P_X'][-n:, -n:]   # final state's diagonal block

        S_Xf_targ_inv = unc.S_XT_targ_inv @ A_postinsert
        tmp_P_Xf_con_val = S_Xf_targ_inv @ P_Xf_full @ S_Xf_targ_inv.T - jnp.eye(n)
        return jnp.log10(mat_lmax(tmp_P_Xf_con_val) + 1)


class TrueStateSqrtCovPropagator(ErrorPropagator):
    """Square-root form of true-state feedback covariance propagation."""

    @staticmethod
    def subarc_sqrt_step(A_j: Array, B_j: Array, Hq_arc: Array, K_arc: Array, Saug_j: Array) -> Array:
        """One subarc step for z_j = [w_arc; x0_arc; x_j].
        w_arc (this arc's frozen process+execution noise draw) is already
        part of the augmented state rather than re-injected each step, so
        this is a pure linear map.

        Hq_arc (d(U_cart)/d(U_repr), identity for cartesian control) applies
        only to the x0_arc column, which carries the feedback path - the
        closed-loop command is Hq_arc @ K_arc @ dx. The w_arc column is the
        noise path (process/execution error, injected in cartesian units
        regardless of representation) and keeps the raw B_j. Same split as
        TrueStateCovPropagator.subarc_cov_step, but the block structure here
        keeps the two paths in separate columns already, so no algebraic
        splitting is needed.
        """
        n, m = A_j.shape[0], B_j.shape[1]
        A_aug = jnp.block([[jnp.eye(m),            jnp.zeros((m, n)),      jnp.zeros((m, n))],
                            [jnp.zeros((n, m)),     jnp.eye(n),             jnp.zeros((n, n))],
                            [B_j,                   B_j @ Hq_arc @ K_arc,   A_j]])
        return A_aug @ Saug_j

    @staticmethod
    def _augment_arc_initial(S0_arc: Array, S_w_arc: Array) -> Array:
        """Build an arc's initial augmented sqrt for z = [w_arc; x0_arc; x_j],
        literally from items 1-6 of the derivation: item 1 (S11=S_w_arc) in
        row 1, items 2/3 (S21=S31=0) and 4/5 (S22=S32=S0_arc) and 6 (S33=0)
        in rows 2/3.
        """
        n = S0_arc.shape[0]
        m, k = S_w_arc.shape

        zeros_col1 = jnp.zeros((n, k))
        zeros_owncol = jnp.zeros((n, n))

        return jnp.block([
            [S_w_arc,    jnp.zeros((m, n)), jnp.zeros((m, n))],  # item 1
            [zeros_col1, S0_arc,            zeros_owncol],        # items 2, 4
            [zeros_col1, S0_arc,            zeros_owncol],        # items 3, 5, 6
        ])

    def init_error_state(self, problem_def: ProblemDefinition) -> Array:
        S_0 = jnp.linalg.cholesky(problem_def.uncertainty.Phat_0, upper=False)
        return S_0

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n, m = dims.state_dim, dims.control_dim

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        _, S_exe_arc_hst = gates2Gexe_vmap(U_arc_hst, unc.gates)

        A_big, B_big = _lifted_transition_matrices(
            sol_data['A_arc_hst'], sol_data['B_arc_hst'],
            sol_data['Hq_arc_hst'], sol_data['K_arc_hst'],
        )
        K_big = _lifted_gain_matrix(sol_data['K_arc_hst'])

        # Per-arc square noise factor: S_w @ S_w.T == G_stoch + G_exe_arc,
        # QR-compressed straight from the two sqrt pieces rather than
        # Cholesky-ing their sum. Block-diagonal across arcs (independent
        # draws), same assumption the covariance form makes.
        S_stoch = jnp.linalg.cholesky(unc.G_stoch, upper=False)
        S_w_arc_hst = jax.vmap(
            lambda S_exe: jnp.linalg.qr(jnp.block([[S_stoch, S_exe]]).T, mode='r').T
        )(S_exe_arc_hst)
        S_U = jax.scipy.linalg.block_diag(*S_w_arc_hst)

        S_x0 = self.init_error_state(problem_def)
        # S_X @ S_X.T == A_big P_x0 A_big.T + B_big P_U B_big.T, never
        # forming either product. No QR here: the factor is ((N+1)n, n+N*m),
        # already minimal-width since its rank is bounded by the column
        # count, so a QR would be size-neutral - and it would have to run on
        # the transposed (wide) orientation, which JAX cannot differentiate
        # (it yields NaN gradients rather than an error).
        S_X = jnp.block([[A_big @ S_x0, B_big @ S_U]])

        # Drop the final block row (dx_N has no control after it) before
        # mapping states to feedback commands.
        n_ctrl = dims.N_arcs * n
        sol_data['S_X'] = S_X
        sol_data['S_Q'] = K_big @ S_X[:n_ctrl, :]
        return sol_data

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n = dims.state_dim

        A_postinsert = sol_data['A_postinsert']
        S_Xf_full = sol_data['S_X'][-n:, :]   # final state's block row

        S_Xf_targ_inv = unc.S_XT_targ_inv @ A_postinsert
        M = S_Xf_targ_inv @ S_Xf_full

        # lam_max(M@M.T - I) + 1 == lam_max(M@M.T) == sigma_max(M)**2
        sigma_max = jnp.linalg.norm(M, ord=2)
        return jnp.log(sigma_max)


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
        A_j: Array, B_j: Array, Hq_arc: Array, K_arc: Array, P_u_arc: Array, Paug0_arc: Array,
        H_j1: Array, P_v_j1: Array, Paug_j: Array, tau_j: Array, gam_j: Array,
        update: float = 1.0,
    ) -> tuple[Array, Array, Array, Array]:
        """One subarc: augmented time-propagation + EKF measurement update.

        (Paug_j, tau_j, gam_j) -> (Paug_j1, tau_j1, gam_j1), also returning
        the Kalman gain L_j1 used at node j+1. Pure function of its arguments
        - no instance state, no loop structure.

        Hq_arc (d(U_cart)/d(U_repr), identity for cartesian control) only
        applies to B_aug, which carries the feedback path (B_j paired with
        K_arc) - not to C_aug, which carries the noise path (B_j paired with
        P_u_arc, injected directly in cartesian units regardless of control
        representation). Same split rationale as TrueStateCovPropagator.
        subarc_cov_step.

        `update` gates the measurement update (matches legacy update_js): 1.0
        applies it (the optimization default, updating at every subarc node),
        0.0 makes the step pure time-propagation with L_j1 == 0 (used by the
        MC per-trial EKF, which propagates the covariance across a subarc's
        detailed micro-steps and only updates at the subarc end).
        """
        n, m = A_j.shape[0], B_j.shape[1]

        A_aug = jnp.block([[A_j, jnp.zeros((n, n))],
                           [jnp.zeros((n, n)), A_j]])
        B_aug = jnp.block([[B_j @ Hq_arc @ K_arc, jnp.zeros((n, n))],
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
        # p_apply = smooth_val2val(val_under, a=100, val1=p_test, val2=1.0)
        p_apply = 1.0

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
        P_control_arc_hst = jnp.zeros((dims.N_arcs, dims.control_dim, dims.control_dim))
        L_arc_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, n, n_meas))

        def arc_body(i, carry):
            Paug0_arc, Paug_hst, P_control_arc_hst, L_arc_hst = carry
            Paug_hst = Paug_hst.at[i, 0, :, :].set(Paug0_arc)

            K_arc = K_arc_hst[i]
            # Control-dispersion covariance uses the estimate block (Phat),
            # since feedback acts on the estimated state.
            P_control_arc_hst = P_control_arc_hst.at[i, :, :].set(K_arc @ Paug0_arc[:n, :n] @ K_arc.T)
            P_u_arc = unc.G_stoch + P_exe_arc_hst[i]
            Hq_arc = sol_data['Hq_arc_hst'][i]

            def subarc_body(j, carry2):
                Paug_js, L_js, tau_j, gam_j = carry2
                Paug_j1, tau_j1, gam_j1, L_j1 = self.subarc_ekf_step(
                    A_hst[i, j], B_hst[i, j], Hq_arc, K_arc, P_u_arc, Paug0_arc,
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

            return Paug_js[-1], Paug_hst, P_control_arc_hst, L_arc_hst

        Paug0_init = self.init_error_state(problem_def)
        _, Paug_hst, P_control_arc_hst, L_arc_hst = jax.lax.fori_loop(
            0, dims.N_arcs, arc_body, (Paug0_init, Paug_hst, P_control_arc_hst, L_arc_hst),
        )

        sol_data['Paug_hst'] = Paug_hst
        sol_data['P_control_arc_hst'] = P_control_arc_hst
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


class EstimatedStateSqrtCovPropagator(ErrorPropagator):
    """Square-root form of estimated-state feedback covariance propagation.

    Extends TrueStateSqrtCovPropagator's approach: augmented vector
    z = [w_arc; xhat_0; xtilde_j; xhat_j] (this arc's frozen noise, frozen
    arc-initial estimate, estimation error xtilde=xhat-xtrue, estimate),
    block-lower-triangular time-propagation (A_aug), and an interleaved
    underweighted-EKF measurement update (D_aug/F_aug) at every subarc node.

    One real structural difference from the true-state case: a measurement
    update injects new independent randomness at *every* subarc node, not
    once per arc - so Saug needs a QR recompression after every update,
    not just at the arc boundary. Every compression here (this class and
    TrueStateSqrtCovPropagator alike) uses the same one-line identity:
    S_new = qr(M.T, mode='r').T satisfies S_new @ S_new.T = M @ M.T
    exactly, regardless of how the columns get redistributed internally -
    called directly wherever needed rather than through a wrapper, since
    it's just that one jax call. Saug is kept as one square,
    block-lower-triangular (Cholesky-style) factor at all times, squaring
    down to (n_z, n_z) (n_z = m+3n) after every subarc - a real design
    choice worth a second look, not just mechanical translation of the
    derivation. S_w_arc (this arc's process+execution-noise sqrt) is
    itself QR-compressed to square (m, m) before use (see propagate_cov),
    same as TrueStateSqrtCovPropagator - which is also what makes the
    arc-initial state's "own-generator" gap in _augment_arc_initial land
    on a clean, literal (n, n) zero block instead of an arbitrary pad.
    """

    def __init__(self, problem_def: ProblemDefinition):
        self._meas = MeasurementModel.build(problem_def)

    @staticmethod
    def subarc_sqrt_step(
        A_j: Array, B_j: Array, Hq_arc: Array, K_arc: Array, Saug_j: Array,
        H_j1: Array, S_v_j1: Array,
    ) -> Array:
        """One subarc: time propagation, underweighted EKF measurement
        update, then QR recompression back to square.
        z = [w_arc; xhat_0; xtilde_j; xhat_j].

        Hq_arc (identity for cartesian control) applies only to the xhat_0
        column, which carries the feedback path. The w_arc column is the
        noise path and keeps the raw B_j - same split as
        TrueStateSqrtCovPropagator.subarc_sqrt_step.
        """
        n = H_j1.shape[1]
        m = B_j.shape[1]

        A_aug = jnp.block([
            [jnp.eye(m),        jnp.zeros((m, n)),      jnp.zeros((m, n)), jnp.zeros((m, n))],
            [jnp.zeros((n, m)), jnp.eye(n),             jnp.zeros((n, n)), jnp.zeros((n, n))],
            [-B_j,              jnp.zeros((n, n)),      A_j,               jnp.zeros((n, n))],
            [jnp.zeros((n, m)), B_j @ Hq_arc @ K_arc,   jnp.zeros((n, n)), A_j],
        ])
        Saug_minus = A_aug @ Saug_j
        S_tilde = Saug_minus[m + n:m + 2 * n, :]

        # Underweighted Kalman gain via two triangular solves - avoids ever
        # forming Ptild, P_v, or the innovations covariance W explicitly:
        # trace(A) = sum(A**2) for a sqrt factor A of a covariance (same
        # identity already used for HS_tilde), so S_v (now supplied
        # directly by MeasurementModel, no Cholesky needed here) is all
        # that's required. QR the wide innovations sqrt into a compact
        # lower-triangular S_innov, then solve_triangular twice rather
        # than inverting it.
        p_test = 5 / 6
        HS_tilde = H_j1 @ S_tilde
        val_under = jnp.sum(HS_tilde ** 2) - p_test / (1 - p_test) * jnp.sum(S_v_j1 ** 2)
        p_apply = smooth_val2val(val_under, a=100, val1=p_test, val2=1.0)

        S_innov_wide = jnp.block([[HS_tilde / jnp.sqrt(p_apply), S_v_j1]])
        S_innov = jnp.linalg.qr(S_innov_wide.T, mode='r').T
        X = jax.scipy.linalg.solve_triangular(S_innov, HS_tilde, lower=True)
        L_j1 = jax.scipy.linalg.solve_triangular(S_innov.T, X @ S_tilde.T).T

        D_aug = jnp.block([
            [jnp.eye(m),        jnp.zeros((m, n)), jnp.zeros((m, n)),        jnp.zeros((m, n))],
            [jnp.zeros((n, m)), jnp.eye(n),        jnp.zeros((n, n)),        jnp.zeros((n, n))],
            [jnp.zeros((n, m)), jnp.zeros((n, n)), jnp.eye(n) - L_j1 @ H_j1, jnp.zeros((n, n))],
            [jnp.zeros((n, m)), jnp.zeros((n, n)), -L_j1 @ H_j1,             jnp.eye(n)],
        ])
        n_meas = L_j1.shape[1]
        F_col = jnp.block([[jnp.zeros((m, n_meas))], [jnp.zeros((n, n_meas))], [L_j1], [L_j1]])

        Saug_wide = jnp.block([[D_aug @ Saug_minus, F_col @ S_v_j1]])
        return jnp.linalg.qr(Saug_wide.T, mode='r').T

    @staticmethod
    def _augment_arc_initial(Shat0_arc: Array, S21_arc: Array, S22_arc: Array, S_w_arc: Array) -> Array:
        """Build an arc's initial augmented sqrt, literally from the
        derivation's arc-initialization rules: S11=S_w_arc (item 1),
        S21=S31=S41=0 (items 2/3), S22=S42=Shat0_arc (item 4 - xhat_0's
        own marginal, also xhat_j's since xhat_0==xhat_j at j=0),
        S32/S33=S21_arc/S22_arc (xtilde's carried-over [S21',S22'] pair
        from the previous arc's boundary QR - see propagate_cov), and
        S43=S44=0 (item 6 - xhat_j at j=0 has no generator of its own).

        S_w_arc is square (m, m) (QR-compressed in propagate_cov, same as
        TrueStateSqrtCovPropagator), so this comes out exactly (m+3n,
        m+3n) - the item-6 zero block is a literal (n, n) block here, not
        a leftover pad to reach some other target shape.
        """
        n = Shat0_arc.shape[0]
        m = S_w_arc.shape[0]

        zeros_col1 = jnp.zeros((n, m))
        zeros_owncol = jnp.zeros((n, n))

        return jnp.block([
            [S_w_arc,    jnp.zeros((m, n)), jnp.zeros((m, n)), jnp.zeros((m, n))],  # item 1 (S11)
            [zeros_col1, Shat0_arc,         zeros_owncol,      zeros_owncol],        # xhat_0
            [zeros_col1, S21_arc,           S22_arc,           zeros_owncol],        # xtilde_0
            [zeros_col1, Shat0_arc,         zeros_owncol,      zeros_owncol],        # xhat_0 == xhat_j at j=0 (item 6)
        ])

    def init_error_state(self, problem_def: ProblemDefinition) -> tuple[Array, Array, Array]:
        """(Shat_0, S21_0, S22_0): Phat_0's sqrt, and Ptild_0's [S21,S22]
        pair (S21_0=0 since Phat_0/Ptild_0 start uncorrelated)."""
        unc = problem_def.uncertainty
        Shat_0 = jnp.linalg.cholesky(unc.Phat_0, upper=False)
        n = Shat_0.shape[0]
        S21_0 = jnp.zeros((n, n))
        S22_0 = jnp.linalg.cholesky(unc.Ptild_0, upper=False)
        return Shat_0, S21_0, S22_0

    def propagate_cov(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        unc = problem_def.uncertainty
        n = dims.state_dim
        m = dims.control_dim

        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, m)
        _, S_exe_arc_hst = gates2Gexe_vmap(U_arc_hst, unc.gates)
        K_arc_hst = sol_data['K_arc_hst']
        A_hst = sol_data['A_hst']
        B_hst = sol_data['B_hst']
        X_hst = sol_data['X_hst']  # (N_arcs, N_subarcs+1, n)

        H_hst = jax.vmap(self._meas.H_vmap)(X_hst)
        S_v_hst = jax.vmap(self._meas.S_v_vmap)(X_hst)

        S_stoch = jnp.linalg.cholesky(unc.G_stoch, upper=False)

        n_z = m + 3 * n
        Saug_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, n_z, n_z))

        def arc_body(i, carry):
            Shat0_arc, S21_arc, S22_arc, Saug_hst = carry

            K_arc = K_arc_hst[i]
            # QR-compressed straight from the two sqrt pieces (never
            # forming/Cholesky-ing G_stoch+P_exe) down to a compact
            # square (m, m) factor - same as TrueStateSqrtCovPropagator.
            S_w_wide = jnp.block([[S_stoch, S_exe_arc_hst[i]]])
            S_w_arc = jnp.linalg.qr(S_w_wide.T, mode='r').T
            Saug0_arc = self._augment_arc_initial(Shat0_arc, S21_arc, S22_arc, S_w_arc)
            Saug_hst = Saug_hst.at[i, 0, :, :].set(Saug0_arc)

            Hq_arc = sol_data['Hq_arc_hst'][i]

            def subarc_body(j, Saug_js):
                Saug_j1 = self.subarc_sqrt_step(
                    A_hst[i, j], B_hst[i, j], Hq_arc, K_arc, Saug_js[j, :, :],
                    H_hst[i, j + 1], S_v_hst[i, j + 1],
                )
                Saug_js = Saug_js.at[j + 1, :, :].set(Saug_j1)
                return Saug_js

            Saug_js = jax.lax.fori_loop(0, dims.N_subarcs, subarc_body, Saug_hst[i, :, :, :])
            Saug_hst = Saug_hst.at[i, :, :, :].set(Saug_js)

            # Arc-boundary reinit: QR-compress the terminal [xhat; xtilde]
            # row-stack (the derivation's arc-to-arc rearrangement), giving
            # the next arc's (Shat0, S21, S22) triple.
            S_hat_row = Saug_js[-1, m + 2 * n:, :]
            S_tilde_row = Saug_js[-1, m + n:m + 2 * n, :]
            S_hat_tilde_wide = jnp.block([[S_hat_row], [S_tilde_row]])
            S_new = jnp.linalg.qr(S_hat_tilde_wide.T, mode='r').T
            Shat_next, S21_next, S22_next = S_new[:n, :n], S_new[n:, :n], S_new[n:, n:]

            return Shat_next, S21_next, S22_next, Saug_hst

        Shat0_init, S21_init, S22_init = self.init_error_state(problem_def)
        Shat_final, S21_final, S22_final, Saug_hst = jax.lax.fori_loop(
            0, dims.N_arcs, arc_body, (Shat0_init, S21_init, S22_init, Saug_hst),
        )

        sol_data['Saug_hst'] = Saug_hst
        # True-state deviation sqrt at the transfer's end: [S11'-S21', S22']
        # (see terminal_constraint_value) - stashed here since it falls
        # straight out of the last arc's own boundary QR, same one every
        # other arc already computes for its own reinitialization.
        sol_data['S_Xf_final'] = jnp.block([[Shat_final - S21_final, S22_final]])
        # Per-arc sqrt factor of the feedback command's covariance, from the
        # xhat_0 row (feedback acts on the estimate):
        # S_control @ S_control.T == K_arc @ Phat0_arc @ K_arc.T.
        sol_data['S_control_arc_hst'] = K_arc_hst @ Saug_hst[:, 0, m:m + n, :]
        return sol_data

    def terminal_constraint_value(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        unc = problem_def.uncertainty

        A_postinsert = sol_data['A_postinsert']
        S_Xf_full = sol_data['S_Xf_final']

        S_Xf_targ_inv = unc.S_XT_targ_inv @ A_postinsert
        M = S_Xf_targ_inv @ S_Xf_full

        sigma_max = jnp.linalg.norm(M, ord=2)
        return jnp.log(sigma_max)

