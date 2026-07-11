"""MonteCarloRunner hierarchy - sampled trial simulation for the stochastic
problem (the analysis-time counterpart to the optimizer-time ErrorPropagator).

Design (see also the discussion in the plan): the base class owns everything
feedback-agnostic - the batch loop, per-arc noise draws, control clamping, and
dV accounting - and defers the per-trial rollout to build_single_trial(), which
each feedback-type subclass implements. build_single_trial() closes over the
mean-trajectory data + static config and returns a PURE function
single_trial(rng_key) -> trial_dict; run() then builds that "executable" once,
jit+vmaps it, and calls it over chunks of keys. This keeps the
true_state/estimated_state split in exactly one place and lets run() stay
completely output-structure-agnostic (it just tree-stacks whatever dict
single_trial returns).

The nonlinear arc integration and A/B sensitivity primitives are reused in
place from the Propagator instance StochasticTOP already built
(propagator._propagate_arc, propagator._arc_dX0_vmap, ...); the estimated-state
per-trial EKF reuses ErrorPropagator.subarc_ekf_step + its MeasurementModel.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm

from .control_noise import MC_U_exe, MC_U_tcm_k, gates2Gexe
from .error_propagator import ErrorPropagator
from src.dynamics.propagator import Propagator
from src.problem.problem_definition import ProblemDefinition
from src.utils.math_utils import cart2sph_vmap


class MonteCarloRunner:
    """+build_single_trial(sol_data) : callable(rng_key) -> dict
    +run(sol_data, seed, n_trials) : dict

    Subclasses implement build_single_trial(); the base owns run() and the
    shared feedback-agnostic per-trial helpers.
    """

    # Trials per vmap batch - caps peak memory, mirrors legacy sim_MC_trajs's
    # MC_N_Loop. run() jit-compiles the batched trial once; a final partial
    # chunk (when N_trials isn't a multiple of this) costs one extra trace.
    _MC_N_LOOP = 5

    def __init__(self, problem_def: ProblemDefinition, propagator: Propagator,
                 error_propagator: ErrorPropagator):
        self._pd = problem_def
        self._prop = propagator              # nonlinear integrator + A/B sensitivities
        self._ep = error_propagator          # reuse subarc_ekf_step + MeasurementModel

    # --- to be implemented by feedback-type subclasses ---------------------

    def build_single_trial(self, sol_data: dict) -> callable:
        """Return a pure fn single_trial(rng_key) -> trial_dict, closed over
        the mean-trajectory data in sol_data and the static config."""
        raise NotImplementedError

    def _postprocess(self, stacked: dict) -> dict:
        """Hook for derived batch statistics (e.g. mean covariances). Base is
        a no-op; run() returns whatever this yields."""
        return stacked

    # --- shared, feedback-agnostic per-trial helpers -----------------------

    def _draw_arc_noise(self, key_exe: Array, key_w: Array, U_arc_det: Array) -> tuple[Array, Array]:
        """One arc's control-execution noise draw (rotated per the gate model)
        and process-noise draw. Both feedback types draw these identically."""
        gates = self._pd.uncertainty.gates
        G_stoch = self._pd.uncertainty.G_stoch

        U_exe = MC_U_exe(U_arc_det, gates, key_exe)
        # Skip the process-noise draw entirely when G_stoch is (numerically)
        # zero - multivariate_normal on a zero covariance is wasted work.
        G_stoch_zero = jnp.all(jnp.isclose(G_stoch, 0.0, atol=1e-20))
        U_w = jax.lax.cond(
            G_stoch_zero,
            lambda k: jnp.zeros(3,),
            lambda k: jax.random.multivariate_normal(k, jnp.zeros(3,), G_stoch),
            key_w,
        )
        return U_exe, U_w

    @staticmethod
    def _clamp_control(U_cmd: Array) -> Array:
        """Cap the commanded control to unit norm (thrust saturation)."""
        norm = jnp.linalg.norm(U_cmd)
        return jax.lax.cond(norm > 1.0, lambda: U_cmd / norm, lambda: U_cmd)

    def _compute_dV(self, U_hst: Array, X_hst: Array, t_hst: Array, dV_mean: float) -> tuple[Array, Array]:
        """Total dimensional dV over a trial and its excess over the nominal
        (dV_tcm). Same accounting as legacy single_MC_trial."""
        dt = t_hst[1] - t_hst[0]
        U_Acc_min_nd = self._pd.spacecraft.U_Acc_min_nd
        Vs = self._pd.Sys['Vs']
        dV_trial = jnp.sum(jnp.linalg.norm(U_hst, axis=1) * U_Acc_min_nd * dt / X_hst[:, -1]) * Vs
        return dV_trial, dV_trial - dV_mean

    # --- batch driver ------------------------------------------------------

    def run(self, sol_data: dict, seed: int = 42, n_trials: int = None) -> dict:
        """Build the single-trial executable once, then vmap it over chunks of
        rng keys and tree-stack the per-trial dicts into batched (N, ...)
        arrays. Output-structure-agnostic: works for any dict single_trial
        returns."""
        N = self._pd.dims.N_trials if n_trials is None else n_trials
        single_trial = self.build_single_trial(sol_data)
        batched = jax.jit(jax.vmap(single_trial), backend='cpu')

        print(f"Running {N} MC Trials...")
        keys = jax.random.split(jax.random.PRNGKey(seed), N)
        chunks = [batched(keys[i:i + self._MC_N_LOOP])
                  for i in tqdm(range(0, N, self._MC_N_LOOP))]
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *chunks)

        return self._postprocess(stacked)


class TrueStateMCRunner(MonteCarloRunner):
    """True-state feedback MC trials.

    Each trial draws a dispersed initial state, then rolls the arcs forward
    through the true nonlinear dynamics under feedback on the TRUE state
    (U_tcm = K @ (X_true - X_nom)) plus sampled execution/process noise,
    followed by the zero-control post-insertion coast. Produces the same
    detailed-resolution histories legacy single_MC_trial does, so the batched
    output drops straight into the legacy save_sol MC schema.
    """

    def build_single_trial(self, sol_data: dict) -> callable:
        dims = self._pd.dims
        n = dims.state_dim
        m = dims.control_dim
        N_arcs = dims.N_arcs
        arc_len = dims.arc_length_det
        transfer_len = dims.transfer_length_det
        post_len = dims.post_insert_length
        length = dims.length
        Phat_0 = self._pd.uncertainty.Phat_0
        tf_T = self._pd.boundary_conditions.tf_T

        # Nominal (mean-trajectory) data the feedback references, captured as
        # constants of the returned pure closure.
        det_X_node = sol_data['X_node_hst']                     # (N_arcs+1, n)
        det_U_arc = sol_data['U_arc_hst'].reshape(N_arcs, m)    # (N_arcs, m)
        K_arc_hst = sol_data['K_arc_hst']                       # (N_arcs, m, n)
        t_node = sol_data['t_node_bound']                       # (N_arcs+1,)
        dV_mean = sol_data['dV_mean']

        propagate_arc = self._prop._propagate_arc               # nonlinear arc integrator

        def single_trial(rng_key: Array) -> dict:
            keys = jax.random.split(rng_key, 1 + 2 * N_arcs)
            key_X0 = keys[0]
            keys_exe = keys[1:1 + N_arcs]
            keys_w = keys[1 + N_arcs:1 + 2 * N_arcs]

            X0_trial = jax.random.multivariate_normal(key_X0, det_X_node[0], Phat_0)

            X_hst = jnp.zeros((length, n)).at[0].set(X0_trial)
            U_hst = jnp.zeros((length, m))
            t_hst = jnp.zeros((length,)).at[0].set(t_node[0])

            # Detailed indices are static (arc_len fixed), so a plain Python
            # loop unrolls cleanly under jit/vmap with static array slices.
            X0_arc = X0_trial
            for k in range(N_arcs):
                i0 = k * (arc_len - 1)
                i_f = (k + 1) * (arc_len - 1)

                U_tcm = MC_U_tcm_k(det_X_node[k], X0_arc, K_arc_hst[k])
                U_exe, U_w = self._draw_arc_noise(keys_exe[k], keys_w[k], det_U_arc[k])
                U_cmd = self._clamp_control(det_U_arc[k] + U_tcm)
                U_tot = U_cmd + U_exe + U_w

                X_arc = propagate_arc(X0_arc, U_tot, t_node[k], t_node[k + 1], arc_len)
                t_arc = jnp.linspace(t_node[k], t_node[k + 1], arc_len)

                X_hst = X_hst.at[i0 + 1:i_f + 1].set(X_arc[1:])
                t_hst = t_hst.at[i0 + 1:i_f + 1].set(t_arc[1:])
                U_hst = U_hst.at[i0:i_f].set(jnp.tile(U_cmd, (arc_len - 1, 1)))
                X0_arc = X_arc[-1]

            # Post-insertion coast (zero control, fixed duration tf_T).
            pi0 = transfer_len - 1
            X_pi = propagate_arc(X_hst[pi0], jnp.zeros(m), t_hst[pi0], t_hst[pi0] + tf_T, post_len)
            t_pi = jnp.linspace(t_hst[pi0], t_hst[pi0] + tf_T, post_len)
            X_hst = X_hst.at[pi0 + 1:].set(X_pi[1:])
            t_hst = t_hst.at[pi0 + 1:].set(t_pi[1:])

            U_hst_sph = cart2sph_vmap(U_hst)
            dV_trial, dV_tcm = self._compute_dV(U_hst, X_hst, t_hst, dV_mean)

            return {'X_hst': X_hst, 'U_hst': U_hst, 'U_hst_sph': U_hst_sph,
                    't_hst': t_hst, 'dV': dV_trial, 'dV_tcm': dV_tcm}

        return single_trial

    def _postprocess(self, stacked: dict) -> dict:
        # Map the batched per-trial arrays onto the legacy MC_Runs keys that
        # save_sol writes, converting to numpy for h5py.
        return {
            'X_hsts': np.asarray(stacked['X_hst']),
            't_hsts': np.asarray(stacked['t_hst']),
            'U_hsts': np.asarray(stacked['U_hst']),
            'U_hsts_sph': np.asarray(stacked['U_hst_sph']),
            'dVs': np.asarray(stacked['dV']),
            'dV_tcms': np.asarray(stacked['dV_tcm']),
        }


class EstimatedStateMCRunner(MonteCarloRunner):
    """Estimated-state feedback MC trials with a per-trial EKF.

    Like the true-state runner, but feedback acts on the trial's EKF-estimated
    state (U_tcm = K @ (Xhat - X_nom)) rather than the true state, and each
    trial carries its own extended Kalman filter: the estimate Xhat is
    propagated across each subarc, the augmented covariance is propagated along
    that (per-trial) estimated trajectory - re-linearized via the propagator's
    A/B primitives and the MeasurementModel - and at each subarc end a noisy
    measurement of the true state updates Xhat via the freshly computed Kalman
    gain. The mean-trajectory covariances still come from the optimizer; this
    per-trial EKF is layered on top for MC-trial realism. Reuses
    EstimatedStateCovPropagator.subarc_ekf_step in place for the covariance step.

    Requires N_save >= 2 (each subarc spans N_save detailed points).
    """

    def build_single_trial(self, sol_data: dict) -> callable:
        dims = self._pd.dims
        unc = self._pd.uncertainty
        n = dims.state_dim
        m = dims.control_dim
        N_arcs = dims.N_arcs
        N_subarcs = dims.N_subarcs
        N_save = dims.N_save
        arc_len = dims.arc_length_det
        transfer_len = dims.transfer_length_det
        post_len = dims.post_insert_length
        length = dims.length
        Phat_0, Ptild_0 = unc.Phat_0, unc.Ptild_0
        G_stoch, gates = unc.G_stoch, unc.gates
        tf_T = self._pd.boundary_conditions.tf_T

        det_X_node = sol_data['X_node_hst']
        det_U_arc = sol_data['U_arc_hst'].reshape(N_arcs, m)
        K_arc_hst = sol_data['K_arc_hst']
        t_node = sol_data['t_node_bound']
        dV_mean = sol_data['dV_mean']

        propagate_arc = self._prop._propagate_arc
        A_vmap = self._prop._arc_dX0_vmap
        B_vmap = self._prop._arc_dU_vmap
        meas = self._ep._meas
        ekf_step = self._ep.subarc_ekf_step
        Paug0_init = self._ep.init_error_state(self._pd)     # (2n, 2n)

        def single_trial(rng_key: Array) -> dict:
            # Key layout mirrors legacy single_MC_trial (estimated_state):
            # X-hat / X-tilde draws, per-arc exe/process noise, per-subarc
            # measurement noise (with the same trailing +1 spare key).
            n_meas_keys = N_arcs * N_subarcs + 1
            keys = jax.random.split(rng_key, 2 + 2 * N_arcs + n_meas_keys)
            key_Xhat0, key_Xtild0 = keys[0], keys[1]
            keys_exe = keys[2:2 + N_arcs]
            keys_w = keys[2 + N_arcs:2 + 2 * N_arcs]
            keys_meas = keys[2 + 2 * N_arcs:]

            Xhat0 = jax.random.multivariate_normal(key_Xhat0, det_X_node[0], Phat_0)
            Xtild0 = jax.random.multivariate_normal(key_Xtild0, jnp.zeros(n), Ptild_0)
            X0_trial = Xhat0 + Xtild0

            X_hst = jnp.zeros((length, n)).at[0].set(X0_trial)
            Xhat_hst = jnp.zeros((length, n)).at[0].set(Xhat0)
            Paug_hst = jnp.zeros((length, 2 * n, 2 * n)).at[0].set(Paug0_init)
            U_hst = jnp.zeros((length, m))
            t_hst = jnp.zeros((length,)).at[0].set(t_node[0])

            X0_arc = X0_trial
            for k in range(N_arcs):
                arc_i0 = k * (arc_len - 1)
                arc_if = (k + 1) * (arc_len - 1)

                # Feedback on the arc-initial ESTIMATE.
                Xhat0_arc = Xhat_hst[arc_i0]
                Paug0_arc = Paug_hst[arc_i0]
                U_tcm = MC_U_tcm_k(det_X_node[k], Xhat0_arc, K_arc_hst[k])
                U_exe, U_w = self._draw_arc_noise(keys_exe[k], keys_w[k], det_U_arc[k])
                U_cmd = self._clamp_control(det_U_arc[k] + U_tcm)
                U_tot = U_cmd + U_exe + U_w
                P_exe_arc, _ = gates2Gexe(det_U_arc[k], gates)
                P_u_arc = G_stoch + P_exe_arc

                # True state across the arc (detailed resolution).
                X_arc = propagate_arc(X0_arc, U_tot, t_node[k], t_node[k + 1], arc_len)
                t_arc = jnp.linspace(t_node[k], t_node[k + 1], arc_len)
                X_hst = X_hst.at[arc_i0 + 1:arc_if + 1].set(X_arc[1:])
                t_hst = t_hst.at[arc_i0 + 1:arc_if + 1].set(t_arc[1:])
                U_hst = U_hst.at[arc_i0:arc_if].set(jnp.tile(U_cmd, (arc_len - 1, 1)))

                # tau/gam cross-correlation carriers reset per arc.
                tau = Paug0_arc
                gam = jnp.zeros((2 * n, m))
                for j in range(N_subarcs):
                    sub_i0 = arc_i0 + j * (N_save - 1)
                    sub_if = arc_i0 + (j + 1) * (N_save - 1)

                    # Propagate the estimate across the subarc (commanded
                    # control only - no noise), re-linearize A/B/H/P_v along it.
                    Xhat_sub = propagate_arc(Xhat_hst[sub_i0], U_cmd, t_hst[sub_i0], t_hst[sub_if], N_save)
                    t_sub = t_hst[sub_i0:sub_if + 1]
                    A_sub = A_vmap(Xhat_sub[:-1], U_cmd, t_sub[:-1], t_sub[1:])
                    B_sub = B_vmap(Xhat_sub[:-1], U_cmd, t_sub[:-1], t_sub[1:])
                    H_sub = meas.H_vmap(Xhat_sub)
                    P_v_sub = meas.P_v_vmap(Xhat_sub)
                    Paug_sub0 = Paug_hst[sub_i0]

                    # Covariance across the subarc's detailed micro-steps;
                    # measurement update gated to the last step only.
                    def micro_body(jj, carry):
                        Paug_js, tau, gam, L_end = carry
                        upd = jnp.where(jj == N_save - 2, 1.0, 0.0)
                        Paug_j1, tau, gam, L = ekf_step(
                            A_sub[jj], B_sub[jj], K_arc_hst[k], P_u_arc, Paug_sub0,
                            H_sub[jj + 1], P_v_sub[jj + 1], Paug_js[jj], tau, gam, upd,
                        )
                        Paug_js = Paug_js.at[jj + 1].set(Paug_j1)
                        L_end = jnp.where(jj == N_save - 2, L, L_end)
                        return Paug_js, tau, gam, L_end

                    Paug_sub = jnp.zeros((N_save, 2 * n, 2 * n)).at[0].set(Paug_sub0)
                    L_end0 = jnp.zeros((n, meas.n_meas))
                    Paug_sub, tau, gam, L_j1 = jax.lax.fori_loop(
                        0, N_save - 1, micro_body, (Paug_sub, tau, gam, L_end0))
                    Paug_hst = Paug_hst.at[sub_i0 + 1:sub_if + 1].set(Paug_sub[1:])

                    # EKF measurement update at the subarc end: noisy
                    # measurement of the TRUE state vs. predicted measurement of
                    # the estimate.
                    z = jax.random.multivariate_normal(
                        keys_meas[k * N_subarcs + j], meas.h_eval(X_hst[sub_if]), meas.P_v_eval(X_hst[sub_if]))
                    z_est = meas.h_eval(Xhat_sub[-1])
                    Xhat_end = Xhat_sub[-1] + L_j1 @ (z - z_est)
                    Xhat_sub = Xhat_sub.at[-1].set(Xhat_end)
                    Xhat_hst = Xhat_hst.at[sub_i0 + 1:sub_if + 1].set(Xhat_sub[1:])

                X0_arc = X_arc[-1]

            # Post-insertion coast (zero control): true + estimate + a pure
            # covariance coast (K=0, no noise, no update).
            pi0 = transfer_len - 1
            X_pi = propagate_arc(X_hst[pi0], jnp.zeros(m), t_hst[pi0], t_hst[pi0] + tf_T, post_len)
            t_pi = jnp.linspace(t_hst[pi0], t_hst[pi0] + tf_T, post_len)
            X_hst = X_hst.at[pi0 + 1:].set(X_pi[1:])
            t_hst = t_hst.at[pi0 + 1:].set(t_pi[1:])

            Xhat_pi = propagate_arc(Xhat_hst[pi0], jnp.zeros(m), t_hst[pi0], t_hst[pi0] + tf_T, post_len)
            Xhat_hst = Xhat_hst.at[pi0 + 1:].set(Xhat_pi[1:])
            A_pi = A_vmap(Xhat_pi[:-1], jnp.zeros(m), t_pi[:-1], t_pi[1:])
            B_pi = B_vmap(Xhat_pi[:-1], jnp.zeros(m), t_pi[:-1], t_pi[1:])
            H_pi = meas.H_vmap(Xhat_pi)
            P_v_pi = meas.P_v_vmap(Xhat_pi)
            zeros_K = jnp.zeros((m, n))
            zeros_Pu = jnp.zeros((m, m))
            Paug_pi0 = Paug_hst[pi0]

            def coast_body(jj, carry):
                Paug_js, tau, gam = carry
                Paug_j1, tau, gam, _ = ekf_step(
                    A_pi[jj], B_pi[jj], zeros_K, zeros_Pu, Paug_pi0,
                    H_pi[jj + 1], P_v_pi[jj + 1], Paug_js[jj], tau, gam, 0.0,
                )
                Paug_js = Paug_js.at[jj + 1].set(Paug_j1)
                return Paug_js, tau, gam

            Paug_post = jnp.zeros((post_len, 2 * n, 2 * n)).at[0].set(Paug_pi0)
            Paug_post, _, _ = jax.lax.fori_loop(
                0, post_len - 1, coast_body, (Paug_post, Paug_pi0, jnp.zeros((2 * n, m))))
            Paug_hst = Paug_hst.at[pi0 + 1:].set(Paug_post[1:])

            U_hst_sph = cart2sph_vmap(U_hst)
            dV_trial, dV_tcm = self._compute_dV(U_hst, X_hst, t_hst, dV_mean)

            return {'X_hst': X_hst, 'Xhat_hst': Xhat_hst, 'Paug': Paug_hst,
                    'U_hst': U_hst, 'U_hst_sph': U_hst_sph, 't_hst': t_hst,
                    'dV': dV_trial, 'dV_tcm': dV_tcm}

        return single_trial

    def _postprocess(self, stacked: dict) -> dict:
        n = self._pd.dims.state_dim
        Paug = stacked['Paug']                       # (N, length, 2n, 2n)
        Phat = Paug[:, :, :n, :n]
        Ptild = Paug[:, :, n:, n:]
        Phattild = Paug[:, :, :n, n:]
        P = Phat - Phattild - Phattild.swapaxes(-1, -2) + Ptild

        out = {
            'X_hsts': np.asarray(stacked['X_hst']),
            'Xhat_hsts': np.asarray(stacked['Xhat_hst']),
            'P_hsts': np.asarray(P),
            'Phat_hsts': np.asarray(Phat),
            'Ptild_hsts': np.asarray(Ptild),
            'Phattild_hsts': np.asarray(Phattild),
            't_hsts': np.asarray(stacked['t_hst']),
            'U_hsts': np.asarray(stacked['U_hst']),
            'U_hsts_sph': np.asarray(stacked['U_hst_sph']),
            'dVs': np.asarray(stacked['dV']),
            'dV_tcms': np.asarray(stacked['dV_tcm']),
        }
        out['P_mean_hst'] = out['P_hsts'].mean(axis=0)
        out['Phat_mean_hst'] = out['Phat_hsts'].mean(axis=0)
        out['Ptild_mean_hst'] = out['Ptild_hsts'].mean(axis=0)
        out['Phattild_mean_hst'] = out['Phattild_hsts'].mean(axis=0)
        return out
