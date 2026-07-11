"""MeasurementModel - the EKF measurement functions for estimated-state feedback.

Owns the SymPy construction of the range / range-rate / angles measurement
functions, their state Jacobians, and their (state-dependent) noise covariances
(internalized from the legacy Lib.dynamics.measurement_model_builder, so src/ no
longer depends on Lib), and lambdifies + vmaps them for jax.

MeasurementModel is a private implementation detail of
EstimatedStateCovPropagator (and the MC runner) - built from problem_def alone
via the .build() factory, so nothing upstream ever has to optionally thread a
measurement model through the pipeline. true_state / true_state_sqrt propagators
simply never construct one.
"""

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from src.problem.problem_definition import ProblemDefinition


def _build_measurement_functions(measurements, params):
    """Symbolic -> jax measurement model. Returns single-point and vmapped
    evaluators for the measurement value h, its Jacobian H, and its noise
    covariance P_v, plus n_meas. measurements is the active subset of
    {position, range, range-rate, angles}; params carries r_obs + sigmas."""
    r_x, r_y, r_z, v_x, v_y, v_z, m = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z, m', real=True)
    X_sym = sp.Matrix([[r_x], [r_y], [r_z], [v_x], [v_y], [v_z], [m]])

    obs_x, obs_y, obs_z = params['r_obs']
    rel_pos = sp.Matrix([[r_x - obs_x], [r_y - obs_y], [r_z - obs_z]])
    rel_vel = sp.Matrix([[v_x], [v_y], [v_z]])

    pos = sp.Matrix([[r_x], [r_y], [r_z]])
    pos_cov = sp.diag(params['pos_sig'] ** 2, params['pos_sig'] ** 2, params['pos_sig'] ** 2)

    rng = sp.sqrt((rel_pos.T @ rel_pos)[0])
    rng_cov = sp.diag(params['range_sig'] ** 2)

    rng_rate = (rel_pos.T @ rel_vel)[0] / rng
    rng_rate_cov = sp.diag(params['rate_sig'] ** 2)

    theta = sp.atan2(rel_pos[1], rel_pos[0])
    phi = sp.asin(rel_pos[2] / rng)
    angles = sp.Matrix([[theta], [phi]])
    angles_cov = sp.diag(params['angles_sig'] ** 2, params['angles_sig'] ** 2)

    meas_parts, cov_blocks = [], []
    for meas in measurements:
        key = meas.lower()
        if key in ("position", "pos"):
            meas_parts.append(pos); cov_blocks.append(pos_cov)
        elif key in ("range", "rng"):
            meas_parts.append(sp.Matrix([[rng]])); cov_blocks.append(rng_cov)
        elif key in ("range-rate", "rng-rate", "rate"):
            meas_parts.append(sp.Matrix([[rng_rate]])); cov_blocks.append(rng_rate_cov)
        elif key in ("angles", "ang"):
            meas_parts.append(angles); cov_blocks.append(angles_cov)
        else:
            raise ValueError(f"Measurement type '{meas}' not recognized.")

    meas_vec = sp.Matrix.vstack(*meas_parts)
    meas_cov = sp.diag(*cov_blocks)
    n_state = int(X_sym.shape[0])
    n_meas = int(meas_vec.shape[0])
    meas_jac = meas_vec.jacobian(X_sym)

    h_raw = sp.lambdify((X_sym,), meas_vec, 'jax')
    H_raw = sp.lambdify((X_sym,), meas_jac, 'jax')
    P_v_raw = sp.lambdify((X_sym,), meas_cov, 'jax')

    h_eval = lambda x: jnp.reshape(h_raw(x), (n_meas,))
    H_eval = lambda x: jnp.reshape(H_raw(x), (n_meas, n_state))
    P_v_eval = lambda x: jnp.reshape(P_v_raw(x), (n_meas, n_meas))

    return {
        'h_eval': h_eval, 'P_v_eval': P_v_eval, 'n_meas': n_meas,
        'h_vmap': jax.vmap(h_eval, in_axes=0),
        'H_vmap': jax.vmap(H_eval, in_axes=0),
        'P_v_vmap': jax.vmap(P_v_eval, in_axes=0),
    }


class MeasurementModel:
    """Vectorized EKF measurement functions.

    h_vmap:    (batch, 7) -> (batch, n_meas)              measurement values
    H_vmap:    (batch, 7) -> (batch, n_meas, 7)           measurement Jacobians
    P_v_vmap:  (batch, 7) -> (batch, n_meas, n_meas)      noise covariances

    P_v_vmap is kept state-dependent: while many measurement noise terms are
    constant, some carry a state-varying component, so the covariance is
    evaluated at each node's state rather than stored as a single constant.
    """

    def __init__(self, h_eval: callable, P_v_eval: callable,
                 h_vmap: callable, H_vmap: callable, P_v_vmap: callable, n_meas: int):
        # Single-point evaluators (h_eval/P_v_eval) are used by the MC-trial
        # measurement draw; the vmapped forms drive the batched EKF loops.
        self.h_eval = h_eval
        self.P_v_eval = P_v_eval
        self.h_vmap = h_vmap
        self.H_vmap = H_vmap
        self.P_v_vmap = P_v_vmap
        self.n_meas = n_meas

    @classmethod
    def build(cls, problem_def: ProblemDefinition) -> "MeasurementModel":
        """Construct from problem_def's measurement parameters and active
        measurement set. r_obs and the (already-normalized) sigmas come from
        problem_def.measurement; the selected measurement types from
        problem_def.toggles.measurements."""
        meas = problem_def.measurement
        meas_params = {
            'r_obs': meas.r_obs,
            'pos_sig': meas.pos_sig,
            'range_sig': meas.range_sig,
            'rate_sig': meas.rate_sig,
            'angles_sig': meas.angles_sig,
        }
        model = _build_measurement_functions(problem_def.toggles.measurements, meas_params)
        return cls(
            h_eval=model['h_eval'],
            P_v_eval=model['P_v_eval'],
            h_vmap=model['h_vmap'],
            H_vmap=model['H_vmap'],
            P_v_vmap=model['P_v_vmap'],
            n_meas=model['n_meas'],
        )
