"""Control execution-error model + Monte Carlo control helpers.

Ported from Lib/dynamics.py. gates2Gexe (the thruster gate execution-error
covariance + its wide square-root factor) is shared by the error propagators and
the MC runners; MC_U_tcm_k / MC_U_exe are the per-arc feedback + sampled
execution-noise draws used by the MC trials.
"""

import jax
import jax.numpy as jnp


def gates2Gexe(U, gates):
    """Control execution-error covariance and its (3,6) square-root factor.

    Returns (P_exe, S_exe) with P_exe == S_exe @ S_exe.T. gates =
    [fixed_mag, prop_mag, fixed_point, prop_point]. The covariance is built in a
    spacecraft frame aligned with the thrust direction, then rotated into the
    rotating frame. S_exe is assembled directly from the gate magnitudes (each
    axis variance is a sum of two squares) without any sqrt/cholesky call.
    """
    eps = 1e-12
    norm_U = jnp.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2 + eps)
    cov_1 = gates[2] ** 2 + (gates[3] * norm_U) ** 2
    cov_3 = gates[0] ** 2 + (gates[1] * norm_U) ** 2
    P_exe_diag = jnp.diag(jnp.array([cov_1, cov_1, cov_3]))

    S_exe_fixed = jnp.diag(jnp.array([gates[2], gates[2], gates[0]]))
    S_exe_prop = jnp.diag(jnp.array([gates[3] * norm_U, gates[3] * norm_U, gates[1] * norm_U]))
    S_exe_diag = jnp.hstack([S_exe_fixed, S_exe_prop])  # (3,6)

    # Thrust-aligned spacecraft frame -> rotating frame.
    Z_hat = U.flatten() / norm_U
    E_vec = jnp.cross(jnp.array([0., 0., 1.]), Z_hat.flatten())
    E_hat = E_vec / jnp.sqrt(E_vec[0] ** 2 + E_vec[1] ** 2 + E_vec[2] ** 2 + eps)
    S_vec = jnp.cross(E_hat, Z_hat)
    S_hat = S_vec / jnp.sqrt(S_vec[0] ** 2 + S_vec[1] ** 2 + S_vec[2] ** 2 + eps)
    rot_mat = jnp.column_stack([S_hat, E_hat, Z_hat])

    P_exe = rot_mat @ P_exe_diag @ rot_mat.T
    S_exe = rot_mat @ S_exe_diag  # (3,6)
    return P_exe, S_exe


gates2Gexe_vmap = jax.vmap(gates2Gexe, in_axes=(0, None))


def MC_U_tcm_k(X_k_nom, X_k_trial, K_k):
    """Feedback trajectory-correction control for one arc: K @ (X_trial - X_nom)."""
    return K_k @ (X_k_trial - X_k_nom)


def MC_U_exe(U_nom, gates, rng_key):
    """Sample a control execution-error draw for one arc from gates2Gexe(U_nom)."""
    P_exe, _ = gates2Gexe(U_nom, gates)
    return jax.random.multivariate_normal(rng_key, jnp.zeros(3), P_exe)
