"""Generic numeric helpers used across the rework.

Ported verbatim from the legacy Lib/math.py (the subset the OOP framework
actually uses), so src/ no longer depends on Lib. Pure functions only - no
problem/state coupling.
"""

import jax
import jax.numpy as jnp
import numpy as np
from astropy.time import Time


# --------------------------------------------------------------------------- #
# Epoch / time
# --------------------------------------------------------------------------- #

def calc_t_elapsed_nd(t0, tf, nodes, t_star):
    """Nondimensional elapsed-time nodes between two ISO epochs, scaled by t_star."""
    delta_t = Time(tf) - Time(t0)
    return jnp.linspace(0.0, delta_t.sec, nodes) / t_star


# --------------------------------------------------------------------------- #
# Covariance / matrix
# --------------------------------------------------------------------------- #

def sig2cov(r_1sig, v_1sig, m_1sig, Sys, m0):
    """Diagonal (nondimensional) 7x7 state covariance from 1-sigma pos/vel/mass."""
    r_cov = (r_1sig / Sys['Ls']) ** 2
    v_cov = (v_1sig / Sys['Vs']) ** 2
    m_cov = (m_1sig / m0) ** 2
    return np.diag(np.array([r_cov, r_cov, r_cov, v_cov, v_cov, v_cov, m_cov]))


def mat_lmax(A):
    """Largest eigenvalue of a symmetric matrix (tiny jitter for conditioning)."""
    eps = 1e-12
    return jnp.linalg.eigvalsh(A + eps * jnp.diag(jnp.linspace(1., 2., A.shape[0])))[-1]


mat_lmax_vmap = jax.vmap(mat_lmax, in_axes=(0,))


# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #

def smooth_val2val(x, a, val1, val2):
    """Smoothly transition from val1 (x<<0) to val2 (x>>0) with sharpness a."""
    return val1 + (val2 - val1) * (-1 / 2 * jax.nn.tanh(a * x) + 1 / 2)


smooth_val2val_vmap = jax.vmap(smooth_val2val, in_axes=(0, None, None, None))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def cart2sph(r_vec):
    """Cartesian -> (magnitude, azimuth[deg], elevation[deg])."""
    x, y, z = r_vec[0], r_vec[1], r_vec[2]
    r = jnp.linalg.norm(r_vec)
    th = jnp.arctan2(y, x) * 180 / jnp.pi
    rho_xy = jnp.hypot(x, y)
    phi_raw = jnp.arctan2(z, rho_xy) * 180 / jnp.pi
    phi = jnp.where(r > 0, phi_raw, 0.0)
    return jnp.array([r, th, phi])


cart2sph_vmap = jax.vmap(cart2sph, in_axes=(0,))


# --------------------------------------------------------------------------- #
# Collision avoidance (deterministic) + adaptive mesh
# --------------------------------------------------------------------------- #

def col_avoid(X, dyn_args):
    """Keep-out margin: d_safe - |r - r_obj| (>0 means inside the keep-out sphere)."""
    r_obj = dyn_args['r_obj']
    safe_d = dyn_args['d_safe']
    delta_X = X[:3] - r_obj
    dist = jnp.sqrt(delta_X[0] ** 2 + delta_X[1] ** 2 + delta_X[2] ** 2)
    return safe_d - dist


col_avoid_vmap = jax.vmap(col_avoid, in_axes=(0, None))


def adaptive_mesh_con_terms(t_node_bound, dt_min=1e-5, dt_max=1e6):
    """Linear (jac/lower/upper) terms enforcing monotone node times + fixed tf."""
    n = t_node_bound.shape[0]
    tf = t_node_bound[-1]

    jac = jnp.zeros((n + 1, n))
    jac = jac.at[0, 0].set(1.0)
    for i in range(n - 1):
        jac = jac.at[i + 1, i].set(-1.0)
        jac = jac.at[i + 1, i + 1].set(1.0)
    jac = jac.at[-1, -1].set(1.0)

    lower = jnp.zeros(n + 1).at[0].set(0.0).at[1:n].set(dt_min).at[-1].set(tf)
    upper = jnp.zeros(n + 1).at[0].set(0.0).at[1:n].set(dt_max).at[-1].set(tf)

    return {'jac': jac, 'lower': lower, 'upper': upper}

# --------------------------------------------------------------------------- #
# Control Functions (Regularization)
# --------------------------------------------------------------------------- #

def U_reg_to_U(U_reg):
    """Convert regularized control to cartesian through non-linear transform"""
    u, w, s = U_reg[0], U_reg[1], U_reg[2]

    U_x = u**2 - w**2 - s**2
    U_y = 2 * u * w
    U_z = 2 * u * s
    U_cart = jnp.array([U_x, U_y, U_z])
    return U_cart

U_reg_to_U_vmap = jax.vmap(U_reg_to_U, in_axes=(0,))