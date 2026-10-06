"""Generic numeric helpers used across the rework.

Ported verbatim from the legacy Lib/math.py (the subset the OOP framework
actually uses), so src/ no longer depends on Lib. Pure functions only - no
problem/state coupling.
"""

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.time import Time
from scipy import stats


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


# --------------------------------------------------------------------------- #
# Control Functions (Regularization)
# --------------------------------------------------------------------------- #

def U_reg_to_U_cart(U_reg):
    """Convert regularized control to cartesian through non-linear transform"""
    reg_x, reg_y, reg_z = U_reg[0], U_reg[1], U_reg[2]

    U_x = reg_x**2 - reg_y**2 - reg_z**2
    U_y = 2 * reg_x * reg_y
    U_z = 2 * reg_x * reg_z
    U_cart = jnp.array([U_x, U_y, U_z])
    return U_cart

U_reg_to_U_cart_vmap = jax.vmap(U_reg_to_U_cart, in_axes=(0,))

def U_cart_to_U_reg(U_cart):
    """Convert cartesian control to regularized through non-linear transform.
    Singular where U_cart points purely in -x (or is zero): reg_x -> 0 there and
    reg_y/reg_z are genuinely undefined (a whole circle of solutions maps to the
    same U_cart) - falls back to the zero vector rather than dividing by ~0.
    jnp.where (not Python if/elif) so this is safe under vmap/grad; the safe
    denominator avoids a real 0/0 in the graph, which would otherwise poison
    gradients through the masked-out branch too."""
    U_x, U_y, U_z = U_cart[0], U_cart[1], U_cart[2]
    U_cart_norm = jnp.sqrt(U_x**2 + U_y**2 + U_z**2)

    reg_x = jnp.sqrt(jnp.maximum(U_x + U_cart_norm, 0.0) / 2)
    safe_reg_x = jnp.where(reg_x > 0, reg_x, 1.0)
    reg_y = jnp.where(reg_x > 0, U_y / (2 * safe_reg_x), 0.0)
    reg_z = jnp.where(reg_x > 0, U_z / (2 * safe_reg_x), 0.0)

    U_reg = jnp.array([reg_x, reg_y, reg_z])
    return U_reg

U_cart_to_U_reg_vmap = jax.vmap(U_cart_to_U_reg, in_axes=(0,))


# --------------------------------------------------------------------------- #
# Stochastic control regularization (Patnaik quantile of q^T q)
# --------------------------------------------------------------------------- #

def patnaik_rho_nu_from_cov(q_bar, P_q):
    """Patnaik (Pearson Type III / scaled chi-squared) approximation of the
    generalized non-central chi-squared distribution of Q = q^T q, where
    q ~ N(q_bar, P_q). Returns (rho, nu) such that Q ~ rho * chi2_nu,
    mean/variance-matched to the true distribution of Q."""
    tr_Pq = jnp.trace(P_q)
    tr_Pq2 = jnp.trace(P_q @ P_q)
    qPq = q_bar @ P_q @ q_bar
    qq = q_bar @ q_bar

    term1 = tr_Pq2 + 2 * qPq
    term2 = tr_Pq + qq

    eps = 1e-12
    rho = term1 / term2
    nu = term2 ** 2 / term1

    return rho, nu


patnaik_rho_nu_from_cov_vmap = jax.vmap(patnaik_rho_nu_from_cov, in_axes=(0, 0))


def _build_chi2_quantile_interp(p):
    """Differentiable F^{-1}_chi2_nu(p) as a function of continuous nu, built
    once via a precomputed grid (scipy, host-side) + diffrax cubic Hermite
    interpolation - same pattern as X0_interp/Xf_interp in problem_definition.py.
    Two pieces: dense near the origin, where the ppf is sharply nonlinear,
    then a sparser, wide-ranging geometric grid out to 1e5, where the ppf is
    practically linear (chi2_nu ~ Normal(nu, 2nu) for large nu)."""
    nu_grid = np.concatenate([
        np.linspace(0.0, 10.0, 1000),
        np.geomspace(10.0, 1e9, 5000)[1:],
    ])
    quantiles = stats.chi2.ppf(p, df=nu_grid)
    nu_grid, quantiles = jnp.asarray(nu_grid), jnp.asarray(quantiles)
    coefs = dfx.backward_hermite_coefficients(nu_grid, quantiles)
    return dfx.CubicInterpolation(nu_grid, coefs)


# Only these two quantiles are needed (99% objective / 99.9% constraint,
# mirroring the existing mx_dV_bound/mx_tcm_bound split) - built once at
# import time since this depends only on (p, nu), never on a specific
# ProblemDefinition.
_CHI2_QUANTILE_INTERPS = {
    0.99: _build_chi2_quantile_interp(0.99),
    0.999: _build_chi2_quantile_interp(0.999),
}


def chi2_inv_cdf(nu, p):
    """F^{-1}_chi2_nu(p) for continuous nu, via the precomputed interpolants
    above. p must be a literal Python float in {0.99, 0.999} (a static
    dispatch key, not a traced value)."""
    return _CHI2_QUANTILE_INTERPS[p].evaluate(nu)


chi2_inv_cdf_vmap = jax.vmap(chi2_inv_cdf, in_axes=(0, None))