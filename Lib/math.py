import jax
import jax.numpy as jnp
import numpy as np
from astropy.time import Time


# ------------------
# Orbital Mechanics
# ------------------
def calc_t_elapsed_nd(t0, tf, nodes, t_star):
    t0_epoch = Time(t0)
    tf_epoch = Time(tf)

    delta_t = tf_epoch - t0_epoch
    t_elapsed_nd = jnp.linspace(0.0, delta_t.sec, nodes)/t_star

    return t_elapsed_nd


# --------------
# JAX Functions
# --------------

def col_avoid(X, dyn_args):
    r_obj = dyn_args['r_obj']
    safe_d = dyn_args['d_safe']

    delta_X = X[:3]-r_obj
    dist = jnp.sqrt(delta_X[0]**2 + delta_X[1]**2 + delta_X[2]**2)
    
    return safe_d - dist

col_avoid_vmap = jax.vmap(col_avoid, in_axes=(0, None))

def stat_col_avoid(X_mean, P_x, dyn_args, cfg_args):
    nx = X_mean.shape[0]

    lam = cfg_args.alpha_UT**2 * (nx + cfg_args.kappa_UT) - nx

    # Sample points
    sigs = jnp.zeros((2*nx+1, nx))

    weights_m = jnp.zeros(2*nx+1)
    weights_p = jnp.zeros(2*nx+1)

    sigs = sigs.at[:].set(X_mean)
    weights_m = weights_m.at[0].set(lam/(nx + lam))
    weights_p = weights_p.at[0].set(lam/(nx + lam) + 1 - cfg_args.alpha_UT**2 + cfg_args.beta_UT)

    weights_m = weights_m.at[1:].set(1/(2*(nx + lam)))
    weights_p = weights_p.at[1:].set(1/(2*(nx + lam)))

    UT_transformer = mat_sqrt((nx + lam)*P_x)

    sigs = sigs.at[1:1+nx,:].add(UT_transformer[:,0:nx])
    sigs = sigs.at[1+nx:2*nx+1,:].add(-UT_transformer[:,0:nx])

    col_avoid_vals = col_avoid_vmap(sigs, dyn_args)

    Y_mean = jnp.sum(weights_m * col_avoid_vals)
    P_y = jnp.sum(weights_p * (col_avoid_vals - Y_mean)**2)

    return Y_mean, P_y

stat_col_avoid_vmap = jax.vmap(stat_col_avoid, in_axes=(0,0,None,None))


def mat_sqrt(A):
    return jnp.linalg.cholesky(A)

def mat_lmax(A):
    eps = 1e-12
    return jnp.linalg.eigvalsh(A + eps*jnp.diag(jnp.linspace(1.,2.,A.shape[0])))[-1]

mat_lmax_vmap = jax.vmap(mat_lmax, in_axes=(0))

def cart2sph(r_vec):
    x, y, z = r_vec[0], r_vec[1], r_vec[2]
    r = jnp.linalg.norm(r_vec)
    th = jnp.arctan2(y, x)*180/jnp.pi
    rho_xy = jnp.hypot(x, y)
    phi_raw = jnp.arctan2(z, rho_xy)*180/jnp.pi

    phi = jnp.where(r>0, phi_raw, 0.0)

    return jnp.array([r, th, phi])

cart2sph_vmap = jax.vmap(cart2sph, in_axes=(0))

# ----------------
# Numpy Functions
# ----------------

def sig2cov(r_1sig, v_1sig, m_1sig, Sys, m0):
    r_cov = (r_1sig/Sys['Ls'])**2
    v_cov = (v_1sig/Sys['Vs'])**2
    m_cov = (m_1sig/m0)**2

    return np.diag(np.array([r_cov, r_cov, r_cov, v_cov, v_cov, v_cov, m_cov]))
