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

def smooth_clip(x):
    a = 30.0
    x_smooth = (jax.nn.softplus(a*x)-jax.nn.softplus(a*(x-1.0)))/a
    return x_smooth

def col_avoid_interp(X_n, X_n1, dyn_args):
    r_obj = dyn_args['r_obj']
    safe_d = dyn_args['d_safe']

    r_n = X_n[:3]
    r_n1 = X_n1[:3]

    t_st = ((r_obj-r_n).T@(r_n1-r_n))/((r_n1-r_n).T@(r_n1-r_n))

    t_smooth = smooth_clip(t_st)
    
    r_t_st = r_n + t_smooth*(r_n1-r_n)
    dist = jnp.linalg.norm(r_t_st-r_obj)
    
    return safe_d - dist

col_avoid_interp_vmap = jax.vmap(col_avoid_interp, in_axes=(0,0,None))

def  interp_col_avoid_vmap(X_hst, dyn_args):
    col_avoid_vals = col_avoid_interp_vmap(X_hst[:-1,:], X_hst[1:,:], dyn_args)
    return col_avoid_vals


def poly_coefs(r_n, r_n1, r_n2):
    a = (r_n+r_n2)/2 - r_n1
    b = (r_n2-r_n)/2
    c = r_n1
    return a, b, c

def poly_fit_d_dd_vals(a,b,c, r_obj, s):
    r_s = a*s**2 + b*s + c
    tmp_val = 2*a*s + b
    norm = jnp.linalg.norm(r_s-r_obj)
    dist_ds = (r_s-r_obj).T@tmp_val/jnp.linalg.norm(r_s-r_obj)
    dist_dds = tmp_val.T@tmp_val/norm - (r_s-r_obj).T@tmp_val*((r_s-r_obj).T@tmp_val/(norm**3)) + (r_s-r_obj).T@(2*a)/norm
    
    return dist_ds, dist_dds

def poly_min_iterate(i, input_dict):
    a = input_dict['a']
    b = input_dict['b']
    c = input_dict['c']
    r_obj = input_dict['r_obj']
    s = input_dict['s']

    dist_ds, dist_dds = poly_fit_d_dd_vals(a,b,c, r_obj, s)

    s_new = s - dist_ds/dist_dds
    s_new_clipped = smooth_clip(s_new)

    output_dict = input_dict.copy()
    output_dict['s'] = s_new_clipped

    return output_dict

def poly_min_col_avoid(X_n, X_n1, X_n2, dyn_args):
    r_obj  = dyn_args['r_obj']
    safe_d = dyn_args['d_safe']

    a, b, c = poly_coefs(X_n[:3], X_n1[:3], X_n2[:3])

    iters = 6
    # Find zero from -1
    s_guess = -1
    s_min = jax.lax.fori_loop(0,iters, poly_min_iterate, {'a': a, 'b': b, 'c': c, 'r_obj': r_obj, 's': s_guess})['s']
    r_s = a*s_min**2 + b*s_min + c
    col_val_left = safe_d - jnp.linalg.norm(r_s-r_obj)

    # Find zero from 1
    s_guess = 1
    s_min = jax.lax.fori_loop(0,iters, poly_min_iterate, {'a': a, 'b': b, 'c': c, 'r_obj': r_obj, 's': s_guess})['s']
    r_s = a*s_min**2 + b*s_min + c
    col_val_right = safe_d - jnp.linalg.norm(r_s-r_obj)

    return jnp.array([col_val_left, col_val_right])

poly_min_col_avoid_vmap = jax.vmap(poly_min_col_avoid, in_axes=(0,0,0,None))

def poly_interp_col_avoid_vmap(X_hst, dyn_args):
    col_avoid_vals = poly_min_col_avoid_vmap(X_hst[:-2,:], X_hst[1:-1,:], X_hst[2:,:], dyn_args)
    return col_avoid_vals.reshape(-1)


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
