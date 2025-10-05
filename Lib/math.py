import jax
import jax.numpy as jnp
import numpy as np
import astropy
from astropy.time import Time
from astropy import units as u


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

def col_avoid(X, r_obs, safe_d):
    delta_X = X[:3]-r_obs
    dist = jnp.sqrt(delta_X[0]**2 + delta_X[1]**2 + delta_X[2]**2)
    
    return safe_d - dist

col_avoid_vmap = jax.vmap(col_avoid, in_axes=(0, None, None))


# ----------------
# Numpy Functions
# ----------------

def sig2cov(r_1sig, v_1sig, m_1sig, Sys):
    r_cov = (r_1sig/Sys['Ls'])**2
    v_cov = (v_1sig/Sys['Vs'])**2
    m_cov = (m_1sig/Sys['Ms'])**2

    return np.diag(np.array([r_cov, r_cov, r_cov, v_cov, v_cov, v_cov, m_cov]))

