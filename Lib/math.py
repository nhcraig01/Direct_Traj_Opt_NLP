from operator import neg
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

def smooth_clip(x,a):
    x_smooth = (jax.nn.softplus(a*x)-jax.nn.softplus(a*(x-1.0)))/a
    return x_smooth

def smooth_sign(x,a):
    return jax.nn.tanh(a*x)

def smooth_box(x,a, t1, t2):
    return .5*(jax.nn.tanh(a*(x+t1))-jax.nn.tanh(a*(x-t2)))

def smooth_val2val(x, a, val1, val2):
    return val1 + (val2-val1)*(-1/2*jax.nn.tanh(a*x)+1/2)

smooth_val2val_vmap = jax.vmap(smooth_val2val, in_axes=(0,None,None,None))

def dot_norm(x,y):
    return jnp.dot(x,y)/(jnp.linalg.norm(x)*jnp.linalg.norm(y)+1e-10)

def col_avoid_interp(X_n, X_n1, dyn_args):
    r_obj = dyn_args['r_obj']
    safe_d = dyn_args['d_safe']

    r_n = X_n[:3]
    v_n = X_n[3:6]
    r_n1 = X_n1[:3]
    v_n1 = X_n1[3:6]

    r_nn1 = r_n1 - r_n

    t_st = ((r_obj-r_n).T@r_nn1)/(r_nn1.T@r_nn1)
    t_st_clipped = smooth_clip(t_st, a=100.0) # Clips t to [0,1]

    r_t_st = r_n + t_st_clipped*r_nn1
    dist_pure = jnp.linalg.norm(r_t_st-r_obj+1.0e-10)
    safe_val_pure = safe_d - dist_pure
    # Now multiply by the sign of the velocity dot to make sure r_n is moving away from the object
    dot_term1 = v_n - v_n1
    dot_term_2 = r_t_st - r_obj
    check_1 = dot_norm(dot_term1, dot_term_2)

    vel_sign_term = smooth_sign(check_1, a=20.0)
    safe_val_pure_signed = safe_val_pure#*vel_sign_term
    # Now multiply by a smooth box and add a small negative when t_st is outside of [0,1]
    neg_val = 0.1
    #safe_val_smoothed = (safe_val_pure_signed+neg_val)*smooth_box(-safe_val_pure,a = 100, t1 = 0.1, t2 = 0.05) - neg_val
    switch_dist = 2.e-2
    safe_val_smoothed = smooth_val2val(-safe_val_pure-switch_dist,a = 500, val1 = safe_val_pure, val2 = safe_val_pure_signed)
    #jax.debug.print("t_st: {t_st}, disp_val: {disp_val}", t_st=t_st, disp_val=safe_val_pure) # Debug print to check values

    return safe_val_smoothed 

    # Ok pick this up later. Right now we are seeing that the multiplication of the velocity sign term is causing
    # some problems that the optimizer is actually leveraging to allow for flying through the exact center of the moon.
    # Specifically this is esenetially dot_term1 going to zero (not sure how that works though when dividing by its norm)
    # In my attempts to get the trajectory to avoid the moon, I have somehow encouraged it to literally fly though
    # THE VERY CENTER............................... this is a problem for after coffee.
col_avoid_interp_vmap = jax.vmap(col_avoid_interp, in_axes=(0,0,None))

def  interp_col_avoid_vmap(X_hst, dyn_args):
    col_avoid_vals = col_avoid_interp_vmap(X_hst[:-1,:], X_hst[1:,:], dyn_args)
    return col_avoid_vals

def col_avoid_keplr(X_n, dyn_args, Sys):
    r_n = X_n[:3]
    v_n = X_n[3:6]
    mu_body = Sys['mu']
    r_obj = dyn_args['r_obj']
    d_safe = dyn_args['d_safe']
    dist_n = jnp.linalg.norm(r_n-r_obj)

    h = jnp.cross(r_n, v_n)
    h_norm = jnp.linalg.norm(h)
    e = jnp.cross(v_n, h)/mu_body - r_n/jnp.linalg.norm(r_n)
    e_norm = jnp.linalg.norm(e)

    r_per = h_norm**2/(mu_body*(1+e_norm))

    safe_val_n = d_safe - dist_n
    safe_val_per = d_safe - r_per
    #jax.debug.print("safe_val_n: {safe_val_n}, safe_val_per: {safe_val_per}", safe_val_n=safe_val_n, safe_val_per=safe_val_per)

    safe_val_smoothed = smooth_val2val(-safe_val_n-d_safe/2,a = 40, val1 = safe_val_n, val2 = safe_val_per)

    return safe_val_smoothed

col_avoid_keplr_vmap = jax.vmap(col_avoid_keplr, in_axes=(0,None,None))

def poly_coefs(X_n, X_n1):
    r_n = X_n[:3]
    v_n = X_n[3:6]
    r_n1 = X_n1[:3]

    a = r_n
    b = v_n
    c = r_n1 - v_n - r_n
    return a, b, c

def poly_fit_d_dd_vals(s, a,b,c, r_obj):
    r_s = a + b*s + c*s**2
    d_r_s = b + 2*c*s
    norm = jnp.linalg.norm(r_s-r_obj+1e-6)
    dist_ds = (r_s-r_obj).T@d_r_s/norm
    dist_dds = d_r_s.T@d_r_s/norm - (r_s-r_obj).T@d_r_s*((r_s-r_obj).T@d_r_s/(norm**3)) + (r_s-r_obj).T@(2*c)/norm
    
    return dist_ds, dist_dds

def poly_dist_eval(s,a,b,c, r_obj):
    r_s = a + b*s + c*s**2
    dist = jnp.linalg.norm(r_s-r_obj)
    return dist

poly_dist_ds = jax.grad(poly_dist_eval, argnums=0)
poly_dist_dds = jax.grad(poly_dist_ds, argnums=0)


def poly_min_iterate(i, input_dict):
    a = input_dict['a']
    b = input_dict['b']
    c = input_dict['c']
    r_obj = input_dict['r_obj']
    s = input_dict['s']

    #dist_ds = poly_dist_ds(s, a, b, c, r_obj)
    #dist_dds = poly_dist_dds(s, a, b, c, r_obj)

    dist_ds, dist_dds = poly_fit_d_dd_vals(s,a,b,c, r_obj)

    s_new = s - dist_ds/dist_dds
    s_new_clipped = smooth_clip(s_new, a = 100.0)

    output_dict = input_dict.copy()
    output_dict['s'] = s_new_clipped

    return output_dict

def poly_min_col_avoid(X_n, X_n1, dyn_args):
    r_obj  = dyn_args['r_obj']
    safe_d = dyn_args['d_safe']
    dt = (dyn_args['t_node_bound'][1] - dyn_args['t_node_bound'][0])/10.0
    X_n = X_n.at[3:6].mul(dt)

    a, b, c = poly_coefs(X_n, X_n1)
    iters = 5
    # Find zero from 0
    s0_guess = 0.5
    s0_min = jax.lax.fori_loop(0,iters, poly_min_iterate, {'a': a, 'b': b, 'c': c, 'r_obj': r_obj, 's': s0_guess})['s']
    col_val_left = safe_d - poly_dist_eval(s0_min, a, b, c, r_obj)
    #jax.debug.print("dis_ds0 {dis_ds0}, dis_dds0 {dis_dds0}", dis_ds0=poly_dist_ds(s_min, a, b, c, r_obj), dis_dds0=poly_dist_dds(s_min, a, b, c, r_obj))
        
    # Find zero from 1
    #s1_guess = 1.0
    #s1_min = jax.lax.fori_loop(0,iters, poly_min_iterate, {'a': a, 'b': b, 'c': c, 'r_obj': r_obj, 's': s1_guess})['s']
    #col_val_right = safe_d - poly_dist_eval(s1_min, a, b, c, r_obj)

    #jax.debug.print("a {a}, b {b}, c {c}, s0_min {s0_min}, col_val_left {col_val_left}", a=a, b=b, c=c, s0_min=s0_min, col_val_left=col_val_left)
    return col_val_left

poly_min_col_avoid_vmap = jax.vmap(poly_min_col_avoid, in_axes=(0,0,None))

def poly_interp_col_avoid_vmap(X_hst, dyn_args):
    col_avoid_vals = poly_min_col_avoid_vmap(X_hst[:-1,:], X_hst[1:,:], dyn_args)
    return col_avoid_vals

# Writing a quick note so you don't forget what you've done while you return to Aerospace work.
# So this is what we've tried and the results and problems:
#   1. Simply using an pure linear interpolation between positions from one sub-arc node to the next. This
#   did work for some cases but the closer you get to the moon or your primary the more there became a possibility 
#   of failure especially with longer time of flights between each point. We observed that the interpolations
#   would mostly skip off the collision sphere but also, with very close, low fuel flybys, we would see the line
#   go on the other side of the body essentailly resulting in no collision detected but could be arbitrarily
#   close at nearest approach between those two nodes. Couple of different solutions attempted here but ultimately
#   none really worked.
#   2. Next, we tried assuming that in the vicinity of the body the trajectory would look mostly keplerian. With 
#   that we could approximate the periapsis distance from the body just using the state at that body and the
#   keplerian gravity parameter mu. This would in theory work just fine, however with this we don't want to be
#   checking the parameter if our point is super far away. To do this we implement a smoothing function to switch
#   between the regular point collision avoidance calculation and the keplerian periapsis term. However, because
#   of the smoothing term we observed that the optimizer just tried to push the point we're evaluating outside of
#   that region and then because it's already far away the collision is not detected even though the keplerian term
#   would result in a violation. Maybe a way around this but so far no immediate idea.
#   3. Finally, we tried fitting a quadratic 3D polynomial to two points on either end of the sub-arc along with
#   the velocity of the first point. The actual approximation did work very well in testing and we have seen some 
#   optimizations that looked very good, however we still have an issue where in some cases it's pushing one of the 
#   end points to the literal border of the keep out zone. For some reason this is causing our iteration scheme that
#   determines the minimum distance to choose that point and not the actual interior point that minimizes that 
#   distance. This is the most viable option and I think there is something to it, just need to debug further 
#   why the iteration scheme is, in that last case of a near passing through the center, causing the constraint to
#   not be violated. 
#   
#   Will pick this up later but need to focus on sub-arc monte carlo functionality...  

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
