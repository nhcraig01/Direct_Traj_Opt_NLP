import jax
import jax.numpy as jnp
import numpy as np
import diffrax as dfx
import sympy as sp

from Lib.math import col_avoid_vmap, mat_lmax_vmap, mat_lmax, cart2sph, stat_col_avoid_vmap

from tqdm import tqdm


# --------------------------------------------
# Dynamical Models and Covariance Propagators
# --------------------------------------------

def CR3BPDynamics(U_Acc_min_nd, ve, mu, safe_d):
    r_x, r_y, r_z, v_x, v_y, v_z = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z', real=True)
    t = sp.symbols("t")
    m = sp.symbols("m", positive=True)
    u1, u2, u3 = sp.symbols('u1, u2, u3', real=True)
    dtmp, rtmp = sp.symbols('dtmp, rtmp', positive=True)

    eta = .75
    alpha = 2
    a = (3*(eta*safe_d)**(-3 - alpha))/alpha
    b = (eta*safe_d)**(-3) + a*(eta*safe_d)**alpha

    dval = sp.sqrt((r_x + mu) ** 2 + r_y ** 2 + r_z ** 2)
    rval = sp.sqrt((r_x - 1 + mu) ** 2 + r_y ** 2 + r_z ** 2)

    u_norm = sp.sqrt(u1 ** 2 + u2 ** 2 + u3 ** 2 + 1e-12)

    states = sp.Matrix([[r_x],
                        [r_y],
                        [r_z],
                        [v_x],
                        [v_y],
                        [v_z],
                        [m]])

    term1 = -(1 - mu)*(r_x + mu)*dtmp - mu*(r_x - 1 + mu)*rtmp + r_x
    term2 = -(1 - mu)*r_y*dtmp - mu*r_y*rtmp + r_y
    term3 = -(1 - mu)*r_z*dtmp - mu*r_z*rtmp

    dmod_term = -a*dval**alpha + b
    rmod_term = -a*rval**alpha + b

    dmod = sp.Piecewise((dmod_term, dval <= eta*safe_d), (1/dval**3, dval > eta*safe_d))
    rmod = sp.Piecewise((rmod_term, rval <= eta*safe_d), (1/rval**3, rval > eta*safe_d))
    subs_dict = {rtmp: rmod, dtmp: dmod}

    eoms_pre = sp.Matrix([[v_x],
                      [v_y],
                      [v_z],
                      [term1 + 2 * v_y + u1 * U_Acc_min_nd / m],
                      [term2 - 2 * v_x + u2 * U_Acc_min_nd / m],
                      [term3 + u3 * U_Acc_min_nd / m],
                      [-u_norm * U_Acc_min_nd / ve]])
    
    eoms = eoms_pre.subs(subs_dict)

    inputs = sp.Matrix([[t],
                        [r_x],
                        [r_y],
                        [r_z],
                        [v_x],
                        [v_y],
                        [v_z],
                        [m],
                        [u1],
                        [u2],
                        [u3]])

    controls = sp.Matrix([[u1],
                          [u2],
                          [u3]])

    Aprop = eoms.jacobian(states)
    Bprop = eoms.jacobian(controls)

    eom_eval = sp.lambdify((t, states, controls), eoms, 'jax')
    Aprop_eval = sp.lambdify((t, states, controls), Aprop, 'jax')
    Bprop_eval = sp.lambdify((t, states, controls), Bprop, 'jax')

    return eom_eval, Aprop_eval, Bprop_eval

def TrueStateFeedback_CovPropagators():

    def perArc_propagator(A_k, B_k, K_k, P_k, G_exe, G_stoch):
        
        # Propagate covariance
        mod_A = A_k + B_k @ K_k

        P_k1 = mod_A @ P_k @ mod_A.T + B_k @ (G_exe + G_stoch) @ B_k.T
        return P_k1

    def innerArc_propagator(A_i, B_i, P_i, tau_i, gam_i, P0_arc, K_arc, G_exe_arc, G_stoch_arc):
        tau_gam_term = A_i @ (tau_i @ K_arc.T + gam_i) @ B_i.T

        P_i1 = A_i@P_i@A_i.T + B_i@(G_exe_arc + G_stoch_arc + K_arc@P0_arc@K_arc.T)@B_i.T + tau_gam_term + tau_gam_term.T
        tau_i1 = A_i @ tau_i + B_i @ K_arc @ P0_arc
        gam_i1 = A_i @ gam_i + B_i @ (G_exe_arc + G_stoch_arc)

        return P_i1, tau_i1, gam_i1

    cov_propagators = {'perArc': perArc_propagator,
                       'innerArc': innerArc_propagator}

    return cov_propagators

def EstimatedStateFeedback_CovPropagators():
    # Pickup here. I think we've finished the general EKF time and measurement update functions.
    # Next we need to make the covariance propagators that follow the EKF dynamics for inside the optimizer.
    return None


# ------------------
# Controller Functions
# ------------------

def gates2Gexe(U, gates):
    eps = 1e-12
    norm_U = jnp.sqrt(U[0]**2 + U[1]**2 + U[2]**2 + eps)
    cov_1 = gates[2]**2 + (gates[3]*norm_U)**2
    cov_3 = gates[0]**2 + (gates[1]*norm_U)**2
    P_exe = jnp.diag(jnp.array([cov_1, cov_1, cov_3])) # control execution covariance matrix (in SC frame)

    # Transform to rotating frame
    Z_hat = U.flatten()/norm_U
    E_vec = jnp.cross(jnp.array([0.,0.,1.]), Z_hat.flatten())
    E_hat = E_vec / jnp.sqrt(E_vec[0]**2 + E_vec[1]**2 + E_vec[2]**2 + eps)

    S_vec = jnp.cross(E_hat, Z_hat)
    S_hat = S_vec / jnp.sqrt(S_vec[0]**2 + S_vec[1]**2 + S_vec[2]**2 + eps)

    rot_mat = jnp.column_stack([S_hat, E_hat, Z_hat])

    G_exe = rot_mat @ P_exe @ rot_mat.T
    
    return G_exe

def MC_U_tcm_k(X_k_nom, X_k_trial, K_k):

    U_tcm = K_k @ (X_k_trial-X_k_nom)

    return U_tcm

def MC_U_exe(U_nom, gates, rng_key):
    U_exe = jax.random.multivariate_normal(rng_key, jnp.zeros((3,1)).flatten(), gates2Gexe(U_nom, gates))

    return U_exe

def xi2K(xi_k, A_k, B_k):
    Bkr = B_k[:3,:]
    Bkv = B_k[3:6,:]
    
    blkdiagr = xi_k[0]*jnp.linalg.inv(Bkr.T@Bkr)
    blkdiagv = xi_k[1]*jnp.linalg.inv(Bkv.T@Bkv)

    weights = jax.scipy.linalg.block_diag(blkdiagr, blkdiagv)
    K_k = -jnp.linalg.inv(jnp.eye(3) + B_k[:6,:].T @ weights @ B_k[:6,:]) @ B_k[:6,:].T @ weights @ A_k[:6,:6]

    return jnp.hstack([K_k, jnp.zeros((3,1))])

# --------------------
# Estimator Functions
# --------------------

def EKF_time(vals_k, dyn_vals, noise_terms):
    # Unpack the current estimate and error values
    Xhat_k = vals_k['Xhat_k']
    Xtild_k = vals_k['Xtild_k']
    Phat_k = vals_k['Phat_k']
    Ptild_k = vals_k['Ptild_k']

    # Unpack the dynamical values
    A_k = dyn_vals['A_k']
    B_k = dyn_vals['B_k']
    K_k = dyn_vals['K_k']

    # Unpack noise terms
    G_exe = noise_terms['G_exe']
    G_stoch = noise_terms['G_stoch']

    Astar_k = A_k + B_k @ K_k

    # Propagate the mean values
    Xhat_k1 = Astar_k @ Xhat_k
    Xtild_k1 = A_k @ Xtild_k

    # Propagate the covariance values
    Phat_k1 = Astar_k @ Phat_k @ Astar_k.T
    Ptild_k1 = A_k @ Ptild_k @ A_k.T + B_k @ (G_exe + G_stoch) @ B_k.T
    P_k1 = Phat_k1 + Ptild_k1

    priori_vals_k1 = {'Xhat_k1': Xhat_k1,
                      'Xtild_k1': Xtild_k1,
                      'Phat_k1': Phat_k1,
                      'Ptild_k1': Ptild_k1,
                      'P_k1': P_k1}

    return priori_vals_k1

def EKF_measurement(priori_vals_k, meas_terms):
    # Unpack the apriori estimate and error values
    Xhat_k = priori_vals_k['Xhat_k']
    Xtild_k = priori_vals_k['Xtild_k']
    Phat_k = priori_vals_k['Phat_k']
    Ptild_k = priori_vals_k['Ptild_k']

    # Unpack the measurement values
    y_k = meas_terms['y_k']
    ybar_k = meas_terms['ybar_k']
    H_k = meas_terms['H_k']
    P_v = meas_terms['P_v']

    # Compute the Kalman Gain
    C_k = Ptild_k @ H_k.T
    W_k = H_k @ Ptild_k @ H_k.T + P_v
    L_k = jax.scipy.linalg.solv(W_k.T, C_k.T).T  # L_k = C_k @ jnp.linalg.inv(W_k)

    # Update the mean values
    update_term = L_k(y_k - (ybar_k + H_k @ Xhat_k))
    Xhat_k_p = Xhat_k + update_term
    Xtild_k_p = Xtild_k + update_term

    # Update the covariance values
    update_term = (jnp.eye(7) - L_k @ H_k)
    Phat_k_p = Phat_k + L_k @ W_k @ L_k.T
    Ptild_k_p = update_term @ Ptild_k @ update_term.T + L_k @ P_v @ L_k.T

    P_k_p = Phat_k_p + Ptild_k_p

    posteriori_vals_k = {'Xhat_k_p': Xhat_k_p,
                         'Xtild_k_p': Xtild_k_p,
                         'Phat_k_p': Phat_k_p,
                         'Ptild_k_p': Ptild_k_p,
                         'P_k_p': P_k_p}

    return posteriori_vals_k


# -------------------
# Measurement Models
# -------------------

def range_measurement_model(r_obs):
    # State to be estimated
    r_x, r_y, r_z, v_x, v_y, v_z, m = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z, m', real=True)
    X = sp.Matrix([[r_x],
                    [r_y],
                    [r_z],
                    [v_x],
                    [v_y],
                    [v_z],
                    [m]])

    # Observer position
    obs_x, obs_y, obs_z = sp.symbols('obs_x, obs_y, obs_z', real=True)
    r_obs = sp.Matrix([[obs_x],
                        [obs_y],
                        [obs_z]])

    range = sp.sqrt((r_x - obs_x)**2 + (r_y - obs_y)**2 + (r_z - obs_z)**2)
    range_jac = range.jacobian(X)

    h_eval = sp.lambdify((X, r_obs), range, 'jax')
    H_eval = sp.lambdify((X, r_obs), range_jac, 'jax')

    return h_eval, H_eval


# --------------------------------
# Numerical Propagation Functions
# --------------------------------

def eoms_gen(t, states, args, dynamics, cfg_args):
    """ This function creates general EOMs wrapped from the basline system EOMS with the stochastic terms if needed
    """
    if cfg_args.det_or_stoch.lower() == 'deterministic' or cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        U = args
        X = states[:7]
        Xdot = dynamics['eom_eval'](t,X,U).reshape(-1)
        
        return Xdot
    
def propagator_gen(X0, U, t0, t1, EOM, cfg_args):
    """ This function creates a general integrator using diffrax
    """

    # Unpack (JAX-variable) inputs and (JAX-static) arguments
    r_tol, a_tol, N_save = cfg_args.r_tol, cfg_args.a_tol, cfg_args.N_save

    term = dfx.ODETerm(EOM)
    solver = dfx.Dopri8()

    stepsize_controller = dfx.PIDController(rtol=r_tol, atol=a_tol)
    if N_save > 1:
        save_t = dfx.SaveAt(ts=jnp.linspace(t0,t1,N_save))
    else:
        save_t = dfx.SaveAt(t1=True)


    sol = dfx.diffeqsolve(term,
                            solver, 
                            t0, 
                            t1, 
                            None, 
                            X0, 
                            args=U,
                            stepsize_controller=stepsize_controller, 
                            adjoint=dfx.DirectAdjoint(),
                            saveat=save_t,
                            max_steps=16**5)

    return sol if N_save > 1 else sol.ys[-1].flatten()

def propagator_cov(A_k, B_k, K_k, P_k, G_exe, G_stoch):
    mod_A = A_k + B_k @ K_k
    P_k1 = mod_A @ P_k @ mod_A.T + B_k @ (G_exe + G_stoch) @ B_k.T
    return P_k1


# --------------------------------
# Iterative Propagation Functions
# --------------------------------

def forward_propagation_iterate(k, input_dict, propagators, dyn_args, cfg_args):
    X0_true_f = input_dict['X0_true_f']
    states = input_dict['states']
    controls = input_dict['controls']   

    t_node_bound = dyn_args['t_node_bound']

    sol_f = propagators['propagator_e'](X0_true_f, controls[k,:], t_node_bound[k], t_node_bound[k+1], cfg_args)
    states = states.at[k,:,:].set(sol_f.ys[:,:7])
 
    
    output_dict = {'states': states, 'controls': controls}
    
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = input_dict['A_ks']
        B_ks = input_dict['B_ks']

        tmp_A = propagators['propagator_dX0_e'](X0_true_f, controls[k,:], t_node_bound[k], t_node_bound[k+1], cfg_args)
        tmp_B = propagators['propagator_dU_e'](X0_true_f, controls[k,:], t_node_bound[k], t_node_bound[k+1], cfg_args)

        A_ks = A_ks.at[k,:,:].set(tmp_A)
        B_ks = B_ks.at[k,:,:].set(tmp_B)

        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks
    
    X0_true_f = sol_f.ys[-1,:7].flatten() 
    output_dict['X0_true_f'] = X0_true_f

    return output_dict

def backward_propagation_iterate(kk,input_dict, propagators, dyn_args, cfg_args):
    X0_true_b = input_dict['X0_true_b']
    states = input_dict['states']
    controls = input_dict['controls']

    t_node_bound = dyn_args['t_node_bound']

    indx_b = dyn_args['indx_b']
    k = indx_b[kk]

    sol_b = propagators['propagator_e'](X0_true_b, controls[k,:], t_node_bound[k+1], t_node_bound[k], cfg_args)
    states = states.at[k,:,:].set(jnp.flipud(sol_b.ys))


    output_dict = {'states': states, 'controls': controls}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = input_dict['A_ks']
        B_ks = input_dict['B_ks']

        tmp_A = propagators['propagator_dX0_e'](X0_true_b, controls[k,:], t_node_bound[k+1], t_node_bound[k], cfg_args)
        tmp_B = propagators['propagator_dU_e'](X0_true_b, controls[k,:], t_node_bound[k+1], t_node_bound[k], cfg_args)

        tmp_A = jnp.linalg.inv(tmp_A)
        tmp_B = -tmp_A @ tmp_B

        A_ks = A_ks.at[k,:,:].set(tmp_A)
        B_ks = B_ks.at[k,:,:].set(tmp_B)

        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks
    
    X0_true_b = sol_b.ys[-1,:7].flatten()
    output_dict['X0_true_b'] = X0_true_b

    return output_dict

def forward_propagation_cov_iterate(k, input_dict, cov_propagators, dyn_args, cfg_args):
    # Unpack inputs
    controls = input_dict['controls']
    xis = input_dict['xis']  
    A_ks = input_dict['A_ks']
    B_ks = input_dict['B_ks']  
    K_ks = input_dict['K_ks']
    P_ks = input_dict['P_ks']
    P_Us = input_dict['P_Us']
    
    A_k = A_ks[k,:,:]
    B_k = B_ks[k,:,:]
    xi_k = xis[k,:]
    
    # Gates execution error covariance
    gates = dyn_args['gates']
    G_exe = gates2Gexe(controls[k,:], gates)

    # Stochastic Gauss ZOH acceleration covariance
    G_stoch = dyn_args['G_stoch']

    # Compute colsed loop gain
    K_k = xi2K(xi_k, A_k, B_k)
    K_ks = K_ks.at[k,:,:].set(K_k)

    # Propagate covariance
    P_k = P_ks[k,:,:]
    P_k1 = cov_propagators['perArc'](A_k, B_k, K_k, P_k, G_exe, G_stoch)
    P_ks = P_ks.at[k+1,:,:].set(P_k1)

    # Compute control covariance
    P_Ui = K_k @ P_k @ K_k.T
    P_Us = P_Us.at[k,:,:].set(P_Ui)

    output_dict = {'controls': controls, 'xis': xis, 'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'P_ks': P_ks, 'P_Us': P_Us} 
    return output_dict


# -----------------------------------
# Constraint and Objective Function
# -----------------------------------

def objective_and_constraints(inputs, Boundary_Conds, iterators, dyn_args, cfg_args):
    output_dict = {}

    # Unpack args
    X_start = Boundary_Conds['X0_interp']
    X_end = Boundary_Conds['Xf_interp']

    N_nodes = cfg_args.N_nodes
    N_arcs = cfg_args.N_arcs    

    # Unpack inputs
    X0 = inputs['X0']
    Xf = inputs['Xf']
    controls = inputs['controls'].reshape(N_arcs,3)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        xis = inputs['xis'].reshape(N_arcs,2)
    if cfg_args.free_phasing:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Forward and backward node indices
    indx_f = dyn_args['indx_f']
    indx_b = dyn_args['indx_b']

    # Initialize histories
    states = jnp.zeros((N_arcs,cfg_args.N_save, 7))
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = jnp.zeros((N_arcs, 7, 7))
        B_ks = jnp.zeros((N_arcs, 7, 3))
        K_ks = jnp.zeros((N_arcs, 3, 7))
        P_ks = jnp.zeros((N_nodes, 7, 7))
        P_ks = P_ks.at[0,:,:].set(dyn_args['init_cov'])
        P_Us = jnp.zeros((N_arcs, 3, 3))

    # Propagate dynamics forward (to half)
    X0_true_f = X0
    forward_input_dict = {'X0_true_f': X0_true_f, 'states': states, 'controls': controls}
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        forward_input_dict['A_ks'] = A_ks
        forward_input_dict['B_ks'] = B_ks
    forward_out = jax.lax.fori_loop(0, len(indx_f), iterators['forward_propagation_iterate_e'], forward_input_dict)

    states = forward_out['states']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = forward_out['A_ks']
        B_ks = forward_out['B_ks']

    # Propagate dynamics backwards (to half)
    X0_true_b = Xf
    backward_input_dict = {'X0_true_b': X0_true_b, 'states': states, 'controls': controls}
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        backward_input_dict['A_ks'] = A_ks
        backward_input_dict['B_ks'] = B_ks
    backward_out = jax.lax.fori_loop(0, len(indx_b), iterators['backward_propagation_iterate_e'], backward_input_dict)

    states = backward_out['states']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = backward_out['A_ks']
        B_ks = backward_out['B_ks']

    # Propagate covariance forward through entire trajectory
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        cov_input_dict = {'controls': controls, 'xis': xis, 'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'P_ks': P_ks, 'P_Us': P_Us}
        cov_out = jax.lax.fori_loop(0, N_arcs, iterators['cov_propagation_iterate_e'], cov_input_dict)

        K_ks = cov_out['K_ks']
        P_ks = cov_out['P_ks']
        P_Us = cov_out['P_Us']

    # Objective and Constraints ouputs
    eps = 1e-12
    control_norms = jnp.sqrt(controls[:, 0]**2 + controls[:, 1]**2 + controls[:, 2]**2 + eps)
    J_det = jnp.sum(control_norms)

    if cfg_args.det_or_stoch.lower() == 'deterministic':
        output_dict['o_mf'] = J_det # obejective - minimizing sum of control norms
        output_dict['c_Us'] = control_norms.flatten() # constraint - control norm
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        control_max_eig = cfg_args.mx_tcm_bound * jnp.sqrt(mat_lmax_vmap(P_Us))

        J_stat = jnp.sum(control_max_eig)
        output_dict['o_mf'] = J_det + J_stat # obejective - minimizing sum of control norms and max eigenvalues of control covariances
        output_dict['c_Us'] = control_norms.flatten() + control_max_eig.flatten() # constraint - stochastic control norm

        P_Xf = dyn_args['targ_cov_inv_sqrt']@P_ks[-1,:,:]@dyn_args['targ_cov_inv_sqrt'].T - jnp.eye(7)
        output_dict['c_P_Xf'] = jnp.log10(mat_lmax(P_Xf)+1) # constraint - final state covariance

    if cfg_args.free_phasing:
        output_dict['c_X0'] = (X0[:7] - jnp.hstack([X_start.evaluate(alpha).flatten(), 1.])).flatten() # constraint - X0
        output_dict['c_Xf'] = (Xf[:6] - X_end.evaluate(beta).flatten()).flatten() # constraint - Xf
    else: 
        output_dict['c_X0'] = (X0[:7] - jnp.hstack([X_start, 1.])).flatten() # constraint - X0
        output_dict['c_Xf'] = (Xf[:6] - X_end.flatten()).flatten() # constraint - Xf

    output_dict['c_X_mp'] = (states[indx_f[-1], -1, :7] - states[indx_b[-1], 0, :7]).flatten() # constraint - state match point
    
    node_states = jnp.zeros((N_nodes, 7))
    node_states = node_states.at[0, :].set(states[0, 0, :7])
    node_states = node_states.at[1:, :].set(states[:, -1, :7])
    if cfg_args.det_col_avoid and not cfg_args.stat_col_avoid:
        col_vals = col_avoid_vmap(node_states, dyn_args)
        output_dict['c_det_col_avoid'] = col_vals.flatten() # constraint - deterministic collision avoidance
        col_print = jnp.max(output_dict['c_det_col_avoid'])
    if cfg_args.stat_col_avoid and cfg_args.det_or_stoch.lower() != 'deterministic':
        Y_mean, P_y = stat_col_avoid_vmap(node_states, P_ks, dyn_args, cfg_args)
        stat_col_vals = Y_mean + cfg_args.mx_col_bound * jnp.sqrt(P_y + eps)
        output_dict['c_stat_col_avoid'] = stat_col_vals.flatten() # constraint - statistical collision avoidance
        col_print = jnp.max(output_dict['c_stat_col_avoid'])

    # still need to add stochastic col avoidance
    
    if cfg_args.det_or_stoch.lower() == 'deterministic':
        base_str = "J: {:.3e},    X0: {:.1e},    Xf: {:.0e},    X_mp: {:.0e},    Col: {:.0e}"

        jax.debug.print(base_str, output_dict['o_mf'], 
                        jnp.max(jnp.abs(output_dict['c_X0'])), 
                        jnp.max(jnp.abs(output_dict['c_Xf'])), 
                        jnp.max(jnp.abs(output_dict['c_X_mp'])), 
                        col_print)
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        base_str = "J_det: {:.3e}, J_stat: {:.3e}, X0: {:.0e}, Xf: {:.0e}, X_mp: {:.0e}, P_Xf: {:.0e}, Col: {:.0e}, max_xi: {:.1e}"

        jax.debug.print(base_str, J_det, J_stat,
                        jnp.max(jnp.abs(output_dict['c_X0'])),
                        jnp.max(jnp.abs(output_dict['c_Xf'])),
                        jnp.max(jnp.abs(output_dict['c_X_mp'])),
                        jnp.max(output_dict['c_P_Xf']),
                        col_print,
                        jnp.max(xis))

    return output_dict


# ------------------
# Solution Analysis
# ------------------
def sim_Det_traj(sol, Sys, propagators, dyn_args, cfg_args):
    # Unpack inputs
    t_node_bound = dyn_args['t_node_bound']
    X0_det = sol.xStar['X0']
    U_arc_hst = sol.xStar['controls'].reshape(cfg_args.N_arcs, 3)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        xi_arc_hst = sol.xStar['xis'].reshape(cfg_args.N_arcs, 2)

    # Unpack propagators
    propagator_e = propagators['propagator_e']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        propagator_dX0_e = propagators['propagator_dX0_e']
        propagator_dU_e = propagators['propagator_dU_e']

        propagator_cov_innerArc = propagators['cov_propagators']['innerArc']

    # Initialize outputs
    arc_length = cfg_args.N_save - 1
    length = cfg_args.N_arcs*(cfg_args.N_save-1) + 1
    X_hst = jnp.zeros((length, 7))
    X_node_hst = jnp.zeros((cfg_args.N_nodes, 7))
    U_hst = jnp.zeros((length, 3))
    t_hst = jnp.zeros((length,))
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        xi_hst = jnp.zeros((length-1, 2))
        K_arc_hst = jnp.zeros((cfg_args.N_arcs, 3, 7))
        TCM_norm_bound_hst = jnp.zeros((length,))
        TCM_norm_dV_hst = jnp.zeros((length,))
        U_norm_bound_hst = jnp.zeros((length,))
        U_norm_dV_hst = jnp.zeros((length,))
        A_hst = jnp.zeros((length-1, 7, 7))
        B_hst = jnp.zeros((length-1, 7, 3))
        K_hst = jnp.zeros((length-1, 3, 7))
        P_hst = jnp.zeros((length, 7, 7))
        P_u_hst = jnp.zeros((length-1, 3, 3))
    
    # Initialize first entries
    X_hst = X_hst.at[0,:].set(X0_det)
    X_node_hst = X_node_hst.at[0,:].set(X0_det)
    t_hst = t_hst.at[0].set(t_node_bound[0])
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        P_hst = P_hst.at[0,:,:].set(dyn_args['init_cov'])

    # Loop through each arc
    for k in range(cfg_args.N_arcs):
        arc_i_0 = k*(cfg_args.N_save-1)  # starting point index for this arc in saved history
        arc_i_f = arc_i_0 + arc_length  # starting point index for next arc in saved history

        # Get the initial conditions for this arc
        X0_arc = X_hst[arc_i_0,:]
        U_arc = U_arc_hst[k,:]
        if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
            xi_arc = xi_arc_hst[k,:]
            A_arc = propagator_dX0_e(X0_arc, U_arc, t_node_bound[k], t_node_bound[k+1], cfg_args)
            B_arc = propagator_dU_e(X0_arc, U_arc, t_node_bound[k], t_node_bound[k+1], cfg_args)
            K_arc = xi2K(xi_arc, A_arc, B_arc)
            K_arc_hst = K_arc_hst.at[k,:,:].set(K_arc)
            G_exe_arc = gates2Gexe(U_arc, dyn_args['gates'])
            G_stoch_arc = dyn_args['G_stoch']
            
            P0_arc = P_hst[arc_i_0,:,:]
            P_u_arc = K_arc @ P0_arc @ K_arc.T

        # Propagate the state across arc
        sol_f_arc = propagator_e(X0_arc, U_arc, t_node_bound[k], t_node_bound[k+1], cfg_args)
        X_hst_arc = sol_f_arc.ys
        t_hst_arc = sol_f_arc.ts
        X_hst = X_hst.at[arc_i_0+1:arc_i_f+1,:].set(X_hst_arc[1:,:])
        t_hst = t_hst.at[arc_i_0+1:arc_i_f+1].set(t_hst_arc[1:])
        U_hst = U_hst.at[arc_i_0:arc_i_f,:].set(jnp.tile(U_arc, (cfg_args.N_save-1,1)))
        X_node_hst = X_node_hst.at[k+1,:].set(X_hst_arc[-1,:])

        # Propagate covariances if stochastic
        if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
            # Propagate control values and tile across arc
            xi_hst = xi_hst.at[arc_i_0:arc_i_f,:].set(jnp.tile(xi_arc, (arc_length,1)))
            K_hst = K_hst.at[arc_i_0:arc_i_f,:,:].set(jnp.tile(K_arc, (arc_length,1,1)))
            P_u_hst = P_u_hst.at[arc_i_0:arc_i_f,:,:].set(jnp.tile(P_u_arc, (cfg_args.N_save-1,1,1)))

            TCM_norm_bound_arc = cfg_args.mx_tcm_bound*jnp.sqrt(mat_lmax(P_u_arc))
            TCM_dV_arc = cfg_args.mx_dV_bound*jnp.sqrt(mat_lmax(P_u_arc))
            U_norm_bound_arc = jnp.linalg.norm(U_arc) + TCM_norm_bound_arc
            U_norm_dV_arc = jnp.linalg.norm(U_arc) + TCM_dV_arc

            TCM_norm_bound_hst = TCM_norm_bound_hst.at[arc_i_0:arc_i_f].set(jnp.tile(TCM_norm_bound_arc, arc_length))
            TCM_norm_dV_hst = TCM_norm_dV_hst.at[arc_i_0:arc_i_f].set(jnp.tile(TCM_dV_arc, arc_length))
            U_norm_bound_hst = U_norm_bound_hst.at[arc_i_0:arc_i_f].set(jnp.tile(U_norm_bound_arc, arc_length))
            U_norm_dV_hst = U_norm_dV_hst.at[arc_i_0:arc_i_f].set(jnp.tile(U_norm_dV_arc, arc_length))


            # Loop through arc to propagate detailed state covariances
            tau = P0_arc
            gam = jnp.zeros((7,3))
            for i in range(cfg_args.N_save-1):

                X_i = X_hst_arc[i,:]
                P_i = P_hst[arc_i_0+i,:,:]
                A_i = propagator_dX0_e(X_i, U_arc, t_hst_arc[i], t_hst_arc[i+1], cfg_args)
                B_i = propagator_dU_e(X_i, U_arc, t_hst_arc[i], t_hst_arc[i+1], cfg_args)

                A_hst = A_hst.at[arc_i_0+i,:,:].set(A_i)
                B_hst = B_hst.at[arc_i_0+i,:,:].set(B_i)

                # You need to propagate additional terms for the assumption of ZOH closed loop feedback and stochastic noise
                P_i1, tau, gam = propagator_cov_innerArc(A_i, B_i, P_i, tau, gam, P0_arc, K_arc, G_exe_arc, G_stoch_arc)

                P_hst = P_hst.at[arc_i_0+i+1,:,:].set(P_i1)

    U_hst_sph = cart2sph(U_hst)


    dt = t_hst[1] - t_hst[0]
    dV_mean = jnp.sum(jnp.linalg.norm(U_hst, axis=1)*cfg_args.U_Acc_min_nd*dt / X_hst[:,-1])*Sys['Vs']

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        dV_stat = jnp.sum(TCM_norm_dV_hst*cfg_args.U_Acc_min_nd*dt / X_hst[:,-1])*Sys['Vs']
        dV_bound = jnp.sum(U_norm_dV_hst*cfg_args.U_Acc_min_nd*dt / X_hst[:,-1])*Sys['Vs']

    output_dict = {'X_hst': X_hst, 
                   'X_node_hst': X_node_hst, 
                   'U_hst': U_hst, 
                   'U_hst_sph': U_hst_sph, 
                   'U_arc_hst': U_arc_hst, 
                   't_hst': t_hst, 
                   't_node_hst': t_node_bound, 
                   'dV_mean': dV_mean}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        output_dict['TCM_norm_dV_hst'] = TCM_norm_dV_hst
        output_dict['TCM_norm_bound_hst'] = TCM_norm_bound_hst
        output_dict['U_norm_dV_hst'] = U_norm_dV_hst
        output_dict['U_norm_bound_hst'] = U_norm_bound_hst
        output_dict['dV_stat'] = dV_stat
        output_dict['dV_bound'] = dV_bound
        output_dict['A_hst'] = A_hst
        output_dict['B_hst'] = B_hst
        output_dict['K_hst'] = K_hst
        output_dict['K_arc_hst'] = K_arc_hst
        output_dict['P_hst'] = P_hst
        output_dict['P_u_hst'] = P_u_hst

    return output_dict

def single_MC_trial(rng_key, inputs, Sys, dyn_args, cfg_args, propagator):
    # Unpack inputs
    t_node_bound = dyn_args['t_node_bound']
    det_X_node_hst = inputs['det_X_node_hst']
    det_U_arc_hst = inputs['det_U_arc_hst']
    K_arc_hst = inputs['K_arc_hst']
    P_k0 = dyn_args['init_cov']
    dV_mean = inputs['dV_mean']

    # Create rng keys
    keys = jax.random.split(rng_key, 1+2*cfg_args.N_arcs)
    key_X0, keys_U_exe, keys_W_dyn = keys[0], keys[1:1+cfg_args.N_arcs], keys[1+cfg_args.N_arcs:1+2*cfg_args.N_arcs]

    # Initial state and controls
    X0_det = det_X_node_hst[0,:]
    X0_trial = jax.random.multivariate_normal(key_X0, X0_det, P_k0)

    # Initialize outputs
    arc_length = cfg_args.N_save - 1
    length = cfg_args.N_arcs*(cfg_args.N_save-1) + 1
    X_hst = jnp.zeros((length, 7))
    U_hst = jnp.zeros((length, 3))
    t_hst = jnp.zeros((length,))

    # Initialize first entries
    X_hst = X_hst.at[0,:].set(X0_trial)
    t_hst = t_hst.at[0].set(t_node_bound[0])

    # Loop through each arc
    for k in range(cfg_args.N_arcs):
        arc_i_0 = k*(cfg_args.N_save-1)  # starting point index for this arc in saved history
        arc_i_f = arc_i_0 + arc_length  # starting point index for next arc in saved history

        # Get the initial conditions for this arc
        X0_arc_det = det_X_node_hst[k,:]
        X0_arc = X_hst[arc_i_0,:]
        U_arc_det = det_U_arc_hst[k,:]
        U_arc_tcm = MC_U_tcm_k(X0_arc_det, X0_arc, K_arc_hst[k,:,:])
        U_arc_exe = MC_U_exe(U_arc_det, dyn_args['gates'], keys_U_exe[k])
        
        G_stoch_zero = jnp.all(jnp.isclose(dyn_args['G_stoch'], 0.0, atol = 1e-20))
        U_arc_w = jax.lax.cond(G_stoch_zero,
                             lambda key: jnp.zeros(3,),
                             lambda key: jax.random.multivariate_normal(key, jnp.zeros(3,), dyn_args['G_stoch']),
                             keys_W_dyn[k])
        
        U_arc_cmd = U_arc_det + U_arc_tcm
        U_arc_cmd_nsy = U_arc_cmd + U_arc_exe
        U_arc_tot = U_arc_cmd_nsy + U_arc_w

        # Propagate the state across arc
        sol_f_arc = propagator(X0_arc, U_arc_tot, t_node_bound[k], t_node_bound[k+1], cfg_args)
        X_hst_arc = sol_f_arc.ys
        t_hst_arc = sol_f_arc.ts

        X_hst = X_hst.at[arc_i_0+1:arc_i_f+1,:].set(X_hst_arc[1:,:])
        t_hst = t_hst.at[arc_i_0+1:arc_i_f+1].set(t_hst_arc[1:])
        U_hst = U_hst.at[arc_i_0:arc_i_f,:].set(jnp.tile(U_arc_cmd, (cfg_args.N_save-1,1)))

        
    U_hst_sph = cart2sph(U_hst)


    dt = t_hst[1] - t_hst[0]
    dV_trial = jnp.sum(jnp.linalg.norm(U_hst, axis=1)*cfg_args.U_Acc_min_nd*dt / X_hst[:,-1])*Sys['Vs']
    dV_tcm_trial = dV_trial - dV_mean


    return X_hst, U_hst, U_hst_sph, t_hst, dV_trial, dV_tcm_trial

def sim_MC_trajs(inputs, seed, Sys, dyn_args, cfg_args, propagator):

    N = cfg_args.N_trials
    keys = jax.random.split(jax.random.PRNGKey(seed), N)

    jax.debug.print("Running {} MC Trials...", N)
    one_trial = lambda key: single_MC_trial(key, inputs, Sys, dyn_args, cfg_args, propagator)
    MC_Batched = jax.jit(jax.vmap(one_trial), backend='cpu')

    length = cfg_args.N_arcs*(cfg_args.N_save-1) + 1
    X_hsts = jnp.zeros((N,length, 7))
    U_hsts = jnp.zeros((N,length, 3))
    U_hsts_sph = jnp.zeros((N,length, 3))
    t_hsts = jnp.zeros((N,length,))
    dVs = jnp.zeros((N,))
    dV_tcms = jnp.zeros((N,))

    MC_N_Loop = 10
    Loop_N = N // MC_N_Loop
    for i in tqdm(range(Loop_N),):
        rng0 = i*MC_N_Loop
        rngf = (i+1)*MC_N_Loop
        keys_loop = keys[rng0:rngf]
        X_hst_i, U_hst_i, U_hst_sph_i, t_hst_i, dV_i, dV_tcms_i = MC_Batched(keys_loop)
        X_hsts = X_hsts.at[rng0:rngf,:,:].set(X_hst_i)
        U_hsts = U_hsts.at[rng0:rngf,:,:].set(U_hst_i)
        U_hsts_sph = U_hsts_sph.at[rng0:rngf,:,:].set(U_hst_sph_i)
        t_hsts = t_hsts.at[rng0:rngf,:].set(t_hst_i)
        dVs = dVs.at[rng0:rngf].set(dV_i)
        dV_tcms = dV_tcms.at[rng0:rngf].set(dV_tcms_i)

    output_dict = {'X_hsts': np.array(X_hsts),
                   't_hsts': np.array(t_hsts), 
                   'U_hsts': np.array(U_hsts), 
                   'U_hsts_sph': np.array(U_hsts_sph),
                   'dVs': np.array(dVs),
                   'dV_tcms': np.array(dV_tcms)}

    return output_dict
