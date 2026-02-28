import jax
import jax.numpy as jnp
import numpy as np
import diffrax as dfx
import sympy as sp

from Lib.math import col_avoid_vmap, interp_col_avoid_vmap, poly_interp_col_avoid_vmap, stat_col_avoid_vmap, mat_lmax_vmap, mat_lmax, cart2sph, cart2sph_vmap

from tqdm import tqdm

from dataclasses import replace


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

    controls = sp.Matrix([[u1],
                          [u2],
                          [u3]])

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

    Aprop = eoms.jacobian(states)
    Bprop = eoms.jacobian(controls)

    eom_eval = sp.lambdify((t, states, controls), eoms, 'jax')
    Aprop_eval = sp.lambdify((t, states, controls), Aprop, 'jax')
    Bprop_eval = sp.lambdify((t, states, controls), Bprop, 'jax')

    U_st_dyn = (1-mu)/dval + mu/rval + 0.5*(r_x**2 + r_y**2)
    U_st_eval = sp.lambdify((states,), U_st_dyn, 'jax')

    JC = 2*U_st_dyn + (v_x**2 + v_y**2 + v_z**2)
    JC_eval = sp.lambdify((states,), JC, 'jax')

    return eom_eval, Aprop_eval, Bprop_eval, U_st_eval, JC_eval

def TrueStateFeedback_CovPropagators():
    
    def perArc_propagator(vals_k, dyn_terms, ctrl_terms):
        # Unpack the current values
        P_k = vals_k['P_k']

        # Unpack the dynamical terms
        A_k = dyn_terms['A_k']
        B_k = dyn_terms['B_k']
        G_stoch = dyn_terms['G_stoch']

        # Unpack the control terms
        K_k = ctrl_terms['K_k']
        G_exe = ctrl_terms['G_exe']

        # Propagate covariance
        mod_A = A_k + B_k @ K_k
        P_k1 = mod_A @ P_k @ mod_A.T + B_k @ (G_exe + G_stoch) @ B_k.T

        vals_k1 = {'P_k1': P_k1}
        return vals_k1
    
    def subArc_iterator(j,input_dict):
        # Unpack unmodified input terms
        A_js = input_dict['A_js']
        B_js = input_dict['B_js']
        K_arc = input_dict['K_arc']
        G_stoch_arc = input_dict['G_stoch_arc']
        G_exe_arc = input_dict['G_exe_arc']

        # Unpack modified input terms
        P_js = input_dict['P_js']
        tau_j = input_dict['tau_j']
        gam_j = input_dict['gam_j']

        # Propagate covariance
        P0_arc = P_js[0,:,:]
        P_u_arc = G_stoch_arc + G_exe_arc
        P_j = P_js[j,:,:]
        A_j = A_js[j,:,:]
        B_j = B_js[j,:,:]

        # Check if propagating across more than one sub arc otherwise do a cheaper version
        if A_js.shape[0] > 1:
            tau_gam_term = A_j @ (tau_j @ K_arc.T + gam_j) @ B_j.T

            P_j1 = A_j@P_j@A_j.T + B_j@(P_u_arc + K_arc@P0_arc@K_arc.T)@B_j.T + tau_gam_term + tau_gam_term.T
            tau_j1 = A_j @ tau_j + B_j @ K_arc @ P0_arc
            gam_j1 = A_j @ gam_j + B_j @ P_u_arc

            tau_j = tau_j1
            gam_j = gam_j1
        else:
            mod_A = A_j + B_j @ K_arc
            P_j1 = mod_A @ P_j @ mod_A.T + B_j @ P_u_arc @ B_j.T

        P_js = P_js.at[j+1,:,:].set(P_j1)
        output_dict = {'A_js': A_js, 'B_js': B_js,
                       'K_arc': K_arc, 'G_stoch_arc': G_stoch_arc, 'G_exe_arc': G_exe_arc,
                       'P_js': P_js, 'tau_j': tau_j, 'gam_j': gam_j}

        return output_dict

    cov_propagators = {'perArc': perArc_propagator,
                       'subArc': subArc_iterator}

    return cov_propagators

def EstimatedStateFeedback_CovPropagators():
    def perArc_propagator(vals_k, dyn_terms, ctrl_terms, meas_terms, meas_update: bool = True):
        # Unpack the current estimate and error values
        Phat_k = vals_k['Phat_k']
        Ptild_k = vals_k['Ptild_k']

        # Unpack the dynamical values
        A_k = dyn_terms['A_k']
        B_k = dyn_terms['B_k']
        G_stoch = dyn_terms['G_stoch']

        # Unpack the control values
        K_k = ctrl_terms['K_k']
        G_exe = ctrl_terms['G_exe']

        # Unpack the measurement values
        if meas_update:
            H_k1 = meas_terms['H_k1']
            P_v = meas_terms['P_v']
        
        # Perform the covariance time propagation
        Astar_k = A_k + B_k @ K_k
        
        Phat_k1 = Astar_k @ Phat_k @ Astar_k.T
        Ptild_k1 = A_k @ Ptild_k @ A_k.T + B_k @ (G_exe + G_stoch) @ B_k.T

        # Perform the measurement update if indicated
        if meas_update:
            # Compute the Kalman Gain
            C_k1 = Ptild_k1 @ H_k1.T
            W_k1 = H_k1 @ Ptild_k1 @ H_k1.T + P_v
            L_k1 = jax.scipy.linalg.solve(W_k1.T, C_k1.T).T  # L_k = C_k1 @ jnp.linalg.inv(W_k1)

            # Update the covariance values
            Phat_k1_p = Phat_k1 + L_k1 @ W_k1 @ L_k1.T
            tmp_mat = jnp.eye(7) - L_k1 @ H_k1
            Ptild_k1_p = tmp_mat @ Ptild_k1 @ tmp_mat.T + L_k1 @ P_v @ L_k1.T
            
            Phat_k1 = Phat_k1_p
            Ptild_k1 = Ptild_k1_p

        vals_k1 = {'Phat_k1': Phat_k1,
                    'Ptild_k1': Ptild_k1,
                    'P_k1': Phat_k1 + Ptild_k1,
                    'L_k1': L_k1 if meas_update else None}

        return vals_k1
    
    def subArc_iterator(j,input_dict):
        # Unpack unmodified input terms
        A_js = input_dict['A_js']
        B_js = input_dict['B_js']
        H_js = input_dict['H_js']
        P_v_js = input_dict['P_v_js']
        K_arc = input_dict['K_arc']
        G_stoch_arc = input_dict['G_stoch_arc']
        G_exe_arc = input_dict['G_exe_arc']


        Update_val = input_dict['Update_val']

        # Unpack modified input terms
        Phat_js = input_dict['Phat_js']
        Ptild_js = input_dict['Ptild_js']
        Phattild_js = input_dict['Phattild_js']
        Tau_hat_j = input_dict['Tau_hat_j']
        Tau_tild_j = input_dict['Tau_tild_j']
        Gam_hat_j = input_dict['Gam_hat_j']
        Gam_tild_j = input_dict['Gam_tild_j']

        # Grab terms for this subArc
        Phat_j = Phat_js[j,:,:]
        Ptild_j = Ptild_js[j,:,:]
        Phattild_j = Phattild_js[j,:,:]
        A_j = A_js[j,:,:]
        B_j = B_js[j,:,:]
        Phat0_arc = Phat_js[0,:,:]
        P_u_arc = G_stoch_arc + G_exe_arc
        H_j1 = H_js[j,:,:]
        P_v_j1 = P_v_js[j,:,:]


        # Propagate the covariances through time
        tmp_a = A_j @ Tau_hat_j @ K_arc.T @ B_j.T
        Phat_j1m = A_j @ Phat_j @ A_j.T + tmp_a + tmp_a.T + B_j @ K_arc @ Phat0_arc @ K_arc.T @ B_j.T
        tmp_b = A_j @ Gam_tild_j @ B_j.T
        Ptild_j1m = A_j @ Ptild_j @ A_j.T - tmp_b - tmp_b.T + B_j @ P_u_arc @ B_j.T
        Phattild_j1m = A_j @ Phattild_j @ A_j.T - A_j @ Gam_hat_j @ B_j.T + B_j @ K_arc @ Tau_tild_j @ A_j.T
        Tau_hat_j1m = A_j @ Tau_hat_j + B_j @ K_arc @ Phat0_arc
        Tau_tild_j1m = A_j @ Tau_tild_j
        Gam_hat_j1m = A_j @ Gam_hat_j
        Gam_tild_j1m = A_j @ Gam_tild_j - B_j @ P_u_arc

        # Perform the measurement update
        C_j1 = Ptild_j1m @ H_j1.T
        W_j1 = H_j1 @ Ptild_j1m @ H_j1.T + P_v_j1
        L_j1 = Update_val[j] * jax.scipy.linalg.solve(W_j1.T, C_j1.T).T  # L_j1 = C_j1 @ jnp.linalg.inv(W_j1)

        tmp_c = Phattild_j1m @ H_j1.T @ L_j1.T
        tmp_d = L_j1 @ P_v_j1 @ L_j1.T
        Phat_j1 = Phat_j1m - tmp_c - tmp_c.T + L_j1 @ H_j1 @ Ptild_j1m @ H_j1.T @ L_j1.T + tmp_d
        tmp_e = jnp.eye(7) - L_j1 @ H_j1
        Ptild_j1 = tmp_e @ Ptild_j1m @ tmp_e.T + tmp_d
        Phattild_j1 = (Phattild_j1m - L_j1 @ H_j1 @ Ptild_j1m) @ tmp_e.T + tmp_d
        Tau_hat_j1 = Tau_hat_j1m - L_j1 @ H_j1 @ Tau_tild_j1m
        Tau_tild_j1 = tmp_e @ Tau_tild_j1m
        Gam_tild_j1 = tmp_e @ Gam_tild_j1m
        Gam_hat_j1 = Gam_hat_j1m - L_j1 @ H_j1 @ Gam_tild_j1m

        Phat_js = Phat_js.at[j+1,:,:].set(Phat_j1)
        Ptild_js = Ptild_js.at[j+1,:,:].set(Ptild_j1)
        Phattild_js = Phattild_js.at[j+1,:,:].set(Phattild_j1)

        output_dict = {'A_js': A_js, 'B_js': B_js, 'H_js': H_js, 'P_v_js': P_v_js,
                       'K_arc': K_arc, 'G_stoch_arc': G_stoch_arc, 'G_exe_arc': G_exe_arc,
                        'Phat_js': Phat_js, 'Ptild_js': Ptild_js, 'Phattild_js': Phattild_js,
                        'Tau_hat_j': Tau_hat_j1, 'Tau_tild_j': Tau_tild_j1, 
                        'Gam_hat_j': Gam_hat_j1, 'Gam_tild_j': Gam_tild_j1}



    
    cov_propagators = {'perArc': perArc_propagator, 'subArc': perArc_propagator}
    return cov_propagators

def AB_rev2fwd(A_rev, B_rev):
    A_fwd = jnp.linalg.inv(A_rev)
    B_fwd = -A_fwd @ B_rev

    return A_fwd, B_fwd

AB_rev2fwd_vmap = jax.vmap(AB_rev2fwd, in_axes=(0,0))

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

gates2Gexe_vmap = jax.vmap(gates2Gexe, in_axes=(0, None))

def MC_U_tcm_k(X_k_nom, X_k_trial, K_k):

    U_tcm = K_k @ (X_k_trial-X_k_nom)

    return U_tcm

def MC_U_exe(U_nom, gates, rng_key):
    U_exe = jax.random.multivariate_normal(rng_key, jnp.zeros((3,1)).flatten(), gates2Gexe(U_nom, gates))

    return U_exe

def GainParameterizers(Gain_Type):
    ''' This functions returns a dictionary of gain parameterization functions based on the type selected
    '''
    
    FeedbackGainFuncs = {}
    
    if Gain_Type.lower() == 'arc_lqr':
        def xi2K(xi_k, A_k, B_k, U_mag = 1.0):
            Bkr = B_k[:3,:]
            Bkv = B_k[3:6,:]
            
            blkdiagr = xi_k[0]*jnp.linalg.inv(Bkr.T@Bkr)
            blkdiagv = xi_k[1]*jnp.linalg.inv(Bkv.T@Bkv)

            weights = jax.scipy.linalg.block_diag(blkdiagr, blkdiagv)
            K_k = -jnp.linalg.inv(jnp.eye(3) + B_k[:6,:].T @ weights @ B_k[:6,:]) @ B_k[:6,:].T @ weights @ A_k[:6,:6]

            return jnp.hstack([K_k, jnp.zeros((3,1))])

        xi2K_vmap = jax.vmap(xi2K, in_axes=(0, 0, 0, 0))

        FeedbackGainFuncs['xi2K'] = xi2K
        FeedbackGainFuncs['xi2K_vmap'] = xi2K_vmap
    elif Gain_Type.lower() == 'fulltraj_lqr':
        def iterate_K(ii, input_dict):
            # Unpack unchanged input terms
            index_back = input_dict['index_back']
            i = index_back[ii]
            A_arc_hst = input_dict['A_arc_hst']
            B_arc_hst = input_dict['B_arc_hst']
            xi_r = input_dict['xi_r']
            xi_v = input_dict['xi_v']
            R_i = input_dict['R_i']

            # Unpack modified 
            K_arc_hst = input_dict['K_arc_hst']
            S_i1 = input_dict['S_i1']

            # Set temporary variables
            A_i = A_arc_hst[i,:,:]
            B_i = B_arc_hst[i,:,:]

            # Nominal Code (Trying something new)
            B_1i_r = B_arc_hst[i-1,:3,:]
            B_1i_v = B_arc_hst[i-1,3:6,:]

            blk_r = xi_r[i]*jnp.linalg.inv(B_1i_r.T @ B_1i_r)
            blk_v = xi_v[i]*jnp.linalg.inv(B_1i_v.T @ B_1i_v)
            Q_i = jax.scipy.linalg.block_diag(blk_r, blk_v)
            #Q_i = jnp.diag(jnp.array([xi_r[i]*jnp.ones(3,), xi_v[i]*jnp.ones(3,)]).flatten())

            # Iterate and compute K_i and S_i
            #R_i_mod = B_i.T @ B_i
            M_i = R_i + B_i.T @ S_i1 @ B_i
            tmp_mat = jnp.linalg.solve(M_i,B_i.T @ S_i1) # M_i_inv @ B_i.T @ S_i1

            K_i = - tmp_mat @ A_i
            K_arc_hst = K_arc_hst.at[i,:,0:6].set(K_i)

            Q_i_mod = Q_i

            S_i = A_i.T @ (S_i1 - S_i1 @ B_i @ tmp_mat) @ A_i + Q_i_mod 
            
            output_dict = {'index_back': index_back, 'A_arc_hst': A_arc_hst, 'B_arc_hst': B_arc_hst,
                           'xi_r': xi_r, 'xi_v': xi_v, 'K_arc_hst': K_arc_hst,'R_i': R_i, 'S_i1': S_i}

            return output_dict
        
        def xi2K_vmap(xi, A_arc_hst, B_arc_hst, U_mag_arc_hst):
            N_arcs = A_arc_hst.shape[0]
            xi_r = xi[:,0]
            xi_v = xi[:,1]
            
            R_i = jnp.eye(3)

            index_back = jnp.arange(N_arcs-1, -1, -1)

            B_1N_r = B_arc_hst[-1,:3,:]
            B_1N_v = B_arc_hst[-1,3:6,:]

            # Nominal Code (Trying something new)
            blk_r = xi_r[-1]*jnp.linalg.inv(B_1N_r.T @ B_1N_r)
            blk_v = xi_v[-1]*jnp.linalg.inv(B_1N_v.T @ B_1N_v)
            Q_N = jax.scipy.linalg.block_diag(blk_r, blk_v)
            

            #Q_N = jnp.diag(jnp.array([xi_r[-1]*jnp.ones(3,), xi_v[-1]*jnp.ones(3,)]).flatten())
            A_rv_arc_hst = A_arc_hst[:,:6,:6]
            B_rv_arc_hst = B_arc_hst[:,:6,:]

            K_arc_hst = jnp.zeros((N_arcs, 3, 7))

            init_input = {'index_back': index_back, 'A_arc_hst': A_rv_arc_hst, 'B_arc_hst': B_rv_arc_hst,
                          'xi_r': xi_r, 'xi_v': xi_v, 'K_arc_hst': K_arc_hst, 'R_i': R_i, 'S_i1': Q_N}
            
            final_output = jax.lax.fori_loop(0, N_arcs, iterate_K, init_input)
            K_arc_hst = final_output['K_arc_hst']

            return K_arc_hst
        
        FeedbackGainFuncs['xi2K_vmap'] = xi2K_vmap

    return FeedbackGainFuncs

# --------------------
# Estimator Functions
# --------------------

def EKF_time(vals_k, dyn_terms, noise_terms):
    # Unpack the current estimate and error values
    Phat_k = vals_k['Phat_k']
    Ptild_k = vals_k['Ptild_k']

    # Unpack the dynamical values
    A_k = dyn_terms['A_k']
    B_k = dyn_terms['B_k']
    K_k = dyn_terms['K_k']

    # Unpack noise terms
    G_exe = noise_terms['G_exe']
    G_stoch = noise_terms['G_stoch']

    # Propagate the covariance values
    Astar_k = A_k + B_k @ K_k
    Phat_k1 = Astar_k @ Phat_k @ Astar_k.T
    Ptild_k1 = A_k @ Ptild_k @ A_k.T + B_k @ (G_exe + G_stoch) @ B_k.T
    P_k1 = Phat_k1 + Ptild_k1

    priori_vals_k1 = {'Phat_k1': Phat_k1,
                      'Ptild_k1': Ptild_k1,
                      'P_k1': P_k1}

    return priori_vals_k1

def EKF_measurement(priori_vals_k1, meas_terms):
    # Unpack the apriori estimate and error values
    Xhat_k1 = priori_vals_k1['Xhat_k1']
    Phat_k1 = priori_vals_k1['Phat_k1']
    Ptild_k1 = priori_vals_k1['Ptild_k1']

    # Unpack the measurement values
    z_k1 = meas_terms['z_k1']
    z_est_k1 = meas_terms['z_est_k1']
    p_v_est_k1 = meas_terms['P_v_est_k1']
    H_est_k1 = meas_terms['H_est_k1']

    # Compute the Kalman Gain
    C_k = Ptild_k1 @ H_est_k1.T
    W_k = H_est_k1 @ Ptild_k1 @ H_est_k1.T + p_v_est_k1
    L_k = jax.scipy.linalg.solve(W_k.T, C_k.T).T  # L_k = C_k @ jnp.linalg.inv(W_k)

    # Update the mean values
    Xhat_k1_p = Xhat_k1 + L_k @ (z_k1 - z_est_k1)

    # Update the covariance values
    update_term = (jnp.eye(7) - L_k @ H_est_k1)
    Phat_k1_p = Phat_k1 + L_k @ W_k @ L_k.T
    Ptild_k1_p = update_term @ Ptild_k1 @ update_term.T + L_k @ p_v_est_k1 @ L_k.T

    P_k1_p = Phat_k1_p + Ptild_k1_p

    posteriori_vals_k1 = {'Xhat_k1_p': Xhat_k1_p,
                         'Phat_k1_p': Phat_k1_p,
                         'Ptild_k1_p': Ptild_k1_p,
                         'P_k1_p': P_k1_p}

    return posteriori_vals_k1


# -------------------
# Measurement Models
# -------------------

def test_pos_measurement_model(pos_sig):
    # State to be estimated
    r_x, r_y, r_z, v_x, v_y, v_z, m = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z, m', real=True)
    X_sym = sp.Matrix([[r_x],
                    [r_y],
                    [r_z],
                    [v_x],
                    [v_y],
                    [v_z],
                    [m]])

    pos_meas = sp.Matrix([[r_x],
                          [r_y],
                          [r_z]])
    pos_meas_jac = pos_meas.jacobian(X_sym)

    h_eval = sp.lambdify((X_sym,), pos_meas, 'jax')
    h_eval_flat = lambda x: jnp.reshape(h_eval(x), (3,))
    H_eval = sp.lambdify((X_sym,), pos_meas_jac, 'jax')
    P_v_eval = lambda X: jnp.diag(jnp.array([pos_sig**2, pos_sig**2, pos_sig**2]))

    meas_model = {'h_eval': h_eval_flat, 'H_eval': H_eval, 'P_v_eval': P_v_eval}

    return meas_model

def range_measurement_model(r_obs, range_sig):
    # State to be estimated
    r_x, r_y, r_z, v_x, v_y, v_z, m = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z, m', real=True)
    X_sym = sp.Matrix([[r_x],
                    [r_y],
                    [r_z],
                    [v_x],
                    [v_y],
                    [v_z],
                    [m]])

    # Observer position
    obs_x, obs_y, obs_z = r_obs

    dist = sp.sqrt((r_x - obs_x)**2 + (r_y - obs_y)**2 + (r_z - obs_z)**2)
    meas = sp.Matrix([[dist]])
    meas_jac = meas.jacobian(X_sym)

    h_eval = sp.lambdify((X_sym,), meas, 'jax')
    h_eval_flat = lambda x: jnp.reshape(h_eval(x), (1,))
    H_eval = sp.lambdify((X_sym,), meas_jac, 'jax')
    P_v_eval = lambda X: jnp.array([[range_sig**2]])

    meas_model = {'h_eval': h_eval_flat, 'H_eval': H_eval, 'P_v_eval': P_v_eval}

    return meas_model

def range_and_rate_measurement_model(r_obs, range_sig, rate_sig):
    # State to be estimated
    r_x, r_y, r_z, v_x, v_y, v_z, m = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z, m', real=True)
    X_sym = sp.Matrix([[r_x],
                    [r_y],
                    [r_z],
                    [v_x],
                    [v_y],
                    [v_z],
                    [m]])

    # Observer position
    obs_x, obs_y, obs_z = r_obs

    rel_pos = sp.Matrix([[r_x - obs_x],
                         [r_y - obs_y],
                         [r_z - obs_z]])
    rel_vel = sp.Matrix([[v_x],
                         [v_y],
                         [v_z]])
    dist = sp.sqrt(rel_pos.T @ rel_pos)[0]
    range_rate = (rel_pos.T @ rel_vel)[0] / dist
    meas = sp.Matrix([[dist],
                      [range_rate]])
    meas_jac = meas.jacobian(X_sym)

    h_eval = sp.lambdify((X_sym,), meas, 'jax')
    h_eval_flat = lambda x: jnp.reshape(h_eval(x), (2,))
    H_eval = sp.lambdify((X_sym,), meas_jac, 'jax')
    P_v_eval = lambda X: jnp.diag(jnp.array([range_sig**2, rate_sig**2]))

    meas_model = {'h_eval': h_eval_flat, 'H_eval': H_eval, 'P_v_eval': P_v_eval}

    return meas_model

def measurement_model_builder(measurements, params):

    # State to be estimated
    r_x, r_y, r_z, v_x, v_y, v_z, m = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z, m', real=True)
    X_sym = sp.Matrix([[r_x],
                    [r_y],
                    [r_z],
                    [v_x],
                    [v_y],
                    [v_z],
                    [m]])
    
    # Observer position
    obs_x, obs_y, obs_z = params['r_obs']

    # Various vectors needed
    rel_pos = sp.Matrix([[r_x - obs_x],
                         [r_y - obs_y],
                         [r_z - obs_z]])
    
    rel_vel = sp.Matrix([[v_x],
                         [v_y],
                         [v_z]])
   

    # Position Measurement Set Up
    pos = sp.Matrix([[r_x],
                          [r_y],
                          [r_z]])
    pos_cov = sp.diag(params['pos_sig']**2, params['pos_sig']**2, params['pos_sig']**2)

    # Range Measurement Set Up
    rng = sp.sqrt((rel_pos.T @ rel_pos)[0])
    rng_cov = sp.diag(params['range_sig']**2)

    # Range_Rate Measurement Set Up 
    rng_rate = (rel_pos.T @ rel_vel)[0] / rng
    rng_rate_cov = sp.diag(params['rate_sig']**2)

    # Angles Measurement Set Up
    theta = sp.atan2(rel_pos[1], rel_pos[0])
    phi = sp.asin(rel_pos[2] / rng)
    angles = sp.Matrix([[theta],
                        [phi]])
    angles_cov = sp.diag(params['angles_sig']**2, params['angles_sig']**2)

    # Build the measurement model based on selected measurements
    meas_parts = []
    cov_blocks = []
    for meas in measurements: 
        key = meas.lower()
        if key in ("position", "pos"):
            meas_parts.append(pos)
            cov_blocks.append(pos_cov)
        elif key in ("range", "rng"):
            meas_parts.append(sp.Matrix([[rng]]))
            cov_blocks.append(rng_cov)
        elif key in ("range-rate", "rng-rate", "rate"):
            meas_parts.append(sp.Matrix([[rng_rate]]))
            cov_blocks.append(rng_rate_cov)
        elif key in ("angles", "ang"):
            meas_parts.append(angles)
            cov_blocks.append(angles_cov)
        else:
            raise ValueError(f"Measurement type '{meas}' not recognized.")
    
    meas_vec = sp.Matrix.vstack(*meas_parts)
    meas_cov = sp.diag(*cov_blocks)

    n_state = int(X_sym.shape[0])
    n_meas = int(meas_vec.shape[0])

    meas_jac = meas_vec.jacobian(X_sym)

    h_eval_raw = sp.lambdify((X_sym,), meas_vec, 'jax')
    H_eval_raw = sp.lambdify((X_sym,), meas_jac, 'jax')
    P_v_eval_raw = sp.lambdify((X_sym,), meas_cov, 'jax')

    h_eval = lambda x: jnp.reshape(h_eval_raw(x), (n_meas,))
    H_eval = lambda x: jnp.reshape(H_eval_raw(x), (n_meas, n_state))
    P_v_eval = lambda x: jnp.reshape(P_v_eval_raw(x), (n_meas, n_meas))

    meas_model = {'h_eval': h_eval, 'H_eval': H_eval, 'P_v_eval': P_v_eval, 'n_meas': n_meas}

    return meas_model


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
    
def propagator_gen(X0, U, t0, t1, EOM, prop_length, cfg_args):
    """ This function creates a general integrator using diffrax
    """

    # Unpack (JAX-variable) inputs and (JAX-static) arguments
    r_tol, a_tol, N_save, N_subarcs = cfg_args.r_tol, cfg_args.a_tol, cfg_args.N_save, cfg_args.N_subarcs
    N_arc_tot = N_subarcs * (N_save - 1) + 1  # number of total points per arc in saved history

    term = dfx.ODETerm(EOM)
    solver = dfx.Dopri8()

    stepsize_controller = dfx.PIDController(rtol=r_tol, atol=a_tol)
    if prop_length >= 1:
        save_t = dfx.SaveAt(ts=jnp.linspace(t0,t1,prop_length))
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
                            adjoint=dfx.ForwardMode(),
                            saveat=save_t,
                            max_steps=16**5)

    return sol if prop_length >= 1 else sol.ys[-1].flatten()


# --------------------------------
# Iterative Propagation Functions
# --------------------------------

def forward_propagation_iterate(i, input_dict, propagators, models, dyn_args, cfg_args):
    X0_true_f = input_dict['X0_true_f']
    X_hst = input_dict['X_hst']
    U_arc_hst = input_dict['U_arc_hst']   

    t_node_bound = dyn_args['t_node_bound']

    sol_f = propagators['propagator_e'](X0_true_f, U_arc_hst[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args.arc_length_opt, cfg_args)
    X_arc_hst = sol_f.ys[:,:7]
    t_arc_hst = sol_f.ts
    X_hst = X_hst.at[i,:,:].set(X_arc_hst)

    output_dict = {'X_hst': X_hst, 'U_arc_hst': U_arc_hst}
    
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_hst = input_dict['A_hst']
        B_hst = input_dict['B_hst']
        A_arc_hst = input_dict['A_arc_hst']
        B_arc_hst = input_dict['B_arc_hst']
        A_i_arc = propagators['propagator_dX0_e'](X0_true_f, U_arc_hst[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
        B_i_arc = propagators['propagator_dU_e'](X0_true_f, U_arc_hst[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
        A_arc_hst = A_arc_hst.at[i,:,:].set(A_i_arc)
        B_arc_hst = B_arc_hst.at[i,:,:].set(B_i_arc)

        if cfg_args.N_subarcs > 1:
            tmp_A_js = propagators['propagator_dX0_arc_vmap_e'](X_arc_hst[:-1,:], U_arc_hst[i,:], t_arc_hst[:-1], t_arc_hst[1:], cfg_args)
            tmp_B_js = propagators['propagator_dU_arc_vmap_e'](X_arc_hst[:-1,:], U_arc_hst[i,:], t_arc_hst[:-1], t_arc_hst[1:], cfg_args)
        else:
            tmp_A_js = A_i_arc
            tmp_B_js = B_i_arc
        A_hst = A_hst.at[i,:,:,:].set(tmp_A_js)
        B_hst = B_hst.at[i,:,:,:].set(tmp_B_js)

        output_dict['A_arc_hst'] = A_arc_hst
        output_dict['B_arc_hst'] = B_arc_hst
        output_dict['A_hst'] = A_hst
        output_dict['B_hst'] = B_hst
        if cfg_args.feedback_type.lower() == 'estimated_state':
            H_hst = input_dict['H_hst']
            P_v_hst = input_dict['P_v_hst']

            # Measurement Model Evaluation
            H_hst = H_hst.at[i,:,:,:].set(models['measurements']['H_vmap'](X_arc_hst))
            P_v_hst = P_v_hst.at[i,:,:,:].set(models['measurements']['P_v_vmap'](X_arc_hst))

            output_dict['H_hst'] = H_hst
            output_dict['P_v_hst'] = P_v_hst

    X0_true_f = sol_f.ys[-1,:7].flatten()
    output_dict['X0_true_f'] = X0_true_f

    return output_dict

def backward_propagation_iterate(ii,input_dict, propagators, models, dyn_args, cfg_args):
    X0_true_b = input_dict['X0_true_b']
    X_hst = input_dict['X_hst']
    U_arc_hst = input_dict['U_arc_hst']

    t_node_bound = dyn_args['t_node_bound']

    indx_b = dyn_args['indx_b']
    i = indx_b[ii]

    sol_b = propagators['propagator_e'](X0_true_b, U_arc_hst[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args.arc_length_opt, cfg_args)
    X_arc_hst_b = sol_b.ys[:,:7]
    t_arc_hst_b = sol_b.ts
    X_arc_hst_f = jnp.flipud(X_arc_hst_b)
    X_hst = X_hst.at[i,:,:].set(X_arc_hst_f)


    output_dict = {'X_hst': X_hst, 'U_arc_hst': U_arc_hst}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_hst = input_dict['A_hst']
        B_hst = input_dict['B_hst']
        A_arc_hst = input_dict['A_arc_hst']
        B_arc_hst = input_dict['B_arc_hst']
        A_i_arc_b = propagators['propagator_dX0_e'](X0_true_b, U_arc_hst[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
        B_i_arc_b = propagators['propagator_dU_e'](X0_true_b, U_arc_hst[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
        A_i_arc_f, B_i_arc_f = AB_rev2fwd(A_i_arc_b, B_i_arc_b)
        A_arc_hst = A_arc_hst.at[i,:,:].set(A_i_arc_f)
        B_arc_hst = B_arc_hst.at[i,:,:].set(B_i_arc_f)

        if cfg_args.N_subarcs > 1:
            tmp_A_js_b = propagators['propagator_dX0_arc_vmap_e'](X_arc_hst_b[:-1,:], U_arc_hst[i,:], t_arc_hst_b[:-1], t_arc_hst_b[1:], cfg_args)
            tmp_B_js_b = propagators['propagator_dU_arc_vmap_e'](X_arc_hst_b[:-1,:], U_arc_hst[i,:], t_arc_hst_b[:-1], t_arc_hst_b[1:], cfg_args)
            tmp_A_js_f, tmp_B_js_f = AB_rev2fwd_vmap(tmp_A_js_b, tmp_B_js_b)

            tmp_A_js_f = jnp.flipud(tmp_A_js_f)
            tmp_B_js_f = jnp.flipud(tmp_B_js_f)
        else:
            tmp_A_js_f = A_i_arc_f
            tmp_B_js_f = B_i_arc_f
        A_hst = A_hst.at[i,:,:,:].set(A_i_arc_f)#tmp_A_js_f)
        B_hst = B_hst.at[i,:,:,:].set(B_i_arc_f)#tmp_B_js_f)

        output_dict['A_arc_hst'] = A_arc_hst
        output_dict['B_arc_hst'] = B_arc_hst
        output_dict['A_hst'] = A_hst
        output_dict['B_hst'] = B_hst
        if cfg_args.feedback_type.lower() == 'estimated_state':
            H_hst = input_dict['H_hst']
            P_v_hst = input_dict['P_v_hst']

            # Measurement Model Evaluation
            H_hst = H_hst.at[i,:,:,:].set(models['measurements']['H_vmap'](X_arc_hst_f))
            P_v_hst = P_v_hst.at[i,:,:,:].set(models['measurements']['P_v_vmap'](X_arc_hst_f))

            output_dict['H_hst'] = H_hst
            output_dict['P_v_hst'] = P_v_hst

    
    X0_true_b = X_arc_hst_b[-1,:7].flatten()
    output_dict['X0_true_b'] = X0_true_b

    return output_dict

def forward_propagation_cov_iterate(i, input_dict, cov_propagators, dyn_args, cfg_args):
    # Unpack unmodified inputs
    A_hst = input_dict['A_hst']
    B_hst = input_dict['B_hst']
    K_arc_hst = input_dict['K_arc_hst']
    G_exe_arc_hst = input_dict['G_exe_arc_hst']
    # Unpack modified inputs
    P_U_arc_hst = input_dict['P_U_arc_hst']

    
    if cfg_args.feedback_type.lower() == 'true_state':
        # Unpack modified inputs
        P0_arc = input_dict['P0_arc']
        P_hst = input_dict['P_hst']
        P_hst = P_hst.at[i,0,:,:].set(P0_arc)

        # Create input dict for subArc iterator
        cov_input_dict = {'A_js': A_hst[i,:,:,:], 'B_js': B_hst[i,:,:,:], 
                        'K_arc': K_arc_hst[i,:,:], 'G_stoch_arc': dyn_args['G_stoch'], 'G_exe_arc': G_exe_arc_hst[i,:,:],
                        'P_js': P_hst[i,:,:,:]}
        K_arc = K_arc_hst[i,:,:]
        P_U_arc = K_arc @ P0_arc @ K_arc.T
        P_U_arc_hst = P_U_arc_hst.at[i,:,:].set(P_U_arc)

        # Set initial terms for arc
        cov_input_dict['tau_j'] = P0_arc
        cov_input_dict['gam_j'] = jnp.zeros((7,3))

        # Propagate covariance over the arc
        cov_output_dict = jax.lax.fori_loop(0, cfg_args.N_subarcs, cov_propagators['subArc'], cov_input_dict)

        P_js = cov_output_dict['P_js']
        P_hst = P_hst.at[i,:,:,:].set(P_js)
        P0_arc = P_hst[i,-1,:,:]
        output_dict = {'A_hst': A_hst, 'B_hst': B_hst, 'K_arc_hst': K_arc_hst, 'G_exe_arc_hst': G_exe_arc_hst,
                    'P0_arc': P0_arc, 'P_hst': P_hst, 'P_U_arc_hst': P_U_arc_hst}
    elif cfg_args.feedback_type.lower() == 'estimated_state':
        # Unpack unmodified inputs
        H_hst = input_dict['H_hst']
        P_v_hst = input_dict['P_v_hst']
        # Unpack modified inputs
        Phat0_arc = input_dict['Phat0_arc']
        Ptild0_arc = input_dict['Ptild0_arc']
        Phattild0_arc = input_dict['Phattild0_arc']
        Phat_hst = input_dict['Phat_hst']
        Ptild_hst = input_dict['Ptild_hst']
        Phattild_hst = input_dict['Phattild_hst']
        Phat_hst = Phat_hst.at[i,0,:,:].set(Phat0_arc)
        Ptild_hst = Ptild_hst.at[i,0,:,:].set(Ptild0_arc)
        Phattild_hst = Phattild_hst.at[i,0,:,:].set(Phattild0_arc)

        # Redo but for the new sub-arc formulation...
        # Create input dict for subArc iterator
        cov_input_dict = {'A_js': A_hst[i,:,:,:], 'B_js': B_hst[i,:,:,:],
                          'H_js': H_hst[i,:,:,:], 'P_v_js': P_v_hst[i,:,:,:],
                          'K_arc': K_arc_hst[i,:,:], 'G_stoch_arc': dyn_args['G_stoch'], 'G_exe_arc': G_exe_arc_hst[i,:,:],
                          'Phat_js': Phat_hst[i,:,:,:], 'Ptild_js': Ptild_hst[i,:,:,:], 'Phattild_js': Phattild_hst[i,:,:,:]}

        # Pick up here, you are in the middle of adding in the sub-arc propagation with the nav terms. Should 
        # be as simple as finishing this code by calling the  sub-arc propagator and outputing the results to the
        # history arrays. Once you do that return to the objective_constraint function and make sure you call
        # it properly there along with initializing the correct values. Make sure to also create the new Phattild
        # array.
        # Otherwise, be thinking about how to implement the MC code as you'll have to linearize about each estimate
        # and compute the individual trial covariance histories. Probably just make a new EKF function for the
        # sub-arc case.....


        # Set Current terms
        Phat_hst = Phat_hst.at[i,0,:,:].set(Phat0_arc)
        Ptild_hst = Ptild_hst.at[i,0,:,:].set(Ptild0_arc)
        K_arc = K_arc_hst[i,:,:]
        P_U_arc = K_arc @ Phat0_arc @ K_arc.T
        P_U_arc_hst = P_U_arc_hst.at[i,:,:].set(P_U_arc)

        # Propagate covariance over the arc
        vals_k = {'Phat_k': Phat0_arc, 'Ptild_k': Ptild0_arc}
        dyn_terms = {'A_k': A_hst[i,0,:,:], 'B_k': B_hst[i,0,:,:], 'G_stoch': dyn_args['G_stoch']}
        ctrl_terms = {'K_k': K_arc_hst[i,:,:], 'G_exe': G_exe_arc_hst[i,:,:]}
        meas_terms = {'H_k1': H_hst[i,1,:,:], 'P_v': P_v_hst[i,1,:,:]}
        vals_k1 = cov_propagators['perArc'](vals_k, dyn_terms, ctrl_terms, meas_terms, meas_update=True)

        Phat_hst = Phat_hst.at[i,1,:,:].set(vals_k1['Phat_k1'])
        Ptild_hst = Ptild_hst.at[i,1,:,:].set(vals_k1['Ptild_k1'])
        Phat0_arc = vals_k1['Phat_k1']
        Ptild0_arc = vals_k1['Ptild_k1']
        output_dict = {'A_hst': A_hst, 'B_hst': B_hst, 'K_arc_hst': K_arc_hst, 'G_exe_arc_hst': G_exe_arc_hst,
                       'Phat0_arc': Phat0_arc, 'Ptild0_arc': Ptild0_arc,
                       'Phat_hst': Phat_hst, 'Ptild_hst': Ptild_hst, 'P_U_arc_hst': P_U_arc_hst,
                       'H_hst': H_hst, 'P_v_hst': P_v_hst}

    return output_dict


# -----------------------------------
# Constraint and Objective Function
# -----------------------------------

def objective_and_constraints(inputs, Boundary_Conds, iterators, propagators, models, dyn_args, cfg_args):
    output_dict = {}

    # Unpack args
    X_start = Boundary_Conds['X0_interp']
    X_end = Boundary_Conds['Xf_interp']

    N_nodes = cfg_args.N_nodes
    N_arcs = cfg_args.N_arcs
    N_subarcs = cfg_args.N_subarcs    

    # Unpack inputs
    X0 = inputs['X0']
    Xf = inputs['Xf']
    U_arc_hst = inputs['U_arc_hst'].reshape(N_arcs,3)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        if cfg_args.gain_param_type.lower() == 'arc_lqr':
            gain_weights = inputs['gain_weights'].reshape(N_arcs,2)
        elif cfg_args.gain_param_type.lower() == 'fulltraj_lqr':
            gain_weights = inputs['gain_weights'].reshape(N_arcs+1,2)
    if cfg_args.free_phasing:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Forward and backward node indices
    indx_f = dyn_args['indx_f']
    indx_b = dyn_args['indx_b']

    # Initialize histories ------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    X_hst = jnp.zeros((N_arcs, N_subarcs+1, 7))
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_hst = jnp.zeros((N_arcs, N_subarcs, 7, 7))
        B_hst = jnp.zeros((N_arcs, N_subarcs, 7, 3))
        A_arc_hst = jnp.zeros((N_arcs, 7, 7))
        B_arc_hst = jnp.zeros((N_arcs, 7, 3))
        P_hst = jnp.zeros((N_arcs, N_subarcs+1, 7, 7))
        K_arc_hst = jnp.zeros((N_arcs, 3, 7))
        P_U_arc_hst = jnp.zeros((N_arcs, 3, 3))
        if cfg_args.feedback_type.lower() == 'true_state':
            P0 = dyn_args['Phat_0']
        if cfg_args.feedback_type.lower() == 'estimated_state':
            Phat0 = dyn_args['Phat_0']
            Ptild0 = dyn_args['Ptild_0']
            H_hst = jnp.zeros((N_arcs, N_subarcs+1, cfg_args.meas_dim, 7))
            P_v_hst = jnp.zeros((N_arcs, N_subarcs+1, cfg_args.meas_dim, cfg_args.meas_dim))
            Phat_hst = jnp.zeros((N_arcs, N_subarcs+1, 7, 7))
            Ptild_hst = jnp.zeros((N_arcs, N_subarcs+1, 7, 7))

    # Propagate state and first-order jacobians forward (to half) ------------------------
    # ------------------------------------------------------------------------------------
    X0_true_f = X0
    forward_input_dict = {'X0_true_f': X0_true_f, 'X_hst': X_hst, 'U_arc_hst': U_arc_hst}
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        forward_input_dict['A_hst'] = A_hst
        forward_input_dict['B_hst'] = B_hst
        forward_input_dict['A_arc_hst'] = A_arc_hst
        forward_input_dict['B_arc_hst'] = B_arc_hst
        if cfg_args.feedback_type.lower() == 'estimated_state':
            forward_input_dict['H_hst'] = H_hst
            forward_input_dict['P_v_hst'] = P_v_hst
    forward_out = jax.lax.fori_loop(0, len(indx_f), iterators['forward_propagation_iterate_e'], forward_input_dict)

    X_hst = forward_out['X_hst']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_hst = forward_out['A_hst']
        B_hst = forward_out['B_hst']
        A_arc_hst = forward_out['A_arc_hst']
        B_arc_hst = forward_out['B_arc_hst']
        if cfg_args.feedback_type.lower() == 'estimated_state':
            H_hst = forward_out['H_hst']
            P_v_hst = forward_out['P_v_hst']

    # Propagate dynamics backwards (to half) ---------------------------------------------
    # ------------------------------------------------------------------------------------
    X0_true_b = Xf
    backward_input_dict = {'X0_true_b': X0_true_b, 'X_hst': X_hst, 'U_arc_hst': U_arc_hst}
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        backward_input_dict['A_hst'] = A_hst
        backward_input_dict['B_hst'] = B_hst
        backward_input_dict['A_arc_hst'] = A_arc_hst
        backward_input_dict['B_arc_hst'] = B_arc_hst
        if cfg_args.feedback_type.lower() == 'estimated_state':
            backward_input_dict['H_hst'] = H_hst
            backward_input_dict['P_v_hst'] = P_v_hst
    backward_out = jax.lax.fori_loop(0, len(indx_b), iterators['backward_propagation_iterate_e'], backward_input_dict)

    X_hst = backward_out['X_hst']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_hst = backward_out['A_hst']
        B_hst = backward_out['B_hst']
        A_arc_hst = backward_out['A_arc_hst']
        B_arc_hst = backward_out['B_arc_hst']
        if cfg_args.feedback_type.lower() == 'estimated_state':
            H_hst = backward_out['H_hst']
            P_v_hst = backward_out['P_v_hst']

    # Propagate covariance forward through entire trajectory ------------------------------
    # -------------------------------------------------------------------------------------
    eps = 1e-12
    control_norms = jnp.sqrt(U_arc_hst[:, 0]**2 + U_arc_hst[:, 1]**2 + U_arc_hst[:, 2]**2 + eps)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        K_arc_hst = models['feedback_gains']['xi2K_vmap'](gain_weights, A_arc_hst, B_arc_hst, control_norms)
        G_exe_arc_hst = gates2Gexe_vmap(U_arc_hst, dyn_args['gates'])
        if cfg_args.feedback_type.lower() == 'true_state':
            cov_input_dict = {'A_hst': A_hst, 'B_hst': B_hst, 'K_arc_hst': K_arc_hst, 'G_exe_arc_hst': G_exe_arc_hst,
                            'P_hst': P_hst, 'P0_arc': P0, 'P_U_arc_hst': P_U_arc_hst}
        if cfg_args.feedback_type.lower() == 'estimated_state':
            cov_input_dict = {'A_hst': A_hst, 'B_hst': B_hst, 'K_arc_hst': K_arc_hst, 'G_exe_arc_hst': G_exe_arc_hst,
                              'Phat_hst': Phat_hst, 'Ptild_hst': Ptild_hst, 'Phat0_arc': Phat0, 'Ptild0_arc': Ptild0,
                              'H_hst': H_hst, 'P_v_hst': P_v_hst, 'P_U_arc_hst': P_U_arc_hst}
        
        cov_out = jax.lax.fori_loop(0, N_arcs, iterators['cov_propagation_iterate_e'], cov_input_dict)

        if cfg_args.feedback_type.lower() == 'true_state':
            P_hst = cov_out['P_hst']
        if cfg_args.feedback_type.lower() == 'estimated_state':
            Phat_hst = cov_out['Phat_hst']
            Ptild_hst = cov_out['Ptild_hst']
        P_U_arc_hst = cov_out['P_U_arc_hst']

    # Objective and Constraints ouputs ------------------------------------------------
    # ---------------------------------------------------------------------------------
    J_det = jnp.sum(control_norms)

    if cfg_args.det_or_stoch.lower() == 'deterministic':
        output_dict['o_mf'] = J_det # obejective - minimizing sum of control norms
        output_dict['c_Us'] = control_norms.flatten() # constraint - control norm
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        control_max_eig = cfg_args.mx_tcm_bound * jnp.sqrt(mat_lmax_vmap(P_U_arc_hst))

        J_stat = jnp.sum(control_max_eig)
        output_dict['o_mf'] = J_det + J_stat # obejective - minimizing sum of control norms and max eigenvalues of control covariances
        output_dict['c_Us'] = control_norms.flatten() + control_max_eig.flatten() # constraint - stochastic control norm
        Af = propagators['propagator_dX0_e'](Xf,jnp.zeros((3,)), dyn_args['t_node_bound'][-1], 
                                             dyn_args['t_node_bound'][-1]+dyn_args['tf_T'], cfg_args)
        S_XT_targ_inv = dyn_args['S_XT_targ_inv']
        S_Xf_targ_inv = S_XT_targ_inv @ Af
        if cfg_args.feedback_type.lower() == 'true_state':
            P_Xf_full = P_hst[-1,-1,:,:]
        if cfg_args.feedback_type.lower() == 'estimated_state':
            P_Xf_full = Phat_hst[-1,-1,:,:] + Ptild_hst[-1,-1,:,:]
        tmp_P_Xf_con_val = S_Xf_targ_inv @ P_Xf_full @ S_Xf_targ_inv.T - jnp.eye(7)
        output_dict['c_P_Xf'] = jnp.log10(mat_lmax(tmp_P_Xf_con_val)+1) # constraint - final state covariance

    if cfg_args.free_phasing:
        output_dict['c_X0'] = (X0[:7] - jnp.hstack([X_start.evaluate(alpha).flatten(), 1.])).flatten() # constraint - X0
        output_dict['c_Xf'] = (Xf[:6] - X_end.evaluate(beta).flatten()).flatten() # constraint - Xf
    else: 
        output_dict['c_X0'] = (X0[:7] - jnp.hstack([X_start, 1.])).flatten() # constraint - X0
        output_dict['c_Xf'] = (Xf[:6] - X_end.flatten()).flatten() # constraint - Xf

    output_dict['c_X_mp'] = (X_hst[indx_f[-1], -1, :7] - X_hst[indx_b[-1], 0, :7]).flatten() # constraint - state match point
    
    #X_node_hst = jnp.zeros((N_nodes, 7))
    #X_node_hst = X_node_hst.at[0, :].set(X_hst[0, 0, :7])
    #X_node_hst = X_node_hst.at[1:, :].set(X_hst[:, -1, :7])
    X_node_hst = jnp.zeros((N_arcs*(N_subarcs)+1, 7))
    X_node_hst = X_node_hst.at[:-1,:].set(X_hst[:,:-1,:7].reshape(N_arcs*(N_subarcs), 7))
    X_node_hst = X_node_hst.at[-1,:].set(X_hst[-1,-1,:7])
    col_print = 0.0
    if cfg_args.det_col_avoid and not cfg_args.stat_col_avoid:
        #col_vals = col_avoid_vmap(X_node_hst[:-1,:7], dyn_args)
        col_vals = interp_col_avoid_vmap(X_node_hst, dyn_args)
        output_dict['c_det_col_avoid'] = col_vals.flatten() # constraint - deterministic collision avoidance
        col_print = jnp.max(output_dict['c_det_col_avoid'])
    if cfg_args.stat_col_avoid and cfg_args.det_or_stoch.lower() != 'deterministic':
        P_node_hst = jnp.zeros((N_nodes, 7, 7))
        P_node_hst = P_node_hst.at[0, :, :].set(P_hst[0, 0, :, :])
        P_node_hst = P_node_hst.at[1:, :, :].set(P_hst[:, -1, :, :])
        Y_mean, P_y = stat_col_avoid_vmap(X_node_hst, P_node_hst, dyn_args, cfg_args)
        stat_col_vals = Y_mean + cfg_args.mx_col_bound * jnp.sqrt(P_y + eps)
        output_dict['c_stat_col_avoid'] = stat_col_vals.flatten() # constraint - statistical collision avoidance
        col_print = jnp.max(output_dict['c_stat_col_avoid'])
    

    # Print Objective and Constraint Info ---------------------------------------------
    # ---------------------------------------------------------------------------------
    if cfg_args.det_or_stoch.lower() == 'deterministic':
        base_str = "J: {:.3e},    X0: {:.1e},    Xf: {:.0e},    X_mp: {:.0e},    Col: {:.0e}"

        jax.debug.print(base_str, output_dict['o_mf'], 
                        jnp.max(jnp.abs(output_dict['c_X0'])), 
                        jnp.max(jnp.abs(output_dict['c_Xf'])), 
                        jnp.max(jnp.abs(output_dict['c_X_mp'])), 
                        col_print)
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        base_str = "J_d: {:.3e}, J_s: {:.3e},   X0: {:.0e}, Xf: {:.0e}, X_mp: {:.0e},   P_Xf: {:.0e}, Col: {:.0e}, max_xi: {:.1e}"

        jax.debug.print(base_str, J_det, J_stat,
                        jnp.max(jnp.abs(output_dict['c_X0'])),
                        jnp.max(jnp.abs(output_dict['c_Xf'])),
                        jnp.max(jnp.abs(output_dict['c_X_mp'])),
                        jnp.max(output_dict['c_P_Xf']),
                        col_print,
                        jnp.max(gain_weights))

    return output_dict


# ------------------
# Solution Analysis
# ------------------
def sim_Det_traj(sol, Sys, propagators, models, dyn_args, cfg_args):
    # Unpack inputs
    t_node_bound = dyn_args['t_node_bound']
    X0_det = sol.xStar['X0']
    Xf_det = sol.xStar['Xf']
    U_arc_hst = sol.xStar['U_arc_hst'].reshape(cfg_args.N_arcs, 3)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        if cfg_args.gain_param_type.lower() == 'arc_lqr':
            gain_weights = sol.xStar['gain_weights'].reshape(cfg_args.N_arcs, 2)
        elif cfg_args.gain_param_type.lower() == 'fulltraj_lqr':
            gain_weights = sol.xStar['gain_weights'].reshape(cfg_args.N_arcs+1, 2)

    # Unpack propagators
    propagator_e = propagators['propagator_e']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        propagator_dX0_e = propagators['propagator_dX0_e']
        propagator_dU_e = propagators['propagator_dU_e']
        propagator_dX0_arc_vmap_e = propagators['propagator_dX0_arc_vmap_e']
        propagator_dU_arc_vmap_e = propagators['propagator_dU_arc_vmap_e']

        propagator_cov_subArc = propagators['cov_propagators']['subArc']

    # Initialize outputs
    arc_length = cfg_args.arc_length_det  # number of steps per arc in saved history (including bounds)
    length = cfg_args.length  # total number of steps in saved history
    X_hst = jnp.zeros((length, 7))
    X_node_hst = jnp.zeros((cfg_args.N_nodes, 7))
    U_hst = jnp.zeros((length, 3))
    t_hst = jnp.zeros((length,))
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        gain_weights_hst = jnp.zeros((length, 2))
        K_arc_hst = jnp.zeros((cfg_args.N_arcs, 3, 7))
        TCM_norm_bound_hst = jnp.zeros((length,))
        TCM_norm_dV_hst = jnp.zeros((length,))
        U_norm_bound_hst = jnp.zeros((length,))
        U_norm_dV_hst = jnp.zeros((length,))
        A_hst = jnp.zeros((length-1, 7, 7))
        B_hst = jnp.zeros((length-1, 7, 3))
        K_hst = jnp.zeros((length-1, 3, 7))
        A_arc_hst = jnp.zeros((cfg_args.N_arcs, 7, 7))
        B_arc_hst = jnp.zeros((cfg_args.N_arcs, 7, 3))
        P_hst = jnp.zeros((length, 7, 7))
        P_Targ_hst = jnp.zeros((cfg_args.post_insert_length, 7, 7))
        P_u_hst = jnp.zeros((length-1, 3, 3))
        if cfg_args.feedback_type.lower() == 'estimated_state':
            h_hst = jnp.zeros((length, cfg_args.meas_dim))
            H_hst = jnp.zeros((length, cfg_args.meas_dim, 7))
            L_hst = jnp.zeros((length, 7, cfg_args.meas_dim))
            P_v_hst = jnp.zeros((length, cfg_args.meas_dim, cfg_args.meas_dim))
            Phat_hst = jnp.zeros((length, 7, 7))
            Ptild_hst = jnp.zeros((length, 7, 7))
    
    # Initialize first entries
    X_hst = X_hst.at[0,:].set(X0_det)
    X_node_hst = X_node_hst.at[0,:].set(X0_det)
    t_hst = t_hst.at[0].set(t_node_bound[0])
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        if cfg_args.feedback_type.lower() == 'true_state':
            P_hst = P_hst.at[0,:,:].set(dyn_args['Phat_0'])
        elif cfg_args.feedback_type.lower() == 'estimated_state':
            Phat_hst = Phat_hst.at[0,:,:].set(dyn_args['Phat_0'])
            Ptild_hst = Ptild_hst.at[0,:,:].set(dyn_args['Ptild_0'])

    # Propagate state and linearized dynamics across each arc
    for k in range(cfg_args.N_arcs):
        arc_i_0 = k*(arc_length-1)  # starting point index for this arc in saved history
        arc_i_f = (k+1)*(arc_length-1)  # starting point index for next arc in saved history

        # Get the initial conditions for this arc
        X0_arc = X_hst[arc_i_0,:]
        U_arc = U_arc_hst[k,:]            

        # Propagate the state across arc
        sol_f_arc = propagator_e(X0_arc, U_arc, t_node_bound[k], t_node_bound[k+1], cfg_args.arc_length_det, cfg_args)
        X_hst_arc = sol_f_arc.ys 
        t_hst_arc = sol_f_arc.ts
        X_hst = X_hst.at[arc_i_0:arc_i_f+1,:].set(X_hst_arc)
        t_hst = t_hst.at[arc_i_0:arc_i_f+1].set(t_hst_arc)
        U_hst = U_hst.at[arc_i_0:arc_i_f,:].set(jnp.tile(U_arc, (arc_length-1,1)))
        X_node_hst = X_node_hst.at[k+1,:].set(X_hst_arc[-1,:])

        # Compute and store linearized dynamics if stochastic
        if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
            A_arc = propagator_dX0_e(X0_arc, U_arc, t_node_bound[k], t_node_bound[k+1], cfg_args)
            B_arc = propagator_dU_e(X0_arc, U_arc, t_node_bound[k], t_node_bound[k+1], cfg_args)
            A_arc_hst = A_arc_hst.at[k,:,:].set(A_arc)
            B_arc_hst = B_arc_hst.at[k,:,:].set(B_arc)

            tmp_A_js = propagator_dX0_arc_vmap_e(X_hst_arc[:-1,:], U_arc, t_hst_arc[:-1], t_hst_arc[1:], cfg_args)
            tmp_B_js = propagator_dU_arc_vmap_e(X_hst_arc[:-1,:], U_arc, t_hst_arc[:-1], t_hst_arc[1:], cfg_args)
            A_hst = A_hst.at[arc_i_0:arc_i_f,:,:].set(tmp_A_js)
            B_hst = B_hst.at[arc_i_0:arc_i_f,:,:].set(tmp_B_js)

            if cfg_args.feedback_type.lower() == 'estimated_state':
                h_js = models['measurements']['h_vmap'](X_hst_arc)
                H_js = models['measurements']['H_vmap'](X_hst_arc)
                P_v_js = models['measurements']['P_v_vmap'](X_hst_arc)
                h_hst = h_hst.at[arc_i_0:arc_i_f+1,:].set(h_js)
                H_hst = H_hst.at[arc_i_0:arc_i_f+1,:,:].set(H_js)
                P_v_hst = P_v_hst.at[arc_i_0:arc_i_f+1,:,:].set(P_v_js)

    # Propagate covariances if stochastic
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        # Compute feedback gains
        K_arc_hst = models['feedback_gains']['xi2K_vmap'](gain_weights, A_arc_hst, B_arc_hst, jnp.linalg.norm(U_arc_hst, axis=1))
        for k in range(cfg_args.N_arcs):
            arc_i_0 = k*(arc_length-1)  # starting point index for this arc in saved history
            arc_i_f = (k+1)*(arc_length-1)  # starting point index for next arc in saved history\

            #gain_weight_arc = gain_weights[k,:]
            K_arc = K_arc_hst[k,:,:]
            U_arc = U_arc_hst[k,:]
            K_arc_hst = K_arc_hst.at[k,:,:].set(K_arc)
            G_exe_arc = gates2Gexe(U_arc, dyn_args['gates'])
            G_stoch_arc = dyn_args['G_stoch']

            # Propagate control values and tile across arc
            if cfg_args.feedback_type.lower() == 'true_state':
                P0_arc = P_hst[arc_i_0,:,:]
                P_u_arc = K_arc @ P0_arc @ K_arc.T  
            elif cfg_args.feedback_type.lower() == 'estimated_state':
                Phat0_arc = Phat_hst[arc_i_0,:,:]
                Ptild0_arc = Ptild_hst[arc_i_0,:,:]
                P_u_arc = K_arc @ Phat0_arc @ K_arc.T  # only consider estimation error for control covariance

            gain_weights_hst = gain_weights_hst.at[arc_i_0:arc_i_f,:].set(jnp.tile(gain_weights[k,:], (arc_length-1,1)))
            K_hst = K_hst.at[arc_i_0:arc_i_f,:,:].set(jnp.tile(K_arc, (arc_length-1,1,1)))
            P_u_hst = P_u_hst.at[arc_i_0:arc_i_f,:,:].set(jnp.tile(P_u_arc, (arc_length-1,1,1)))

            TCM_norm_bound_arc = cfg_args.mx_tcm_bound*jnp.sqrt(mat_lmax(P_u_arc))
            TCM_dV_arc = cfg_args.mx_dV_bound*jnp.sqrt(mat_lmax(P_u_arc))
            U_norm_bound_arc = jnp.linalg.norm(U_arc) + TCM_norm_bound_arc
            U_norm_dV_arc = jnp.linalg.norm(U_arc) + TCM_dV_arc

            TCM_norm_bound_hst = TCM_norm_bound_hst.at[arc_i_0:arc_i_f].set(jnp.tile(TCM_norm_bound_arc, arc_length-1))
            TCM_norm_dV_hst = TCM_norm_dV_hst.at[arc_i_0:arc_i_f].set(jnp.tile(TCM_dV_arc, arc_length-1))
            U_norm_bound_hst = U_norm_bound_hst.at[arc_i_0:arc_i_f].set(jnp.tile(U_norm_bound_arc, arc_length-1))
            U_norm_dV_hst = U_norm_dV_hst.at[arc_i_0:arc_i_f].set(jnp.tile(U_norm_dV_arc, arc_length-1))


            # Set input dict for arc covariance propagation
            if cfg_args.feedback_type.lower() == 'true_state':
                cov_input_dict = {'A_js': A_hst[arc_i_0:arc_i_f,:,:],
                                    'B_js': B_hst[arc_i_0:arc_i_f,:,:],
                                    'K_arc': K_arc, 'G_stoch_arc': G_stoch_arc, 'G_exe_arc': G_exe_arc,
                                    'P_js': P_hst[arc_i_0:arc_i_f+1,:,:],
                                    'tau_j': P0_arc,
                                    'gam_j': jnp.zeros((7,3))}
                cov_output_dict = jax.lax.fori_loop(0, arc_length, propagator_cov_subArc, cov_input_dict)

                # Set arc covariance outputs
                P_js = cov_output_dict['P_js']
                P_hst = P_hst.at[arc_i_0+1:arc_i_f+1,:,:].set(P_js[1:,:,:])
            elif cfg_args.feedback_type.lower() == 'estimated_state':
                # add in terms for navigational case later
                vals_k = {'Phat_k': Phat0_arc, 'Ptild_k': Ptild0_arc}
                dyn_terms = {'A_k': A_hst[arc_i_0,:,:], 'B_k': B_hst[arc_i_0,:,:], 'G_stoch': G_stoch_arc}
                ctrl_terms = {'K_k': K_arc, 'G_exe': G_exe_arc}
                meas_terms = {'H_k1': H_hst[arc_i_0+1,:,:], 'P_v': P_v_hst[arc_i_0+1,:,:]}
                vals_k1 = propagator_cov_subArc(vals_k, dyn_terms, ctrl_terms, meas_terms, meas_update=True)

                # Set arc covariance outputs
                Phat_hst = Phat_hst.at[arc_i_0+1,:,:].set(vals_k1['Phat_k1'])
                Ptild_hst = Ptild_hst.at[arc_i_0+1,:,:].set(vals_k1['Ptild_k1'])
                L_hst = L_hst.at[arc_i_0+1,:,:].set(vals_k1['L_k1'])


            

    # Post-insertion propagation
    post_insert_i0 = cfg_args.transfer_length_det - 1
    X0_post_insert = X_hst[post_insert_i0,:]
    t0_post_insert = t_hst[post_insert_i0]
    sol_post_insert = propagator_e(X0_post_insert, jnp.zeros((3,)),t0_post_insert, t0_post_insert+dyn_args['tf_T'], cfg_args.post_insert_length, cfg_args)
    X_hst_post_insert = sol_post_insert.ys
    t_hst_post_insert = sol_post_insert.ts
    X_hst = X_hst.at[post_insert_i0+1:,:].set(X_hst_post_insert[1:,:])
    t_hst = t_hst.at[post_insert_i0+1:].set(t_hst_post_insert[1:])

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_hst_post_insert = propagator_dX0_arc_vmap_e(X_hst_post_insert[:-1,:], jnp.zeros((3,)), t_hst_post_insert[:-1], t_hst_post_insert[1:], cfg_args)
        B_hst_post_insert = propagator_dU_arc_vmap_e(X_hst_post_insert[:-1,:], jnp.zeros((3,)), t_hst_post_insert[:-1], t_hst_post_insert[1:], cfg_args)
        A_hst = A_hst.at[post_insert_i0:,:].set(A_hst_post_insert)
        B_hst = B_hst.at[post_insert_i0:,:].set(B_hst_post_insert)

        # Propagate the predicted covariance throught post-insertion arc
        if cfg_args.feedback_type.lower() == 'true_state':
            cov_input_dict = {'A_js': A_hst[post_insert_i0:,:,:],
                            'B_js': B_hst[post_insert_i0:,:,:],
                            'K_arc': jnp.zeros((3,7)), 'G_stoch_arc': jnp.zeros((3,3)), 'G_exe_arc': jnp.zeros((3,3)),
                            'P_js': P_hst[post_insert_i0:,:,:],
                            'tau_j': P_hst[post_insert_i0,:,:],
                            'gam_j': jnp.zeros((7,3))}
            cov_output_dict = jax.lax.fori_loop(0, cfg_args.post_insert_length, propagator_cov_subArc, cov_input_dict)
            P_js_post_insert = cov_output_dict['P_js']
            P_hst = P_hst.at[post_insert_i0+1:,:].set(P_js_post_insert[1:,:,:])
        elif cfg_args.feedback_type.lower() == 'estimated_state':
            A_pst_insert = propagator_dX0_e(X0_post_insert, jnp.zeros((3,)), t0_post_insert, t0_post_insert+dyn_args['tf_T'], cfg_args)
            Phat0_arc = Phat_hst[post_insert_i0,:,:]
            Ptild0_arc = Ptild_hst[post_insert_i0,:,:]
            vals_k = {'Phat_k': Phat0_arc, 'Ptild_k': Ptild0_arc}
            dyn_terms = {'A_k': A_pst_insert, 'B_k': jnp.zeros((7,3)), 'G_stoch': jnp.zeros((3,3))}
            ctrl_terms = {'K_k': jnp.zeros((3,7)), 'G_exe': jnp.zeros((3,3))}
            meas_terms = {}
            vals_k1 = propagator_cov_subArc(vals_k, dyn_terms, ctrl_terms, meas_terms, meas_update=False)
            Phat_hst = Phat_hst.at[post_insert_i0+1,:,:].set(vals_k1['Phat_k1'])
            Ptild_hst = Ptild_hst.at[post_insert_i0+1,:,:].set(vals_k1['Ptild_k1'])

            P_hst = Phat_hst + Ptild_hst

        # Set Final Covaraiance target terms
        P_XT_targ = dyn_args['P_XT_targ']
        Af = propagators['propagator_dX0_e'](Xf_det,jnp.zeros((3,)), t0_post_insert, t0_post_insert+dyn_args['tf_T'], cfg_args)
        Af_inv = jnp.linalg.inv(Af)
        P_Xf_targ = Af_inv @ P_XT_targ @ Af_inv.T

        P_Targ_hst = P_Targ_hst.at[0,:,:].set(P_Xf_targ)

        # Propagate the target covariance through post-insertion arc
        if cfg_args.feedback_type.lower() == 'true_state':
            cov_input_dict = {'A_js': A_hst[post_insert_i0:,:,:],
                            'B_js': B_hst[post_insert_i0:,:,:],
                            'K_arc': jnp.zeros((3,7)), 'G_stoch_arc': jnp.zeros((3,3)), 'G_exe_arc': jnp.zeros((3,3)),
                            'P_js': P_Targ_hst}
            cov_input_dict['tau_j'] = P_Xf_targ
            cov_input_dict['gam_j'] = jnp.zeros((7,3))
            cov_output_dict = jax.lax.fori_loop(0, cfg_args.post_insert_length, propagator_cov_subArc, cov_input_dict)
            P_Targ_hst = cov_output_dict['P_js']
        elif cfg_args.feedback_type.lower() == 'estimated_state':
            P_Targ_hst = P_Targ_hst.at[-1,:,:].set(P_XT_targ)

    U_hst_sph = cart2sph_vmap(U_hst)

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
                   'dV_mean': dV_mean,
                   'length_transfer': cfg_args.N_arcs*(arc_length-1)+1,
                   'length_arc': arc_length}

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
        output_dict['gain_weights_hst'] = gain_weights_hst
        output_dict['K_arc_hst'] = K_arc_hst
        output_dict['P_hst'] = P_hst
        if cfg_args.feedback_type.lower() == 'estimated_state':
            output_dict['Phat_hst'] = Phat_hst
            output_dict['Ptild_hst'] = Ptild_hst
            output_dict['h_hst'] = h_hst
            output_dict['H_hst'] = H_hst
            output_dict['L_hst'] = L_hst
            output_dict['P_v_hst'] = P_v_hst
        output_dict['P_u_hst'] = P_u_hst
        output_dict['P_Xf_targ'] = P_Xf_targ
        output_dict['P_XT_targ'] = P_XT_targ
        output_dict['P_Targ_hst'] = P_Targ_hst

    return output_dict

def single_MC_trial(rng_key, inputs, Sys, dyn_args, cfg_args, propagators, models):
    # Unpack inputs
    t_node_bound = dyn_args['t_node_bound']
    det_X_hst = inputs['det_X_hst']
    det_X_node_hst = inputs['det_X_node_hst']
    det_U_arc_hst = inputs['det_U_arc_hst']
    K_arc_hst = inputs['K_arc_hst']
    dV_mean = inputs['dV_mean']
    if cfg_args.feedback_type.lower() == 'estimated_state':
        P_v_hst = inputs['P_v_hst']
        L_hst = inputs['L_hst']

    # Create rng keys
    if cfg_args.feedback_type.lower() == 'true_state':
        key_len = 1+2*cfg_args.N_arcs
    elif cfg_args.feedback_type.lower() == 'estimated_state':
        key_len = 1+1+1+2*cfg_args.N_arcs + cfg_args.N_arcs*(cfg_args.N_subarcs-1)+1
    keys = jax.random.split(rng_key, key_len)
    if cfg_args.feedback_type.lower() == 'true_state':
        key_X0 = keys[0]
        keys_U_exe = keys[1:1+cfg_args.N_arcs]
        keys_W_dyn = keys[1+cfg_args.N_arcs:1+2*cfg_args.N_arcs]
    elif cfg_args.feedback_type.lower() == 'estimated_state':
        key_X0 = keys[0]
        key_Xhat0 = keys[1]
        key_Xtilde0 = keys[2]
        keys_U_exe = keys[3:3+cfg_args.N_arcs]
        keys_W_dyn = keys[3+cfg_args.N_arcs:3+2*cfg_args.N_arcs]
        keys_meas_nsy = keys[3+2*cfg_args.N_arcs:]

    # Initial state and controls
    X0_det = det_X_node_hst[0,:]
    if cfg_args.feedback_type.lower() == 'true_state':
        X0_trial = jax.random.multivariate_normal(key_X0, X0_det, dyn_args['Phat_0'])
    if cfg_args.feedback_type.lower() == 'estimated_state':
        Xhat0_trial = jax.random.multivariate_normal(key_Xhat0, X0_det, dyn_args['Phat_0'])
        Xtilde0_trial = jax.random.multivariate_normal(key_Xtilde0, jnp.zeros(7,), dyn_args['Ptild_0'])
        X0_trial = Xhat0_trial + Xtilde0_trial


    # Initialize outputs
    arc_length = cfg_args.arc_length_det  # number of steps per arc in saved history
    length = cfg_args.length  # total number of steps in saved history
    X_hst = jnp.zeros((length, 7))
    if cfg_args.feedback_type.lower() == 'estimated_state':
        Xhat_hst = jnp.zeros((length, 7))
        Phat_hst = jnp.zeros((length, 7, 7))
        Ptild_hst = jnp.zeros((length, 7, 7))
    U_hst = jnp.zeros((length, 3))
    t_hst = jnp.zeros((length,))

    # Initialize first entries
    X_hst = X_hst.at[0,:].set(X0_trial)
    if cfg_args.feedback_type.lower() == 'estimated_state':
        Xhat_hst = Xhat_hst.at[0,:].set(Xhat0_trial)
        Phat_hst = Phat_hst.at[0,:,:].set(dyn_args['Phat_0'])
        Ptild_hst = Ptild_hst.at[0,:,:].set(dyn_args['Ptild_0'])
    t_hst = t_hst.at[0].set(t_node_bound[0])
    # Loop through each arc
    for k in range(cfg_args.N_arcs):
        arc_i_0 = k*(arc_length-1)  # starting point index for this arc in saved history
        arc_i_f = (k+1)*(arc_length-1)  # starting point index for next arc in saved history

        # Get the initial conditions for this arc and compute control inputs
        X0_arc_det = det_X_node_hst[k,:]
        X0_arc = X_hst[arc_i_0,:]
        if cfg_args.feedback_type.lower() == 'true_state':
            U_arc_tcm = MC_U_tcm_k(X0_arc_det, X0_arc, K_arc_hst[k,:,:])
        if cfg_args.feedback_type.lower() == 'estimated_state':
            Xhat0_arc = Xhat_hst[arc_i_0,:]
            U_arc_tcm = MC_U_tcm_k(X0_arc_det, Xhat0_arc, K_arc_hst[k,:,:])
            G_exe_arc = gates2Gexe(det_U_arc_hst[k,:], dyn_args['gates'])
        U_arc_det = det_U_arc_hst[k,:]
        U_arc_exe = MC_U_exe(U_arc_det, dyn_args['gates'], keys_U_exe[k])
        
        G_stoch_zero = jnp.all(jnp.isclose(dyn_args['G_stoch'], 0.0, atol = 1e-20))
        U_arc_w = jax.lax.cond(G_stoch_zero,
                             lambda key: jnp.zeros(3,),
                             lambda key: jax.random.multivariate_normal(key, jnp.zeros(3,), dyn_args['G_stoch']),
                             keys_W_dyn[k])
        
        U_arc_cmd = U_arc_det + U_arc_tcm
        U_arc_tot = U_arc_cmd + U_arc_exe + U_arc_w
        
        # Propagate the true state across the arc
        sol_f_arc = propagators['propagator_e'](X0_arc, U_arc_tot, t_node_bound[k], t_node_bound[k+1], cfg_args.arc_length_det, cfg_args)
        X_hst_arc = sol_f_arc.ys
        t_hst_arc = sol_f_arc.ts

        X_hst = X_hst.at[arc_i_0+1:arc_i_f+1,:].set(X_hst_arc[1:,:])
        t_hst = t_hst.at[arc_i_0+1:arc_i_f+1].set(t_hst_arc[1:])
        U_hst = U_hst.at[arc_i_0:arc_i_f,:].set(jnp.tile(U_arc_cmd, (arc_length-1,1)))

        # Propagate the estimated state and associated covariances across the arc
        if cfg_args.feedback_type.lower() == 'estimated_state':
            # Loop through each sub-arc, only updating with measurement at the end of each sub-arc
            for j in range(cfg_args.N_subarcs):
                subarc_i_0 = arc_i_0 + j*(cfg_args.N_save-1)  # starting point index for this sub-arc in saved history
                subarc_i_f = subarc_i_0 + (cfg_args.N_save-1)  # starting point index for next sub-arc in saved history
                Xhat0_subarc = Xhat_hst[subarc_i_0,:]
                Phat0_subarc = Phat_hst[subarc_i_0,:,:]
                Ptild0_subarc = Ptild_hst[subarc_i_0,:,:]

                # EKF Time Propagation
                sol_est_subarc = propagators['propagator_e'](Xhat0_subarc, U_arc_cmd, t_hst[subarc_i_0], t_hst[subarc_i_f], cfg_args.N_save, cfg_args)
                Xhat_hst_subarc = sol_est_subarc.ys
                A_hst_subarc = propagators['propagator_dX0_arc_vmap_e'](Xhat_hst_subarc[:-1,:], U_arc_cmd, t_hst[subarc_i_0:subarc_i_f], t_hst[subarc_i_0+1:subarc_i_f+1], cfg_args)
                B_hst_subarc = propagators['propagator_dU_arc_vmap_e'](Xhat_hst_subarc[:-1,:], U_arc_cmd, t_hst[subarc_i_0:subarc_i_f], t_hst[subarc_i_0+1:subarc_i_f+1], cfg_args)
                
                vals_k = {'Phat_k': Phat0_subarc, 'Ptild_k': Ptild0_subarc}
                dyn_terms = {'A_k': A_hst_subarc[0,:,:], 'B_k': B_hst_subarc[0,:,:], 'K_k':K_arc_hst[k,:,:]}
                noise_terms = {'G_stoch': dyn_args['G_stoch'], 'G_exe': G_exe_arc}

                priori_vals_k1 = EKF_time(vals_k, dyn_terms, noise_terms)

                # EKF Measurement Update
                priori_vals_k1['Xhat_k1'] = Xhat_hst_subarc[-1,:]
                
                z_k1 = jax.random.multivariate_normal(keys_meas_nsy[k*cfg_args.N_subarcs + j], 
                                                      models['measurements']['h_eval'](X_hst[subarc_i_f,:]), 
                                                      models['measurements']['P_v_eval'](X_hst[subarc_i_f,:]))
                z_est_k1 = models['measurements']['h_eval'](Xhat_hst_subarc[-1,:])
                P_v_est_k1 = models['measurements']['P_v_eval'](Xhat_hst_subarc[-1,:])
                H_est_k1 = models['measurements']['H_eval'](Xhat_hst_subarc[-1,:])
                meas_terms = {'z_k1': z_k1, 'z_est_k1': z_est_k1, 'P_v_est_k1': P_v_est_k1, 'H_est_k1': H_est_k1}

                posteriori_vals_k1 = EKF_measurement(priori_vals_k1, meas_terms)
                
                Xhat_hst_subarc = Xhat_hst_subarc.at[-1,:].set(posteriori_vals_k1['Xhat_k1_p'])
                Xhat_hst = Xhat_hst.at[subarc_i_0+1:subarc_i_f+1,:].set(Xhat_hst_subarc[1:,:]) 
                Phat_hst = Phat_hst.at[subarc_i_0+1:subarc_i_f+1,:,:].set(posteriori_vals_k1['Phat_k1_p'])
                Ptild_hst = Ptild_hst.at[subarc_i_0+1:subarc_i_f+1,:,:].set(posteriori_vals_k1['Ptild_k1_p'])
    

    # Post-insertion propagation
    post_insert_i0 = cfg_args.transfer_length_det - 1
    X0_post_insert = X_hst[post_insert_i0,:]
    t0_post_insert = t_hst[post_insert_i0]
    sol_post_insert = propagators['propagator_e'](X0_post_insert, jnp.zeros((3,)),t0_post_insert, t0_post_insert+dyn_args['tf_T'], cfg_args.post_insert_length, cfg_args)
    X_hst_post_insert = sol_post_insert.ys
    t_hst_post_insert = sol_post_insert.ts
    X_hst = X_hst.at[post_insert_i0+1:,:].set(X_hst_post_insert[1:,:])
    t_hst = t_hst.at[post_insert_i0+1:].set(t_hst_post_insert[1:])
    
    if cfg_args.feedback_type.lower() == 'estimated_state':
        Xhat0_hst_post_insert = Xhat_hst[post_insert_i0,:]
        sol_post_insert_est = propagators['propagator_e'](Xhat0_hst_post_insert, jnp.zeros((3,)),t0_post_insert, t0_post_insert+dyn_args['tf_T'], cfg_args.post_insert_length, cfg_args)
        Xhat_hst_post_insert = sol_post_insert_est.ys
        Xhat_hst = Xhat_hst.at[post_insert_i0+1:,:].set(Xhat_hst_post_insert[1:,:])

        A_pst_insert = propagators['propagator_dX0_e'](Xhat0_hst_post_insert, jnp.zeros((3,)), t0_post_insert, t0_post_insert+dyn_args['tf_T'], cfg_args)
        Phat0_arc = Phat_hst[post_insert_i0,:,:]
        Ptild0_arc = Ptild_hst[post_insert_i0,:,:]
        vals_k = {'Phat_k': Phat0_arc, 'Ptild_k': Ptild0_arc}
        dyn_terms = {'A_k': A_pst_insert, 'B_k': jnp.zeros((7,3)), 'G_stoch': jnp.zeros((3,3))}
        ctrl_terms = {'K_k': jnp.zeros((3,7)), 'G_exe': jnp.zeros((3,3))}
        meas_terms = {}
        vals_k1 = propagators['cov_propagators']['subArc'](vals_k, dyn_terms, ctrl_terms, meas_terms, meas_update=False)
        Phat_hst = Phat_hst.at[post_insert_i0+1,:,:].set(vals_k1['Phat_k1'])
        Ptild_hst = Ptild_hst.at[post_insert_i0+1,:,:].set(vals_k1['Ptild_k1'])
    

    U_hst_sph = cart2sph_vmap(U_hst)


    dt = t_hst[1] - t_hst[0]
    dV_trial = jnp.sum(jnp.linalg.norm(U_hst, axis=1)*cfg_args.U_Acc_min_nd*dt / X_hst[:,-1])*Sys['Vs']
    dV_tcm_trial = dV_trial - dV_mean

    if cfg_args.feedback_type.lower() == 'true_state':
        return X_hst, U_hst, U_hst_sph, t_hst, dV_trial, dV_tcm_trial
    elif cfg_args.feedback_type.lower() == 'estimated_state':
        return X_hst, Xhat_hst, Phat_hst, Ptild_hst, U_hst, U_hst_sph, t_hst, dV_trial, dV_tcm_trial

def sim_MC_trajs(inputs, seed, Sys, dyn_args, cfg_args, propagator, models):

    N = cfg_args.N_trials
    keys = jax.random.split(jax.random.PRNGKey(seed), N)

    jax.debug.print("Running {} MC Trials...", N)
    one_trial = lambda key: single_MC_trial(key, inputs, Sys, dyn_args, cfg_args, propagator, models)
    # MC_Batched = jax.jit(jax.vmap(one_trial),backend='cpu')
    One_MC = jax.jit(one_trial,backend='cpu')

    arc_length_det = cfg_args.arc_length_det  # number of steps per arc in saved history
    length = cfg_args.length  # total number of steps in saved history
    X_hsts = jnp.zeros((N,length, 7))
    if cfg_args.feedback_type.lower() == 'estimated_state':
        Xhat_hsts = jnp.zeros((N, length, 7))
        Phat_hsts = jnp.zeros((N, length, 7, 7))
        Ptild_hsts = jnp.zeros((N, length, 7, 7))
    U_hsts = jnp.zeros((N,length, 3))
    U_hsts_sph = jnp.zeros((N,length, 3))
    t_hsts = jnp.zeros((N,length,))
    dVs = jnp.zeros((N,))
    dV_tcms = jnp.zeros((N,))

    MC_N_Loop = 1
    Loop_N = N // MC_N_Loop
    for i in tqdm(range(Loop_N),):
        rng0 = i*MC_N_Loop
        rngf = (i+1)*MC_N_Loop
        keys_loop = keys[rng0:rngf]
        if cfg_args.feedback_type.lower() == 'true_state':
            X_hst_i, U_hst_i, U_hst_sph_i, t_hst_i, dV_i, dV_tcms_i = One_MC(keys_loop[0])
        elif cfg_args.feedback_type.lower() == 'estimated_state':
            X_hst_i, Xhat_hst_i, Phat_hst_i, Ptild_hst_i, U_hst_i, U_hst_sph_i, t_hst_i, dV_i, dV_tcms_i = One_MC(keys_loop[0])

            Xhat_hsts = Xhat_hsts.at[rng0:rngf,:,:].set(Xhat_hst_i)
            Phat_hsts = Phat_hsts.at[rng0:rngf,:,:,:].set(Phat_hst_i)
            Ptild_hsts = Ptild_hsts.at[rng0:rngf,:,:,:].set(Ptild_hst_i)
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
    if cfg_args.feedback_type.lower() == 'estimated_state':
        output_dict['Xhat_hsts'] = np.array(Xhat_hsts)
        output_dict['Phat_hsts'] = np.array(Phat_hsts)
        output_dict['Ptild_hsts'] = np.array(Ptild_hsts)
        output_dict['Phat_mean_hst'] = np.mean(Phat_hsts, axis=0)
        output_dict['Ptild_mean_hst'] = np.mean(Ptild_hsts, axis=0)

    return output_dict
