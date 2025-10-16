import jax
import jax.numpy as jnp
from jax_tqdm import loop_tqdm
import numpy as np
import diffrax as dfx
import sympy as sp

from Lib.math import col_avoid_vmap, mat_sqrt, mat_lmax_vmap, mat_lmax, cart2sph

from joblib import Parallel, delayed
from tqdm import tqdm


# ------------------
# Dynamical Systems
# ------------------

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

# ------------------
# Control Functions
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

# --------------------------------
# Numerical Integration Functions
# --------------------------------

def eoms_gen(t, states, args, dynamics, cfg_args):
    """ This function creates general EOMs wrapped from the basline system EOMS with the stochastic terms if needed
    """
    if cfg_args.det_or_stoch.lower() == 'deterministic' or cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        U = args
        X = states[:7]
        Xdot = dynamics['eom_eval'](t,X,U).reshape(-1)
        
        return Xdot
    """
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        U = args
        X = states[:7]
        
        Xdot = dynamics['eom_eval'](t,X,U).reshape(-1)

        Phi_A = states[7:7+7**2].reshape(7,7)
        Phi_B = states[7+7**2:7+7**2+7*3].reshape(7,3)

        A = dynamics['Aprop_eval'](t, X, U)
        B = dynamics['Bprop_eval'](t, X, U)

        Phi_Adot = A @ Phi_A
        Phi_Bdot = A @ Phi_B + B

        states_dot = jnp.hstack([Xdot.flatten(), Phi_Adot.flatten(), Phi_Bdot.flatten()])
    
    elif cfg_args.det_or_stoch.lower() == 'stochastic_brownian':
        return
    """
    
def propagator_gen(X0, U, t0, t1, EOM, cfg_args, G_stoch=None):
    """ This function creates a general integrator using diffrax
    """

    # Unpack (JAX-variable) inputs and (JAX-static) arguments
    # X0, U, t0, t1 = inputs['X0'], inputs['U'], inputs['t0'], inputs['t1']
    r_tol, a_tol, int_save = cfg_args.r_tol, cfg_args.a_tol, cfg_args.int_save

    if cfg_args.det_or_stoch.lower() == 'deterministic' or cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        eom_args = (U)
    elif cfg_args.det_or_stoch == 'Stochastic_Brownian':
        eom_args = (U, G_stoch)

    term = dfx.ODETerm(EOM)
    solver = dfx.Dopri8()

    stepsize_controller = dfx.PIDController(rtol=r_tol, atol=a_tol)
    save_t = dfx.SaveAt(ts=jnp.linspace(t0,t1,int_save))


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
                            max_steps=16**4)
    
    return sol

def propagator_gen_fin(X0, U, t0, t1, EOM, cfg_args, G_stoch=None):
    """This function creates a general integrator but returns only the final state for sensitivity calculations
    """

    # Unpack (JAX-variable) inputs and (JAX-static) arguments
    # X0, U, t0, t1 = inputs['X0'], inputs['U'], inputs['t0'], inputs['t1']
    r_tol, a_tol, int_save = cfg_args.r_tol, cfg_args.a_tol, cfg_args.int_save

    if cfg_args.det_or_stoch.lower() == 'deterministic' or cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        eom_args = (U)
    elif cfg_args.det_or_stoch == 'Stochastic_Brownian':
        eom_args = (U, G_stoch)

    term = dfx.ODETerm(EOM)
    solver = dfx.Dopri8()

    stepsize_controller = dfx.PIDController(rtol=r_tol, atol=a_tol)

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
                            max_steps=16**4)
    
    return sol.ys[-1].flatten()


# --------------------------------
# Iterative Propagation Functions
# --------------------------------

def forward_propagation_iterate(i, input_dict, propagators, dyn_args, cfg_args):
    X0_true_f = input_dict['X0_true_f']
    states = input_dict['states']
    controls = input_dict['controls']   

    """ Pick up here after Aerospace interview. You were in the process of replacing the propagator
    with just the nodes, not including saves inbetween."""

    t_node_bound = dyn_args['t_node_bound']

    sol_f = propagators['propagator_e'](X0_true_f, controls[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
    states = states.at[i,:,:].set(sol_f.ys[:,:7])
 
    
    output_dict = {'states': states, 'controls': controls}
    
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = input_dict['A_ks']
        B_ks = input_dict['B_ks']
    
        tmp_A = propagators['propagator_dX0_e'](X0_true_f, controls[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
        tmp_B = propagators['propagator_dU_e'](X0_true_f, controls[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
        # tmp_A = sol_f.ys[-1,7:7+7**2].reshape(7,7)
        # tmp_B = sol_f.ys[-1,7+7**2:7+7**2+7*3].reshape(7,3)

        A_ks = A_ks.at[i,:,:].set(tmp_A)
        B_ks = B_ks.at[i,:,:].set(tmp_B)

        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks

        # X0_true_f = jnp.hstack([X0_true_f, jnp.eye(7).flatten(), jnp.zeros(7*3)]) # initial state + Phi_A + Phi_B
    
    X0_true_f = sol_f.ys[-1,:7].flatten() 
    output_dict['X0_true_f'] = X0_true_f

    return output_dict

def backward_propagation_iterate(ii,input_dict, propagators, dyn_args, cfg_args):
    X0_true_b = input_dict['X0_true_b']
    states = input_dict['states']
    controls = input_dict['controls']

    t_node_bound = dyn_args['t_node_bound']

    indx_b = dyn_args['indx_b']
    i = indx_b[ii]

    sol_b = propagators['propagator_e'](X0_true_b, controls[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
    states = states.at[i,:,:].set(jnp.flipud(sol_b.ys))


    output_dict = {'states': states, 'controls': controls}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = input_dict['A_ks']
        B_ks = input_dict['B_ks']

        tmp_A = propagators['propagator_dX0_e'](X0_true_b, controls[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
        tmp_B = propagators['propagator_dU_e'](X0_true_b, controls[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
        # tmp_A = sol_b.ys[-1,7:7+7**2].reshape(7,7)
        # tmp_B = sol_b.ys[-1,7+7**2:7+7**2+7*3].reshape(7,3)

        tmp_A = jnp.linalg.inv(tmp_A)
        tmp_B = -tmp_A @ tmp_B

        A_ks = A_ks.at[i,:,:].set(tmp_A)
        B_ks = B_ks.at[i,:,:].set(tmp_B)

        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks

        # X0_true_b = jnp.hstack([X0_true_b, jnp.eye(7).flatten(), jnp.zeros(7*3)]) # initial state + Phi_A + Phi_B
    
    X0_true_b = sol_b.ys[-1,:7].flatten()
    output_dict['X0_true_b'] = X0_true_b

    return output_dict

def forward_propagation_cov_iterate(i, input_dict, dyn_args, cfg_args):
    # Unpack inputs
    controls = input_dict['controls']
    xis = input_dict['xis']  
    A_ks = input_dict['A_ks']
    B_ks = input_dict['B_ks']  
    K_ks = input_dict['K_ks']
    P_ks = input_dict['P_ks']
    P_Us = input_dict['P_Us']
    
    A_k = A_ks[i,:,:]
    B_k = B_ks[i,:,:]
    xi_k = xis[i,:]
    
    # Gates execution error covariance
    gates = dyn_args['gates']
    G_exe = gates2Gexe(controls[i,:], gates)

    # Stochastic Gauss ZOH acceleration covariance
    G_stoch = dyn_args['G_stoch']

    # Compute colsed loop gain
    K_k = xi2K(xi_k, A_k, B_k)

    K_ks = K_ks.at[i,:,:].set(K_k)

    mod_A = A_k + B_k @ K_k

    P_ki = P_ks[i,:,:]
    P_ki1 = mod_A @ P_ki @ mod_A.T + B_k @ (G_exe + G_stoch) @ B_k.T

    P_ks = P_ks.at[i+1,:,:].set(P_ki1)

    P_Ui = K_k @ P_ki @ K_k.T
    P_Us = P_Us.at[i,:,:].set(P_Ui)

    output_dict = {'controls': controls, 'xis': xis, 'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'P_ks': P_ks, 'P_Us': P_Us} 
    return output_dict


# -----------------------------------
# Constraint and Objective Functions
# -----------------------------------

def objective_and_constraints(inputs, Boundary_Conds, iterators, dyn_args, cfg_args):
    output_dict = {}

    # Unpack args
    X_start = Boundary_Conds['X0_interp']
    X_end = Boundary_Conds['Xf_interp']

    nodes = cfg_args.nodes    

    # Unpack inputs
    X0 = inputs['X0']
    Xf = inputs['Xf']
    controls = inputs['controls'].reshape(nodes,3)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        xis = inputs['xis'].reshape(nodes,2)
    if cfg_args.free_phasing:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Forward and backward node indices
    indx_f = dyn_args['indx_f']
    indx_b = dyn_args['indx_b']

    # Initialize histories
    states = jnp.zeros((nodes,cfg_args.int_save, 7))
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = jnp.zeros((nodes, 7, 7))
        B_ks = jnp.zeros((nodes, 7, 3))
        K_ks = jnp.zeros((nodes, 3, 7))
        P_ks = jnp.zeros((nodes+1, 7, 7))
        P_ks = P_ks.at[0,:,:].set(dyn_args['init_cov'])
        P_Us = jnp.zeros((nodes, 3, 3))

    # Propagate dynamics forward (to half)
    # if cfg_args.det_or_stoch.lower() == 'deterministic':
    X0_true_f = X0
    # elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
    #     X0_true_f = jnp.hstack([X0, jnp.eye(7).flatten(), jnp.zeros(7*3)]) # initial state + Phi_A + Phi_B
    forward_input_dict = {'X0_true_f': X0_true_f, 'states': states, 'controls': controls}
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        forward_input_dict['A_ks'] = A_ks
        forward_input_dict['B_ks'] = B_ks
    forward_out = jax.lax.fori_loop(0, len(indx_f), iterators['forward_propagation_iterate_e'], forward_input_dict)

    states = forward_out['states']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = forward_out['A_ks']
        B_ks = forward_out['B_ks']

    # Propagae dynamics dynamics backwards (to half)
    # if cfg_args.det_or_stoch.lower() == 'deterministic':
    X0_true_b = Xf
    # elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
    #     X0_true_b = jnp.hstack([Xf, jnp.eye(7).flatten(), jnp.zeros(7*3)]) # initial state + Phi_A + Phi_B
    backward_input_dict = {'X0_true_b': X0_true_b, 'states': states, 'controls': controls}
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        backward_input_dict['A_ks'] = A_ks
        backward_input_dict['B_ks'] = B_ks
    backward_out = jax.lax.fori_loop(0, len(indx_b), iterators['backward_propagation_iterate_e'], backward_input_dict)

    states = backward_out['states']
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = backward_out['A_ks']
        B_ks = backward_out['B_ks']

    # jax.debug.print("B_ks[0,:,:]: {}", B_ks[0,:,:])
    # jax.debug.print("B_ks[-1,:,:]: {}", B_ks[-1,:,:])

    # Propagate covariance forward through entire trajectory
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        cov_input_dict = {'controls': controls, 'xis': xis, 'A_ks': A_ks, 'B_ks': B_ks, 'K_ks': K_ks, 'P_ks': P_ks, 'P_Us': P_Us}
        cov_out = jax.lax.fori_loop(0, nodes, iterators['cov_propagation_iterate_e'], cov_input_dict)

        K_ks = cov_out['K_ks']
        P_ks = cov_out['P_ks']
        P_Us = cov_out['P_Us']

        # jax.debug.print("P_ks[0,:,:]: {}", P_ks[0,:,:])
        # jax.debug.print("P_ks[-1,:,:]: {}", P_ks[-1,:,:])

    # Objective and Constraints ouputs
    eps = 1e-12
    control_norms = jnp.sqrt(controls[:, 0]**2 + controls[:, 1]**2 + controls[:, 2]**2 + eps)
    J_det = jnp.sum(control_norms)
    if cfg_args.det_or_stoch.lower() == 'deterministic':
        output_dict['o_mf'] = J_det # obejective - minimizing sum of control norms
        output_dict['c_Us'] = control_norms # constraint - control norm
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        control_max_eig = cfg_args.mx * jnp.sqrt(mat_lmax_vmap(P_Us))

        J_stat = jnp.sum(control_max_eig)
        output_dict['o_mf'] = J_det + J_stat # obejective - minimizing sum of control norms and max eigenvalues of control covariances
        output_dict['c_Us'] = control_norms + control_max_eig # constraint - stochastic control norm

        P_Xf = dyn_args['targ_cov_inv_sqrt']@P_ks[-1,:,:]@dyn_args['targ_cov_inv_sqrt'].T - jnp.eye(7)
        output_dict['c_P_Xf'] = jnp.log10(mat_lmax(P_Xf)+1) # constraint - final state covariance

        # jax.debug.print("P_Xf: {}", P_ks[-1,:,:])
        # jax.debug.print("Targ_inv_sqrt: {}", dyn_args['targ_cov_inv_sqrt'])
        # jax.debug.print("Max eig P_Xf: {}", mat_lmax(P_Xf))
        # jax.debug.print("c_P_Xf: {}", output_dict['c_P_Xf'])

    if cfg_args.free_phasing:
        output_dict['c_X0'] = X0[:7] - jnp.hstack([X_start.evaluate(alpha).flatten(), 1.]) # constraint - X0
        output_dict['c_Xf'] = Xf[:6] - X_end.evaluate(beta).flatten() # constraint - Xf
    else: 
        output_dict['c_X0'] = X0[:7] - jnp.hstack([X_start, 1.]) # constraint - X0
        output_dict['c_Xf'] = Xf[:6] - X_end.flatten() # constraint - Xf

    output_dict['c_X_mp'] = states[indx_f[-1], -1, :7] - states[indx_b[-1], 0, :7] # constraint - state match point    
    
    node_states = jnp.zeros((nodes+1, 7))
    node_states = node_states.at[0, :].set(states[0, 0, :7])
    node_states = node_states.at[1:, :].set(states[:, -1, :7])
    if cfg_args.det_col_avoid:
        r_obs = dyn_args['r_obs']
        d_safe = dyn_args['d_safe'] 
        col_vals = col_avoid_vmap(node_states, r_obs, d_safe)
        output_dict['c_det_col_avoid'] = col_vals # constraint - deterministic collision avoidance
        
        
    # still need to add stochastic col avoidance
    
    if cfg_args.det_or_stoch.lower() == 'deterministic':
        base_str = "J: {:.2e},    X0: {:.2e},    Xf: {:.2e},    X_mp: {:.2e},    Col: {:.2e}"

        jax.debug.print(base_str, output_dict['o_mf'].astype(float), 
                        jnp.max(jnp.abs(output_dict['c_X0'])).astype(float), 
                        jnp.max(jnp.abs(output_dict['c_Xf'])).astype(float), 
                        jnp.max(jnp.abs(output_dict['c_X_mp'])).astype(float), 
                        jnp.max(output_dict['c_det_col_avoid']).astype(float))
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        base_str = "J_det: {:.2e}, J_stat: {:.2e}, X0: {:.2e}, Xf: {:.2e}, X_mp: {:.2e}, P_Xf: {:.2e}, ccol: {:.2e}, max_xi: {:.2e}"

        jax.debug.print(base_str, J_det.astype(float), J_stat.astype(float),
                        jnp.max(jnp.abs(output_dict['c_X0'])).astype(float), 
                        jnp.max(jnp.abs(output_dict['c_Xf'])).astype(float), 
                        jnp.max(jnp.abs(output_dict['c_X_mp'])).astype(float), 
                        jnp.max(output_dict['c_P_Xf']).astype(float),
                        jnp.max(output_dict['c_det_col_avoid']).astype(float),
                        jnp.max(xis).astype(float))

    return output_dict


# ------------------
# Solution Analysis
# ------------------
def sim_Det_traj(sol, Sys, propagators, dyn_args, cfg_args):
    # Unpack inputs
    t_node_bound = dyn_args['t_node_bound']
    X0_det = sol.xStar['X0']
    U_node_hst = sol.xStar['controls'].reshape(cfg_args.nodes, 3)
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        xis = sol.xStar['xis'].reshape(cfg_args.nodes, 2)

    # Initialize outputs
    length = cfg_args.nodes*(cfg_args.int_save-1)+1
    X_hst = jnp.zeros((length, 7))
    X_node_hst = jnp.zeros((cfg_args.nodes+1, 7))
    U_hst = jnp.zeros((length, 3))
    t_hst = jnp.zeros((length,))
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = jnp.zeros((cfg_args.nodes, 7, 7))
        B_ks = jnp.zeros((cfg_args.nodes, 7, 3))
        K_ks = jnp.zeros((cfg_args.nodes, 3, 7))
        P_ks = jnp.zeros((cfg_args.nodes+1, 7, 7))

    X_hst = X_hst.at[0,:].set(X0_det)
    X_node_hst = X_node_hst.at[0,:].set(X0_det)
    t_hst = t_hst.at[0].set(t_node_bound[0])

    ptr = 0
    for k in range(cfg_args.nodes):
        X_k = X_hst[ptr,:]
        U_k = U_node_hst[k,:]
        sol_f = propagators['propagator_e'](X_k, U_k, t_node_bound[k], t_node_bound[k+1], cfg_args)

        X_hst = X_hst.at[ptr+1:ptr+cfg_args.int_save,:].set(sol_f.ys[1:,:])
        X_node_hst = X_node_hst.at[k+1,:].set(sol_f.ys[-1,:])
        U_hst = U_hst.at[ptr:ptr+cfg_args.int_save-1,:].set(jnp.tile(U_k, (cfg_args.int_save-1,1)))
        t_hst = t_hst.at[ptr+1:ptr+cfg_args.int_save].set(sol_f.ts[1:])

        if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
            A_k = propagators['propagator_dX0_e'](X_k, U_k, t_node_bound[k], t_node_bound[k+1], cfg_args)
            B_k = propagators['propagator_dU_e'](X_k, U_k, t_node_bound[k], t_node_bound[k+1], cfg_args)

            A_ks = A_ks.at[k,:,:].set(A_k)
            B_ks = B_ks.at[k,:,:].set(B_k)

            xi_k = xis[k,:]
            K_k = xi2K(xi_k, A_k, B_k)
            K_ks = K_ks.at[k,:,:].set(K_k)

            if k == 0:
                P_ks = P_ks.at[0,:,:].set(dyn_args['init_cov'])

            mod_A = A_k + B_k @ K_k
            P_k = P_ks[k,:,:]
            P_k1 = mod_A @ P_k @ mod_A.T + B_k @ (dyn_args['G_stoch'] + gates2Gexe(U_k, dyn_args['gates'])) @ B_k.T
            P_ks = P_ks.at[k+1,:,:].set(P_k1)

        ptr += cfg_args.int_save - 1

    U_hst_sph = cart2sph(U_hst)

    dt = t_hst[1] - t_hst[0]
    dV = jnp.sum(jnp.linalg.norm(U_hst, axis=1)*cfg_args.T_max_nd*dt / X_hst[:,-1])*Sys['Vs']

    output_dict = {'X_hst': X_hst, 'X_node_hst': X_node_hst, 'U_hst': U_hst, 'U_node_hst': U_node_hst, 't_hst': t_hst, 't_node_hst': t_node_bound, 'U_hst_sph': U_hst_sph, 'dV': dV}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks
        output_dict['K_ks'] = K_ks
        output_dict['P_ks'] = P_ks

    return output_dict

def single_MC_trial(rng_key, inputs, dyn_args, cfg_args, propagator):
    # Unpack inputs
    t_node_bound = dyn_args['t_node_bound']
    det_X_node_hst = inputs['det_X_node_hst']
    det_U_node_hst = inputs['det_U_node_hst']
    K_ks = inputs['K_ks']
    P_k0 = dyn_args['init_cov']

    # Create rng keys
    keys = jax.random.split(rng_key, 1+cfg_args.nodes)
    key_X0, keys_U_exe = keys[0], keys[1:]

    # Initial state and controls
    X0_det = det_X_node_hst[0,:]
    X0_trial = jax.random.multivariate_normal(key_X0, X0_det, P_k0)

    # Initialize outputs
    length = cfg_args.nodes*(cfg_args.int_save-1)+1
    X_hst = jnp.zeros((length, 7))
    U_hst = jnp.zeros((length, 3))
    t_hst = jnp.zeros((length,))

    X_hst = X_hst.at[0,:].set(X0_trial)
    t_hst = t_hst.at[0].set(t_node_bound[0])

    ptr = 0
    for k in range(cfg_args.nodes):
        X_k_det = det_X_node_hst[k,:]
        X_k = X_hst[ptr,:]

        U_k_det = det_U_node_hst[k,:]
        U_k_tcm = MC_U_tcm_k(X_k_det, X_k, K_ks[k,:,:])
        U_k_exe = MC_U_exe(U_k_det, dyn_args['gates'], keys_U_exe[k])
        U_k = U_k_det + U_k_tcm + U_k_exe

        sol_f = propagator(X_k, U_k, t_node_bound[k], t_node_bound[k+1], cfg_args)

        X_hst = X_hst.at[ptr+1:ptr+cfg_args.int_save,:].set(sol_f.ys[1:,:])
        U_hst = U_hst.at[ptr:ptr+cfg_args.int_save-1,:].set(jnp.tile(U_k, (cfg_args.int_save-1,1)))
        t_hst = t_hst.at[ptr+1:ptr+cfg_args.int_save].set(sol_f.ts[1:])

        ptr += cfg_args.int_save - 1
    
    U_hst_sph = cart2sph(U_hst)

    return X_hst, U_hst, U_hst_sph, t_hst

def MC_worker_par(id, inputs, seed, dyn_args, cfg_args, propagator):
    rng_key = jax.random.PRNGKey(seed + id)

    X_hst, U_hst, U_hst_sph, t_hst = single_MC_trial(rng_key, inputs, dyn_args, cfg_args, propagator)

    return np.array(X_hst), np.array(U_hst), np.array(U_hst_sph), np.array(t_hst)

def sim_MC_trajs(inputs, seed, dyn_args, cfg_args, propagator, n_jobs = 8):
    N = cfg_args.N_trials
    keys = jax.random.split(jax.random.PRNGKey(seed), N)

    jax.debug.print("Running {} MC Trials...", N)
    MC_Batched = jax.vmap(single_MC_trial, in_axes=(0,None,None,None,None))
    X_hsts, U_hsts, U_hsts_sph, t_hsts = MC_Batched(keys, inputs, dyn_args, cfg_args, propagator)
            
    output_dict = {'X_hsts': np.array(X_hsts), 
                   't_hsts': np.array(t_hsts), 
                   'U_hsts': np.array(U_hsts), 
                   'U_hsts_sph': np.array(U_hsts_sph)}

    return output_dict
