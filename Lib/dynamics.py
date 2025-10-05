import jax
import jax.numpy as jnp
import numpy as np
import diffrax as dfx
import sympy as sp

from Lib.math import col_avoid_vmap


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
    elif cfg_args.det_or_stoch.lower() == 'stochastic_brownian':
        return
    
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
    save_t = jnp.linspace(t0,t1,int_save)


    sol = dfx.diffeqsolve(term,
                            solver, 
                            t0, 
                            t1, 
                            None, 
                            X0, 
                            args=U,
                            stepsize_controller=stepsize_controller, 
                            adjoint=dfx.RecursiveCheckpointAdjoint(),
                            saveat=dfx.SaveAt(ts=save_t),
                            max_steps=16**3)
    
    return sol


# -------------------------------
# (Looped) Numerical Integration
# -------------------------------

def forward_propagation_iterate(i, input_dict, propagator, dyn_args, cfg_args, propagator_dX0=None, propagator_dU=None):
    X0_true_f = input_dict['X0_true_f']
    states = input_dict['states']
    controls = input_dict['controls']
    times = input_dict['times']

    t_node_bound = dyn_args['t_node_bound']

    prop_inputs = {'X0': X0_true_f, 
                   'U': controls[i,:], 
                   't0': t_node_bound[i],
                   't1': t_node_bound[i+1]}
    sol_f = propagator(X0_true_f, controls[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
    states = states.at[i,:,:].set(sol_f.ys)
    times = times.at[i,:].set(sol_f.ts)
    X0_true_f = sol_f.ys[-1,:].flatten()
    
    output_dict = {'X0_true_f': X0_true_f, 'states': states, 'controls': controls, 'times': times}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = input_dict['A_ks']
        B_ks = input_dict['B_ks']

        tmp_A = propagator_dX0(X0_true_f, controls[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)
        tmp_B = propagator_dU(X0_true_f, controls[i,:], t_node_bound[i], t_node_bound[i+1], cfg_args)

        A_ks = A_ks.at[i,:,:].set(tmp_A)
        B_ks = B_ks.at[i,:,:].set(tmp_B)

        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks


    return output_dict

def backward_propagation_iterate(ii,input_dict, propagator, dyn_args, cfg_args, propagator_dX0=None, propagator_dU=None):
    X0_true_b = input_dict['X0_true_b']
    states = input_dict['states']
    controls = input_dict['controls']
    times = input_dict['times']

    t_node_bound = dyn_args['t_node_bound']

    indx_b = dyn_args['indx_b']
    i = indx_b[ii]

    prop_inputs = {'X0': X0_true_b,
                   'U': controls[i,:],
                   't0': t_node_bound[i+1],
                   't1': t_node_bound[i]}
    sol_b = propagator(X0_true_b, controls[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
    states = states.at[i,:,:].set(jnp.flipud(sol_b.ys))
    times = times.at[i,:].set(jnp.flipud(sol_b.ts))
    X0_true_b = sol_b.ys[-1,:].flatten()

    output_dict = {'X0_true_b': X0_true_b, 'states': states, 'controls': controls, 'times': times}

    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        A_ks = input_dict['A_ks']
        B_ks = input_dict['B_ks']

        tmp_A = propagator_dX0(X0_true_b, controls[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)
        tmp_B = propagator_dU(X0_true_b, controls[i,:], t_node_bound[i+1], t_node_bound[i], cfg_args)

        tmp_A = jnp.linalg.inv(tmp_A)
        tmp_B = -tmp_A @ tmp_B

        A_ks = A_ks.at[i,:,:].set(tmp_A)
        B_ks = B_ks.at[i,:,:].set(tmp_B)

        output_dict['A_ks'] = A_ks
        output_dict['B_ks'] = B_ks
    
    return output_dict


# -----------------------------------
# Constraint and Objective Functions
# -----------------------------------

def objective_and_constraints(inputs, Boundary_Conds, for_prop_iter, back_prop_iter, dyn_args, cfg_args):
    output_dict = {}

    # Unpack args
    t_node_bound = dyn_args['t_node_bound']
    X_start = Boundary_Conds['X0_interp']
    X_end = Boundary_Conds['Xf_interp']

    nodes = cfg_args.nodes
    int_save = cfg_args.int_save

    r_obs = dyn_args['r_obs']
    d_safe = dyn_args['d_safe']

    free_phasing = cfg_args.free_phasing

    # Unpack inputs
    X0 = inputs['X0']
    Xf = inputs['Xf']
    controls = inputs['controls'].reshape(nodes,3)

    if free_phasing:
        alpha = inputs['alpha']
        beta = inputs['beta']

    # Forward and backward node indices
    indx_f = dyn_args['indx_f']
    indx_b = dyn_args['indx_b']

    # Initialize histories
    states = jnp.zeros((nodes, int_save, 7))
    times = jnp.zeros((nodes, int_save))

    # Propagate dynamics forward (to half)
    X0_true_f = X0
    forward_input_dict = {'X0_true_f': X0_true_f, 'states': states, 'controls': controls, 'times': times}
    forward_out = jax.lax.fori_loop(0, len(indx_f), for_prop_iter, forward_input_dict)

    states = forward_out['states']
    times = forward_out['times']

    # Propagae dynamics dynamics backwards (to half)
    X0_true_b = Xf
    backward_input_dict = {'X0_true_b': X0_true_b, 'states': states, 'controls': controls, 'times': times}
    backward_out = jax.lax.fori_loop(0, len(indx_b), back_prop_iter, backward_input_dict)

    states = backward_out['states']
    times = backward_out['times']

    # Objective and Constraint arrays
    node_states = jnp.zeros((nodes+1, 7))
    node_states = node_states.at[0, :].set(states[0, 0, :7])
    node_states = node_states.at[1:, :].set(states[:, -1, :7])
    
    col_vals = col_avoid_vmap(node_states, r_obs, d_safe)

    control_norms = jnp.sqrt(controls[:, 0]**2 + controls[:, 1]**2 + controls[:, 2]**2 + 1e-12)


    # Objective and Constraints ouputs
    output_dict['o_mf'] = jnp.sum(control_norms) # obejective - minimizing sum of control norms

    output_dict['c_Us'] = control_norms # constraint - control norm
    output_dict['c_X_mp'] = states[indx_f[-1], -1, :7] - states[indx_b[-1], 0, :7] # constraint - state match point
    output_dict['c_det_col_avoid'] = col_vals # constraint - deterministic collision avoidance
    
    if free_phasing:
        output_dict['c_X0'] = X0[:7] - jnp.hstack([X_start.evaluate(alpha).flatten(), 1.]) # constraint - X0
        output_dict['c_Xf'] = Xf[:6] - X_end.evaluate(beta).flatten() # constraint - Xf
    else: 
        output_dict['c_X0'] = X0[:7] - jnp.hstack([X_start, 1.]) # constraint - X0
        output_dict['c_Xf'] = Xf[:6] - X_end.flatten() # constraint - Xf
    
    base_str = "J: {:.2e},    X0: {:.2e},    Xf: {:.2e},    X_mp: {:.2e},    Col: {:.2e}"

    jax.debug.print(base_str, output_dict['o_mf'].astype(float), 
                    jnp.max(jnp.abs(output_dict['c_X0'])).astype(float), 
                    jnp.max(jnp.abs(output_dict['c_Xf'])).astype(float), 
                    jnp.max(jnp.abs(output_dict['c_X_mp'])).astype(float), 
                    jnp.max(output_dict['c_det_col_avoid']).astype(float))

    return output_dict
