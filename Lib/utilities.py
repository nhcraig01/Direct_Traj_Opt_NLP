import yaml
import numpy as np

import jax
from jax import numpy as jnp
import diffrax as dfx

from Lib.math import calc_t_elapsed_nd, sig2cov
from Lib.dynamics import CR3BPDynamics, TrueStateFeedback_CovPropagators, EstimatedStateFeedback_CovPropagators, GainParameterizers, range_measurement_model, test_pos_measurement_model, forward_propagation_iterate, backward_propagation_iterate, objective_and_constraints, forward_propagation_cov_iterate, sim_MC_trajs, sim_Det_traj

from scipy import stats

from dataclasses import dataclass, replace

import h5py

from astropy.coordinates import spherical_to_cartesian as sph_to_cart

from scipy.io import savemat

#----
# IO
#----

def yaml_load(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

def yaml_save(config, filename):
    with open(filename, 'w') as file:
        yaml.safe_dump(config, file)
    return

def load_family(input_loc: str) -> dict:
    family_data = {}
    with h5py.File(input_loc, "r") as f:
        family_data["X_hst"]   = f["X_hst"][:]     # (6 x n x N) array
        family_data["t_hst"]   = f["t_hst"][:]     # (1 x n x N) array
        family_data["JCs"]     = f["JCs"][:]       # (1 x N) array
        family_data["STMs"]    = f["STMs"][:]      # (6 x 6 x N) array
        family_data["BrkVals"] = f["BrkVals"][:]   # (whatever shape you saved)

    return family_data


# -----
# Misc
# -----

def process_sparsity(grad_nonsparse):
    grad_sparse = {}
    
    for key, val in grad_nonsparse.items():
        cur_obj_constr = val
        new_obj_constr = {}
        for key2, val2_jax in cur_obj_constr.items():
            val2 = np.array(val2_jax)
            
            if len(val2.shape) != 2:
                if key == 'c_P_Xf':
                    new_obj_constr[key2] = val2.reshape(1,-1)
                else:
                    new_obj_constr[key2] = val2.reshape(-1,1)
            else:
                new_obj_constr[key2] = val2
            
            if jnp.all(new_obj_constr[key2] == 0):
                new_obj_constr.pop(key2, None)
        
        grad_sparse[key] = new_obj_constr
    
    return grad_sparse

def process_config(config, det_or_stoch: str, feedback_control_type: str, gain_parametrization_type: str):
    """ This function processes the configuration file and returns the optimization arguments and boundary conditions
    """
    # Name
    cfg_name = config['name']

    # High Level Dyanamical Constants
    g0 = 9.81/1000 # standard gravitational acceleration [km/s^3]
    Sys = yaml_load(config['dynamics']['sys_param_file'])

    # Integration
    a_tol = config['integration']['a_tol']
    r_tol = config['integration']['r_tol']

    # Trajectory Parameters
    N_arcs = config['traj_parameters']['control_arcs']
    N_subarcs = config['traj_parameters']['sub_arcs']
    N_nodes = N_arcs + 1
    N_trials = config['traj_parameters']['MC_trials']
    N_save = config['traj_parameters']['save_pts_detailed']
    arc_length_opt = N_subarcs+1
    arc_length_det = N_subarcs*(N_save-1)+1
    transfer_length_det = N_arcs*(arc_length_det-1) + 1    

    # Forward and backward indices
    indx_f = jnp.array(np.arange(0, N_arcs//2))
    indx_b = jnp.array(np.flip(np.arange(N_arcs//2,N_arcs)))

    # Process Boundary Conditions
    t0 = config['boundary_conditions']['t0']
    tf = config['boundary_conditions']['tf']
    tf_T = config['boundary_conditions']['tf_T']*(24*3600/Sys['Ts'])
    t_node_bound = calc_t_elapsed_nd(t0, tf, N_nodes, Sys['Ts'])
    phasing = config['boundary_conditions']['type']
    dt_detail = (t_node_bound[1] - t_node_bound[0])/(arc_length_det - 1)
    post_insert_length = int(tf_T//dt_detail + 1)
    tf_T = dt_detail*(post_insert_length - 1)
    
    length = transfer_length_det + post_insert_length - 1

    Family0 = load_family(config['boundary_conditions']['initial_orbit']['family_path'])
    Familyf = load_family(config['boundary_conditions']['final_orbit']['family_path'])
    
    Orb0_X_hst = Family0['X_hst'][config['boundary_conditions']['initial_orbit']['Orb_ID'],:,:]
    Orb0_X_hst = jnp.array(np.roll(Orb0_X_hst, shift = -config['boundary_conditions']['initial_orbit']['Start_Idx'], axis=0))
    Orb0_t_hst = jnp.array(Family0['t_hst'][config['boundary_conditions']['initial_orbit']['Orb_ID'],:])
    
    Orbf_X_hst = Familyf['X_hst'][config['boundary_conditions']['final_orbit']['Orb_ID'],:,:]
    Orbf_X_hst = jnp.array(np.roll(Orbf_X_hst, shift = -config['boundary_conditions']['final_orbit']['Start_Idx'], axis=0))
    Orbf_t_hst = jnp.array(Familyf['t_hst'][config['boundary_conditions']['final_orbit']['Orb_ID'],:])

    Boundary_Conds = {}
    Boundary_Conds['Orb0'] = {}
    Boundary_Conds['Orb0']['X_hst'] = Orb0_X_hst
    Boundary_Conds['Orb0']['t_hst'] = Orb0_t_hst

    Boundary_Conds['Orbf'] = {}
    Boundary_Conds['Orbf']['X_hst'] = Orbf_X_hst
    Boundary_Conds['Orbf']['t_hst'] = Orbf_t_hst

    Boundary_Conds['X0_init'] = Orb0_X_hst[0,:]
    Boundary_Conds['Xf_init'] = Orbf_X_hst[0,:]

    if phasing == 'free':
        print("Creating Orbit Interpolants")

        Boundary_Conds['alpha_min'] = config['boundary_conditions']['alpha']['min']
        Boundary_Conds['alpha_max'] = config['boundary_conditions']['alpha']['max']
        Boundary_Conds['beta_min'] = config['boundary_conditions']['beta']['min']
        Boundary_Conds['beta_max'] = config['boundary_conditions']['beta']['max']

        Orb0_coefs = dfx.backward_hermite_coefficients(Orb0_t_hst/jnp.max(Orb0_t_hst), Orb0_X_hst)
        Orbf_coefs = dfx.backward_hermite_coefficients(Orbf_t_hst/jnp.max(Orbf_t_hst), Orbf_X_hst)
        
        Boundary_Conds['X0_interp'] = dfx.CubicInterpolation(Orb0_t_hst/jnp.max(Orb0_t_hst), Orb0_coefs)
        Boundary_Conds['Xf_interp'] = dfx.CubicInterpolation(Orbf_t_hst/jnp.max(Orbf_t_hst), Orbf_coefs)
    else:
        Boundary_Conds['X0_interp'] = Boundary_Conds['X0_init']
        Boundary_Conds['Xf_interp'] = Boundary_Conds['Xf_init']
    
    # Spacecraft Parameters
    m0 = config['engine']['m0'] # Initial mass [kg]
    Isp = config['engine']['Isp']
    U_max = config['engine']['T_max'] # N [kg m/s^3]
    U_Acc_min_nd = (U_max/1000)/(Sys['As']*m0) # Minimum nd acceleration at max mass
    ve = Isp*g0/Sys['Vs']

    if U_Acc_min_nd*(t_node_bound[-1] - t_node_bound[0])/(1-1e-2) > ve:
        print("S/C has insufficient mass to continuously thrust:")
        input("Press any key to continue regardless...")

    # Extract Uncertainty Terms
    r_disp = config['uncertainty']['covariance']['initial_dispersion']['pos_sig'] # km
    v_disp = config['uncertainty']['covariance']['initial_dispersion']['vel_sig'] # km/s
    m_disp = config['uncertainty']['covariance']['initial_dispersion']['mass_sig'] # kg
    Phat_0 = sig2cov(r_disp, v_disp, m_disp, Sys, m0) # Initial dispersion covariance

    r_err = config['uncertainty']['covariance']['initial_error']['pos_sig'] # km
    v_err = config['uncertainty']['covariance']['initial_error']['vel_sig'] # km/s
    m_err = config['uncertainty']['covariance']['initial_error']['mass_sig'] # kg
    Ptild_0 = sig2cov(r_err, v_err, m_err, Sys, m0) # Initial estimation error covariance

    r_targ = config['uncertainty']['covariance']['post_insert_target_total']['pos_sig'] # km
    v_targ = config['uncertainty']['covariance']['post_insert_target_total']['vel_sig'] # km/s
    m_targ = config['uncertainty']['covariance']['post_insert_target_total']['mass_sig'] # kg
    P_XT_targ = sig2cov(r_targ, v_targ, m_targ, Sys, m0)

    a_err = config['uncertainty']['acc_sig']
    U_dyn_err = (a_err/Sys['As'])/U_Acc_min_nd # Normalize against the minimum acceleration used in EOMs
    G_stoch = np.diag(np.array([U_dyn_err, U_dyn_err, U_dyn_err])) # stochastic model error

    fixed_mag = config['uncertainty']['gates']['fixed_mag'] # fraction
    prop_mag = config['uncertainty']['gates']['prop_mag'] # fraction
    fixed_point = config['uncertainty']['gates']['fixed_point'] # fraction
    prop_point = config['uncertainty']['gates']['prop_point'] # fraction
    gates = np.array([fixed_mag, prop_mag, fixed_point, prop_point])
    
    tcm_stat_bound = config['uncertainty']['tcm_stat_bound']
    dV_bound = config['uncertainty']['dV_bound']
    mx_tcm_bound = np.sqrt(stats.chi2.ppf(tcm_stat_bound,3))
    mx_dV_bound = np.sqrt(stats.chi2.ppf(dV_bound,3))

    # Collision Avoidance
    col_obj_body = config['constraints']['col_avoid']['parameters']['obj_body']
    r_obj = np.asarray(Sys[col_obj_body])
    d_safe = jnp.array(config['constraints']['col_avoid']['parameters']['safe_d'])/Sys['Ls']

    det_col_avoid = config['constraints']['col_avoid']['det']['bool']
    stat_col_avoid = config['constraints']['col_avoid']['stat']['bool']

    # Stat chance constrained col avoidance terms
    stat_col_bound = config['constraints']['col_avoid']['stat']['bound']
    mx_col_bound = np.sqrt(stats.norm.ppf(stat_col_bound))
    alpha_UT = config['constraints']['col_avoid']['stat']['UT']['alpha']
    beta_UT = config['constraints']['col_avoid']['stat']['UT']['beta']
    kappa_UT = config['constraints']['col_avoid']['stat']['UT']['kappa']
    
    # Choose Dynamics
    if config['dynamics']['type'] == 'CR3BP':
        dyn_safe = 1737.5/Sys['Ls']

        eom_eval, Aprop_eval, Bprop_eval, U_st_eval, JC_eval = CR3BPDynamics(U_Acc_min_nd, ve, Sys['mu'],dyn_safe)
        dynamics = {'eom_eval': eom_eval,
                 'Aprop_eval': Aprop_eval, 
                 'Bprop_eval': Bprop_eval,
                 'U_st_eval': U_st_eval,
                 'JC_eval': JC_eval}

    # Choose Covariance Propagators
    if feedback_control_type.lower() == 'true_state':
        cov_propagators = TrueStateFeedback_CovPropagators()
    elif feedback_control_type.lower() == 'estimated_state':
        cov_propagators = EstimatedStateFeedback_CovPropagators()

    # Choose Gain Parameterization
    FeedbackGainFuncs = GainParameterizers(gain_parametrization_type)

    # Choose Measurement Model
    meas_type = config['uncertainty']['measurement']['type']
    if meas_type.lower() == 'range':
        meas_dim = 1
        observer_body = config['uncertainty']['measurement']['observer_body']
        alt_lat_lon = config['uncertainty']['measurement']['observer_alt_lat_lon']
        r_obs_x, r_obs_y, r_obs_z = sph_to_cart(alt_lat_lon[0], alt_lat_lon[1], alt_lat_lon[2])
        r_obs_body = np.array([r_obs_x, r_obs_y, r_obs_z])
        r_obs = (Sys['dim'][observer_body] + r_obs_body)/Sys['Ls']
        range_sig = config['uncertainty']['measurement']['range_sig']/Sys['Ls']
        meas_model = range_measurement_model(r_obs, range_sig)
    elif meas_type.lower() == 'position_test':
        meas_dim = 3
        pos_sig = config['uncertainty']['measurement']['pos_sig']/Sys['Ls']
        meas_model = test_pos_measurement_model(pos_sig)

    meas_model['h_vmap'] = jax.vmap(meas_model['h_eval'], in_axes=(0,))
    meas_model['H_vmap'] = jax.vmap(meas_model['H_eval'], in_axes=(0,))
    meas_model['P_v_vmap'] = jax.vmap(meas_model['P_v_eval'], in_axes=(0,))

    # Create models dict
    models = {'dynamics': dynamics,
              'covariance': cov_propagators,
              'measurements': meas_model,
              'feedback_gains': FeedbackGainFuncs}

    # Static configuration arguments
    @dataclass()
    class args_static:
        cfg_name: str
        det_or_stoch: str
        feedback_type: str
        gain_param_type: str
        r_tol: float
        a_tol: float
        N_nodes: int
        N_arcs: int
        N_subarcs: int
        N_trials: int
        N_save: int
        arc_length_det: int
        arc_length_opt: int
        transfer_length_det: int
        post_insert_length: int
        length: int
        meas_dim: int
        ve: float
        U_Acc_min_nd: float
        free_phasing: bool
        det_col_avoid: bool 
        stat_col_avoid: bool
        alpha_UT: float
        beta_UT: float
        kappa_UT: float
        mx_tcm_bound: float
        mx_dV_bound: float
        mx_col_bound: float
    cfg_args = args_static(cfg_name, det_or_stoch, feedback_control_type, gain_parametrization_type, r_tol, a_tol, N_nodes, N_arcs, N_subarcs, N_trials, 
                           N_save, arc_length_det, arc_length_opt, transfer_length_det, post_insert_length, length, meas_dim, ve, U_Acc_min_nd, 
                           True if phasing == 'free' else False, det_col_avoid, stat_col_avoid, alpha_UT, beta_UT, kappa_UT, 
                           mx_tcm_bound, mx_dV_bound, mx_col_bound)

    # Dynamic arguments (arrays and such)
    dyn_args = {'t_node_bound': t_node_bound,
                'tf_T': tf_T,
                'indx_f': indx_f, 
                'indx_b': indx_b,
                'r_obj': r_obj, 
                'd_safe': d_safe,
                'Phat_0': Phat_0,
                'Ptild_0': Ptild_0,
                'P_XT_targ': P_XT_targ,
                'S_XT_targ': np.linalg.cholesky(P_XT_targ),
                'S_XT_targ_inv': np.linalg.inv(np.linalg.cholesky(P_XT_targ)),
                'G_stoch': G_stoch,
                'gates': gates}

    return Sys, models, Boundary_Conds, cfg_args, dyn_args


# ------------------------------------
# Propagation and Optimizer Functions
# ------------------------------------

def prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, cfg_args):
    
    # EOM given time, states and arguments
    dynamics = models['dynamics']
    eom_e = lambda t, states, args: eoms_gen(t, states, args, dynamics, cfg_args)

    # Propagator given intial condition and arguments over span
    propagator_e = lambda X0, U, t0, t1, prop_length, cfg_args: propagator_gen(X0, U, t0, t1, eom_e, prop_length, cfg_args)
    propagators = {'propagator_e': propagator_e}
    iterators = {}
    # Forward ode iteration given index and input_dict
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        propagator_e_fin = lambda X0, U, t0, t1, cfg_args: propagator_gen(X0, U, t0, t1, eom_e, 0, cfg_args)
        propagator_dX0_e = jax.jacfwd(propagator_e_fin, argnums=0)
        propagator_dU_e = jax.jacfwd(propagator_e_fin, argnums=1)
        propagator_dX0_arc_vmap_e = jax.vmap(propagator_dX0_e, in_axes=(0, None, 0, 0, None))
        propagator_dU_arc_vmap_e = jax.vmap(propagator_dU_e, in_axes=(0, None, 0, 0, None))

        propagators['propagator_dX0_e'] = propagator_dX0_e
        propagators['propagator_dU_e'] = propagator_dU_e
        propagators['propagator_dX0_arc_vmap_e'] = propagator_dX0_arc_vmap_e
        propagators['propagator_dU_arc_vmap_e'] = propagator_dU_arc_vmap_e

        cov_propagators = models['covariance']
        propagators['cov_propagators'] = cov_propagators
        cov_propagation_iterate_e = lambda k, input_dict: forward_propagation_cov_iterate(k, input_dict, cov_propagators, dyn_args, cfg_args)
        iterators['cov_propagation_iterate_e'] = cov_propagation_iterate_e

    forward_propagation_iterate_e = lambda k, input_dict: forward_propagation_iterate(k, input_dict, propagators, models, dyn_args, cfg_args)
    backward_propagation_iterate_e = lambda k, input_dict: backward_propagation_iterate(k, input_dict, propagators, models, dyn_args, cfg_args)

    iterators['forward_propagation_iterate_e'] = forward_propagation_iterate_e
    iterators['backward_propagation_iterate_e'] = backward_propagation_iterate_e
    return eom_e, propagators, iterators

def prepare_opt_funcs(Boundary_Conds, iterators, propagators, models, dyn_args, cfg_args):

    func = lambda inputs: objective_and_constraints(inputs, Boundary_Conds, iterators, propagators, models, dyn_args, cfg_args)
    vals = jax.jit(func, backend='cpu')
    grad = jax.jit(jax.jacfwd(func), backend='cpu')
    sens = jax.jit(lambda inputs, cvals: grad(inputs), backend='cpu')

    return vals, grad, sens


# -------------------------------
# Solution Processing and Saving
# -------------------------------

def prepare_sol(solution, Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args):
    
    t_node_bound = dyn_args['t_node_bound']

    # Set up ouput dictionary
    output = {}
    output["Name"] = cfg_args.cfg_name
    output['Orb0'] = {}
    output['Orbf'] = {}
    output['Det'] = {}

    # Create Departure and Arrival Orbit Histories (But Phased with Optimal Transfer)
    orb0_T = Boundary_Conds['Orb0']['t_hst'][-1]
    orb0_X0 = solution.xStar['X0']
    ode_sol = propagators['propagator_e'](orb0_X0, jnp.zeros((3,)), 0, orb0_T, Boundary_Conds['Orb0']['t_hst'].shape[0], cfg_args)
    output['Orb0']['X_hst'] = ode_sol.ys
    output['Orb0']['t_hst'] = ode_sol.ts
    orbf_T = Boundary_Conds['Orbf']['t_hst'][-1]
    orbf_Xf = solution.xStar['Xf']
    ode_sol = propagators['propagator_e'](orbf_Xf, jnp.zeros((3,)), 0, orbf_T, Boundary_Conds['Orbf']['t_hst'].shape[0], cfg_args)
    output['Orbf']['X_hst'] = ode_sol.ys
    output['Orbf']['t_hst'] = ode_sol.ts

    print("Evaluating Detailed Deterministic Trajectory...")
    output['Det'] = sim_Det_traj(solution, Sys, propagators, models, dyn_args, cfg_args)

    # Run Monte Carlo Simulations
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        inputs = {'t_node_bound': t_node_bound,
                  'det_X_hst': output['Det']['X_hst'],
                  'det_X_node_hst': output['Det']['X_node_hst'],
                  'det_U_arc_hst': output['Det']['U_arc_hst'],
                  'K_arc_hst': output['Det']['K_arc_hst'],
                  'dV_mean': output['Det']['dV_mean']}
        if cfg_args.feedback_type.lower() == 'estimated_state':
            inputs['A_hst'] = output['Det']['A_hst']
            inputs['B_hst'] = output['Det']['B_hst']
            inputs['h_hst'] = output['Det']['h_hst']
            inputs['H_hst'] = output['Det']['H_hst']
            inputs['L_hst'] = output['Det']['L_hst']
            inputs['P_v_hst'] = output['Det']['P_v_hst']
        
        rng_seed = 0
        print("Running Detailed Monte Carlo Simulations...")
        output['MC_Runs'] = sim_MC_trajs(inputs, rng_seed, Sys, dyn_args, cfg_args, propagators['propagator_e'], models)
        # output['MC_Runs']['MC_P_ks'], output['MC_Runs']['MC_dV_tcm_scale'] = process_MC_results(output)

    return output
    
def MC_to_cov(errors):
    N_trials = errors.shape[0]
    N_fv = errors.shape[1]

    cov = np.zeros((N_fv, N_fv))

    for k in range(N_trials):
        err_k = errors[k,:].reshape(-1,1)
        cov += err_k @ err_k.T

    cov = cov / N_trials
    return cov

def process_MC_results(output):
    # Evaluate MC State Deviation Covariances
    X_MC_errors = output['MC_Runs']['X_hsts'] - output['Det']['X_hst']  # (N_trials x N x 7)

    length = X_MC_errors.shape[1]
    MC_P_ks = np.zeros((length, 7, 7))
    for k in range(length):
        X_errors_k = X_MC_errors[:,k,:]
        P_k = MC_to_cov(X_errors_k)
        MC_P_ks[k,:,:] = P_k

    # Evaluate scale of the dV_tcm chi2 distribution
    dV_tcms = output['MC_Runs']['dV_tcms']
    MC_dV_tcm_scale = stats.chi2.fit(dV_tcms, fdf=3, floc=0)[2]

    return MC_P_ks, MC_dV_tcm_scale

def save_sol(output, Sys, save_loc: str, dyn_args, cfg_args):
    with h5py.File(save_loc+"data.h5", "w") as f:
        f.create_dataset("Name", data=output['Name'])
        f.create_dataset("info_N_arcs", data=cfg_args.N_arcs)
        f.create_dataset("info_N_subarcs", data=cfg_args.N_subarcs)
        f.create_dataset("info_N_save", data=cfg_args.N_save)
        f.create_dataset("info_length_transfer", data=output['Det']['length_transfer'])
        f.create_dataset("info_length_arc", data=output['Det']['length_arc'])
        f.create_dataset("Det_or_stoch", data=cfg_args.det_or_stoch)
        f.create_dataset("Orb0_X_hst", data=output['Orb0']['X_hst'])
        f.create_dataset("Orb0_t_hst", data=output['Orb0']['t_hst'])
        f.create_dataset("Orbf_X_hst", data=output['Orbf']['X_hst'])
        f.create_dataset("Orbf_t_hst", data=output['Orbf']['t_hst'])
        f.create_dataset("Det_X_hst", data=output['Det']['X_hst'])
        f.create_dataset("Det_X_node_hst", data=output['Det']['X_node_hst'])
        f.create_dataset("Det_U_hst", data=output['Det']['U_hst'])
        f.create_dataset("Det_U_hst_sph", data=output['Det']['U_hst_sph'])
        f.create_dataset("Det_U_arc_hst", data=output['Det']['U_arc_hst'])
        f.create_dataset("Det_t_hst", data=output['Det']['t_hst'])
        f.create_dataset("Det_t_node_hst", data=output['Det']['t_node_hst'])
        f.create_dataset("Det_dV_mean", data=output['Det']['dV_mean'])

        if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
            f.create_dataset("Feedback_type", data=cfg_args.feedback_type)
            f.create_dataset("Det_TCM_norm_dV_hst", data=output['Det']['TCM_norm_dV_hst'])
            f.create_dataset("Det_TCM_norm_bound_hst", data=output['Det']['TCM_norm_bound_hst'])
            f.create_dataset("Det_U_norm_dV_hst", data=output['Det']['U_norm_dV_hst'])
            f.create_dataset("Det_U_norm_bound_hst", data=output['Det']['U_norm_bound_hst'])
            f.create_dataset("Det_dV_stat", data=output['Det']['dV_stat'])
            f.create_dataset("Det_dV_bound", data=output['Det']['dV_bound'])
            f.create_dataset("Det_A_hst", data=output['Det']['A_hst'])
            f.create_dataset("Det_B_hst", data=output['Det']['B_hst'])
            f.create_dataset("Det_K_hst", data=output['Det']['K_hst'])
            f.create_dataset("Det_gain_weights_hst", data=output['Det']['gain_weights_hst'])
            f.create_dataset("Det_P_hst", data=output['Det']['P_hst'])
            if cfg_args.feedback_type.lower() == 'estimated_state':
                f.create_dataset("Det_Phat_hst", data=output['Det']['Phat_hst'])
                f.create_dataset("Det_Ptild_hst", data=output['Det']['Ptild_hst'])
            f.create_dataset("Det_P_Xf_targ", data=output['Det']['P_Xf_targ'])
            f.create_dataset("Det_P_XT_targ", data=output['Det']['P_XT_targ'])
            f.create_dataset("Det_P_Targ_hst", data=output['Det']['P_Targ_hst'])
            f.create_dataset("MC_X_hsts", data=output['MC_Runs']['X_hsts'])
            if cfg_args.feedback_type.lower() == 'estimated_state':
                f.create_dataset("MC_Xhat_hsts", data=output['MC_Runs']['Xhat_hsts'])
            f.create_dataset("MC_t_hsts", data=output['MC_Runs']['t_hsts'])
            f.create_dataset("MC_U_hsts", data=output['MC_Runs']['U_hsts'])
            f.create_dataset("MC_U_hsts_sph", data=output['MC_Runs']['U_hsts_sph'])
            f.create_dataset("MC_dVs", data=output['MC_Runs']['dVs'])


    savemat(save_loc+"Sys.mat",Sys)
    return

def save_OptimizerSol(solution, cfg_arg, save_loc: str):
    with h5py.File(save_loc, "w") as f:
        f.create_dataset("X0", data=solution.xStar['X0'])
        f.create_dataset("Xf", data=solution.xStar['Xf'])
        f.create_dataset("U_arc_hst", data=solution.xStar['U_arc_hst'])
        if cfg_arg.free_phasing:
            f.create_dataset("alpha", data=solution.xStar['alpha'])
            f.create_dataset("beta", data=solution.xStar['beta'])
        if cfg_arg.det_or_stoch == 'stochastic_gauss_zoh':
            f.create_dataset("gain_weights", data=solution.xStar['gain_weights'])
    return

def load_OptimizerSol(input_loc: str) -> dict:
    sol = {}
    with h5py.File(input_loc, "r") as f:
        sol["X0"] = f["X0"][:]
        sol["Xf"] = f["Xf"][:]
        sol["U_arc_hst"] = f["U_arc_hst"][:]
        if "alpha" in f.keys():
            sol["alpha"] = f["alpha"][:]
        if "beta" in f.keys():
            sol["beta"] = f["beta"][:]
        if "gain_weights" in f.keys():
            sol["gain_weights"] = f["gain_weights"][:]

    return sol


import copy
from typing import Dict, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp


def _to_numpy_tree(tree):
    """Convert a pytree of arrays to numpy arrays."""
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _to_jax_dict(d):
    """Convert a dict of arrays to JAX arrays."""
    return {k: jnp.asarray(v) for k, v in d.items()}


def check_jacobian_fd_vs_ad(
    func,
    grad_func,
    x0: Dict[str, Any],
    h: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, np.ndarray]],
           Dict[str, Dict[str, np.ndarray]],
           Dict[str, Dict[str, Any]]]:
    """
    Compare JAX autodiff Jacobian against central finite-difference Jacobian.

    Parameters
    ----------
    func : callable
        Function mapping `inputs_dict -> outputs_dict`.
        Example: outputs = func(inputs)
        outputs must be a dict: {out_key: array}.
    grad_func : callable
        Function mapping `inputs_dict -> jac_dict`, typically from jax.jacrev.
        Must return nested dict: {out_key: {in_key: jac_array}}.
    x0 : dict
        Initial inputs dict. Keys are input names (e.g. 'X0', 'Xf', 'controls'),
        values are numpy/JAX arrays.
    h : float
        Finite-difference step size (central difference).
    atol : float
        Absolute tolerance for considering AD vs FD “different”.
    rtol : float
        Relative tolerance for considering AD vs FD “different”.
    verbose : bool
        If True, print a summary of max errors.

    Returns
    -------
    fd_jac : dict
        Nested dict {out_key: {in_key: fd_jac_array}} with FD Jacobians.
    ad_jac : dict
        Nested dict {out_key: {in_key: ad_jac_array}} converted to numpy.
    diff_report : dict
        Nested dict {out_key: {in_key: {...}}} with:
            - 'max_abs_err'
            - 'max_rel_err'
            - 'bad_indices': list of index tuples where error > tol.
              Indices are in the combined shape: out_shape + in_shape.
    """

    # Make a numpy copy of the initial point so we can mutate it
    x0_np = {k: np.array(v, dtype=float) for k, v in x0.items()}

    # Evaluate function at x0 to get output structure
    y0 = func(_to_jax_dict(x0_np))
    y0_np = _to_numpy_tree(y0)

    # Evaluate autodiff Jacobian and convert to numpy
    ad_jac_raw = grad_func(_to_jax_dict(x0_np))
    ad_jac = {}
    for out_key, inner in ad_jac_raw.items():
        ad_jac[out_key] = {}
        for in_key, arr in inner.items():
            ad_jac[out_key][in_key] = np.asarray(arr)

    out_keys = list(y0_np.keys())
    in_keys = list(x0_np.keys())

    # Build FD Jacobian with same structure: J[out][in] has shape y[out].shape + x[in].shape
    fd_jac: Dict[str, Dict[str, np.ndarray]] = {
        out_key: {
            in_key: np.zeros(y0_np[out_key].shape + x0_np[in_key].shape, dtype=float)
            for in_key in in_keys
        }
        for out_key in out_keys
    }

    # Central difference loop over each input variable and each element of that variable
    for in_key in in_keys:
        x_base = x0_np[in_key]
        x_shape = x_base.shape
        if x_shape == ():  # scalar
            index_iter = [()]
        else:
            index_iter = np.ndindex(*x_shape)

        for idx in index_iter:
            # Create perturbed copies of all inputs
            x_plus = {k: v.copy() for k, v in x0_np.items()}
            x_minus = {k: v.copy() for k, v in x0_np.items()}

            x_plus[in_key][idx] += h
            x_minus[in_key][idx] -= h

            # Evaluate function at x+h and x-h
            y_plus = _to_numpy_tree(func(_to_jax_dict(x_plus)))
            y_minus = _to_numpy_tree(func(_to_jax_dict(x_minus)))

            # FD derivative for each output wrt this single input element
            for out_key in out_keys:
                dy = (y_plus[out_key] - y_minus[out_key]) / (2.0 * h)  # shape = y_shape
                # Place it into the correct slice of the Jacobian
                fd_jac[out_key][in_key][(Ellipsis,) + idx] = dy

    # Compare AD vs FD
    diff_report: Dict[str, Dict[str, Any]] = {}
    for out_key in out_keys:
        diff_report[out_key] = {}
        for in_key in in_keys:
            fd_arr = fd_jac[out_key][in_key]
            ad_arr = ad_jac[out_key].get(in_key, None)

            if ad_arr is None:
                # grad_func didn't return this block; skip
                diff_report[out_key][in_key] = {
                    "present_in_ad": False,
                    "present_in_fd": True,
                    "note": "AD jacobian missing this block.",
                }
                continue

            if fd_arr.shape != ad_arr.shape:
                diff_report[out_key][in_key] = {
                    "present_in_ad": True,
                    "present_in_fd": True,
                    "shape_mismatch": (fd_arr.shape, ad_arr.shape),
                }
                continue

            abs_err = np.abs(ad_arr - fd_arr)
            denom = np.maximum(np.abs(ad_arr), np.abs(fd_arr))
            denom = np.where(denom == 0.0, 1.0, denom)
            rel_err = abs_err / denom

            max_abs = float(abs_err.max()) if abs_err.size > 0 else 0.0
            max_rel = float(rel_err.max()) if rel_err.size > 0 else 0.0

            bad_mask = abs_err > (atol + rtol * denom)
            bad_indices = [tuple(idx) for idx in np.argwhere(bad_mask)]

            diff_report[out_key][in_key] = {
                "present_in_ad": True,
                "present_in_fd": True,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "num_bad": int(bad_mask.sum()),
                "bad_indices": bad_indices,
            }

            if verbose and (max_abs > atol and max_rel > rtol):
                print(
                    f"[{out_key}] wrt [{in_key}]: "
                    f"max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e}, "
                    f"num_bad={len(bad_indices)}"
                )

    return fd_jac, ad_jac, diff_report
