import yaml
import numpy as np

import jax
from jax import numpy as jnp
import diffrax as dfx

from Lib.math import calc_t_elapsed_nd, sig2cov
from Lib.dynamics import CR3BPDynamics,TrueStateFeedback_CovPropagators, forward_propagation_iterate, backward_propagation_iterate, objective_and_constraints, forward_propagation_cov_iterate, sim_MC_trajs, sim_Det_traj

from scipy import stats

from dataclasses import dataclass, replace

import h5py

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

def process_config(config, det_or_stoch: str, feedback_control_type: str):
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
    N_nodes = N_arcs + 1
    N_trials = config['traj_parameters']['MC_trials']
    N_save = config['traj_parameters']['save_pts_detailed']

    # Forward and backward indices
    indx_f = jnp.array(np.arange(0, N_arcs//2))
    indx_b = jnp.array(np.flip(np.arange(N_arcs//2,N_arcs)))

    # Process Boundary Conditions
    t0 = config['boundary_conditions']['t0']
    tf = config['boundary_conditions']['tf']
    t_node_bound = calc_t_elapsed_nd(t0, tf, N_nodes, Sys['Ts'])
    phasing = config['boundary_conditions']['type']

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

    # Uncertainty
    r_1sig = config['uncertainty']['covariance']['initial']['pos_sig'] # km
    v_1sig = config['uncertainty']['covariance']['initial']['vel_sig'] # km/s
    m_1sig = config['uncertainty']['covariance']['initial']['mass_sig'] # kg

    r_1sig_t = config['uncertainty']['covariance']['target']['pos_sig'] # km
    v_1sig_t = config['uncertainty']['covariance']['target']['vel_sig'] # km/s
    m_1sig_t = config['uncertainty']['covariance']['target']['mass_sig'] # kg

    a_err = config['uncertainty']['acc_sig']

    fixed_mag = config['uncertainty']['gates']['fixed_mag'] # fraction
    prop_mag = config['uncertainty']['gates']['prop_mag'] # fraction
    fixed_point = config['uncertainty']['gates']['fixed_point'] # fraction
    prop_point = config['uncertainty']['gates']['prop_point'] # fraction
    
    tcm_stat_bound = config['uncertainty']['tcm_stat_bound']
    dV_bound = config['uncertainty']['dV_bound']
    mx_tcm_bound = np.sqrt(stats.chi2.ppf(tcm_stat_bound,3))
    mx_dV_bound = np.sqrt(stats.chi2.ppf(dV_bound,3))

    # Covariance Terms
    init_cov = sig2cov(r_1sig, v_1sig, m_1sig, Sys, m0)
    targ_cov = sig2cov(r_1sig_t, v_1sig_t, m_1sig_t, Sys, m0)
    U_dyn_err = (a_err/Sys['As'])/U_Acc_min_nd # Normalize against the minimum acceleration used in EOMs

    G_stoch = np.diag(np.array([U_dyn_err, U_dyn_err, U_dyn_err])) # stochastic model error
    gates = np.array([fixed_mag, prop_mag, fixed_point, prop_point])

    # Collision Avoidance
    r_obs = jnp.array(config['constraints']['col_avoid']['parameters']['r_obs'])/Sys['Ls']
    d_safe = jnp.array(config['constraints']['col_avoid']['parameters']['safe_d'])/Sys['Ls']

    det_col_avoid = config['constraints']['col_avoid']['det']['bool']
    stat_col_avoid = config['constraints']['col_avoid']['stat']['bool']

    # Stat chance constrained col avoidance terms
    stat_col_bound = config['constraints']['col_avoid']['stat']['bound']
    mx_col_bound = np.sqrt(stats.norm.ppf(stat_col_bound))
    alpha_UT = config['constraints']['col_avoid']['stat']['UT']['alpha']
    beta_UT = config['constraints']['col_avoid']['stat']['UT']['beta']
    kappa_UT = config['constraints']['col_avoid']['stat']['UT']['kappa']
    
    # SNOPT Options
    optOptions = {"Major optimality tolerance": config['SNOPT']['major_opt_tol'],
                  "Major feasibility tolerance": config['SNOPT']['major_feas_tol'],
                  "Minor feasibility tolerance": config['SNOPT']['minor_feas_tol'],
                  'Major iterations limit': config['SNOPT']['major_iter_limit'],
                  'Partial price': config['SNOPT']['partial_price'],
                  'Linesearch tolerance': config['SNOPT']['linesearch_tol'],
                  'Function precision': config['SNOPT']['function_prec'],
                  'Verify level': -1,
                  'Nonderivative linesearch': 0,
                  'Elastic weight': config['SNOPT']['elastic_weight']}

    # Choose Dynamics
    if config['dynamics']['type'] == 'CR3BP':
        dyn_safe = 1737.5/Sys['Ls']

        eom_eval, Aprop_eval, Bprop_eval = CR3BPDynamics(U_Acc_min_nd, ve, Sys['mu'],dyn_safe)
        dynamics = {'eom_eval': eom_eval,
                 'Aprop_eval': Aprop_eval, 
                 'Bprop_eval': Bprop_eval}
    
    state_cov_dynamics = {'dynamics': dynamics}

    if feedback_control_type.lower() == 'true_state':
        cov_propagators = TrueStateFeedback_CovPropagators()
    
    state_cov_dynamics['cov_propagators'] = cov_propagators

    # Static configuration arguments
    @dataclass()
    class args_static:
        cfg_name: str
        det_or_stoch: str
        feedback_control_type: str
        r_tol: float
        a_tol: float
        N_nodes: int
        N_arcs: int
        N_trials: int
        N_save: int
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
    cfg_args = args_static(cfg_name, det_or_stoch, feedback_control_type, r_tol, a_tol, N_nodes, N_arcs, N_trials, N_save, ve, U_Acc_min_nd,
                           True if phasing == 'free' else False, det_col_avoid, stat_col_avoid, alpha_UT, beta_UT, 
                           kappa_UT, mx_tcm_bound, mx_dV_bound, mx_col_bound)

    # Dynamic arguments (arrays and such)
    dyn_args = {'t_node_bound': t_node_bound,
                'indx_f': indx_f, 
                'indx_b': indx_b,
                'r_obs': r_obs, 
                'd_safe': d_safe,
                'init_cov': init_cov,
                'targ_cov': targ_cov,
                'targ_cov_inv': np.linalg.inv(targ_cov),
                'targ_cov_inv_sqrt': np.linalg.cholesky(np.linalg.inv(targ_cov)),
                'G_stoch': G_stoch,
                'gates': gates}

    return Sys, state_cov_dynamics, Boundary_Conds, cfg_args, dyn_args, optOptions


# ------------------------------------
# Propagation and Optimizer Functions
# ------------------------------------

def prepare_prop_funcs(eoms_gen, state_cov_dynamics, propagator_gen, dyn_args, cfg_args):
    
    # EOM given time, states and arguments
    dynamics = state_cov_dynamics['dynamics']
    eom_e = lambda t, states, args: eoms_gen(t, states, args, dynamics, cfg_args)

    # Propagator given intial condition and arguments over span
    propagator_e = lambda X0, U, t0, t1, cfg_args: propagator_gen(X0, U, t0, t1, eom_e, cfg_args)
    propagators = {'propagator_e': propagator_e}
    iterators = {}
    # Forward ode iteration given index and input_dict
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        propagator_e_fin = lambda X0, U, t0, t1, cfg_args: propagator_gen(X0, U, t0, t1, eom_e, replace(cfg_args, N_save=1))
        propagator_dX0_e = jax.jacfwd(propagator_e_fin, argnums=0)
        propagator_dU_e = jax.jacfwd(propagator_e_fin, argnums=1)

        propagators['propagator_dX0_e'] = propagator_dX0_e
        propagators['propagator_dU_e'] = propagator_dU_e

        cov_propagators = state_cov_dynamics['cov_propagators']
        propagators['cov_propagators'] = cov_propagators
        cov_propagation_iterate_e = lambda k, input_dict: forward_propagation_cov_iterate(k, input_dict, cov_propagators, dyn_args, cfg_args)
        iterators['cov_propagation_iterate_e'] = cov_propagation_iterate_e

    forward_propagation_iterate_e = lambda k, input_dict: forward_propagation_iterate(k, input_dict, propagators, dyn_args, cfg_args)
    backward_propagation_iterate_e = lambda k, input_dict: backward_propagation_iterate(k, input_dict, propagators, dyn_args, cfg_args)

    iterators['forward_propagation_iterate_e'] = forward_propagation_iterate_e
    iterators['backward_propagation_iterate_e'] = backward_propagation_iterate_e
    return eom_e, propagators, iterators

def prepare_opt_funcs(Boundary_Conds, iterators, dyn_args, cfg_args):

    func = lambda inputs: objective_and_constraints(inputs, Boundary_Conds, iterators, dyn_args, cfg_args)
    vals = jax.jit(func, backend='cpu')
    grad = jax.jit(jax.jacfwd(func), backend='cpu')
    sens = jax.jit(lambda inputs, cvals: grad(inputs), backend='cpu')

    return vals, grad, sens


# -------------------------------
# Solution Processing and Saving
# -------------------------------

def prepare_sol(solution, Sys, Boundary_Conds, propagators, dyn_args, cfg_args):
    
    t_node_bound = dyn_args['t_node_bound']

    # Set up ouput dictionary
    output = {}
    output["Name"] = cfg_args.cfg_name
    output['Orb0'] = {}
    output['Orbf'] = {}
    output['Det'] = {}

    # Store departure and arrival orbits
    output['Orb0']['X_hst'] = Boundary_Conds['Orb0']['X_hst']
    output['Orb0']['t_hst'] = Boundary_Conds['Orb0']['t_hst']
    output['Orbf']['X_hst'] = Boundary_Conds['Orbf']['X_hst']
    output['Orbf']['t_hst'] = Boundary_Conds['Orbf']['t_hst']

    print("Evaluating Detailed Deterministic Trajectory...")
    output['Det'] = sim_Det_traj(solution, Sys, propagators, dyn_args, cfg_args)

    # Run Monte Carlo Simulations
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        inputs = {'t_node_bound': t_node_bound,
                  'det_X_node_hst': output['Det']['X_node_hst'],
                  'det_U_arc_hst': output['Det']['U_arc_hst'],
                  'K_arc_hst': output['Det']['K_arc_hst'],
                  'dV_mean': output['Det']['dV_mean']}
        
        rng_seed = 0
        print("Running Detailed Monte Carlo Simulations...")
        output['MC_Runs'] = sim_MC_trajs(inputs, rng_seed, Sys, dyn_args, cfg_args, propagators['propagator_e'])
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
            f.create_dataset("Det_TCM_norm_dV_hst", data=output['Det']['TCM_norm_dV_hst'])
            f.create_dataset("Det_TCM_norm_bound_hst", data=output['Det']['TCM_norm_bound_hst'])
            f.create_dataset("Det_U_norm_dV_hst", data=output['Det']['U_norm_dV_hst'])
            f.create_dataset("Det_U_norm_bound_hst", data=output['Det']['U_norm_bound_hst'])
            f.create_dataset("Det_dV_stat", data=output['Det']['dV_stat'])
            f.create_dataset("Det_dV_bound", data=output['Det']['dV_bound'])
            f.create_dataset("Det_A_hst", data=output['Det']['A_hst'])
            f.create_dataset("Det_B_hst", data=output['Det']['B_hst'])
            f.create_dataset("Det_K_hst", data=output['Det']['K_hst'])
            f.create_dataset("Det_P_hst", data=output['Det']['P_hst'])
            f.create_dataset("Det_P_Xf_targ", data=dyn_args['targ_cov'])
            f.create_dataset("MC_X_hsts", data=output['MC_Runs']['X_hsts'])
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
        f.create_dataset("controls", data=solution.xStar['controls'])
        if cfg_arg.free_phasing:
            f.create_dataset("alpha", data=solution.xStar['alpha'])
            f.create_dataset("beta", data=solution.xStar['beta'])
        if cfg_arg.det_or_stoch == 'stochastic_gauss_zoh':
            f.create_dataset("xis", data=solution.xStar['xis'])
    return

def load_OptimizerSol(input_loc: str) -> dict:
    sol = {}
    with h5py.File(input_loc, "r") as f:
        sol["X0"] = f["X0"][:]
        sol["Xf"] = f["Xf"][:]
        sol["controls"] = f["controls"][:]
        if "alpha" in f.keys():
            sol["alpha"] = f["alpha"][:]
        if "beta" in f.keys():
            sol["beta"] = f["beta"][:]
        if "xis" in f.keys():
            sol["xis"] = f["xis"][:]

    return sol
