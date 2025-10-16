import yaml
import numpy as np

import jax
from jax import numpy as jnp
import diffrax as dfx

from Lib.math import calc_t_elapsed_nd, sig2cov, cart2sph
from Lib.dynamics import CR3BPDynamics, forward_propagation_iterate, backward_propagation_iterate, objective_and_constraints, forward_propagation_cov_iterate, sim_MC_trajs, sim_Det_traj, MC_worker_par

import functools as ft

import scipy as sp
from scipy.stats import chi2

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

def process_jac_ordered(jax_jac_fun, func_order, var_order):
    """
    Wrap a JAX jacobian-producing function so it returns a sparse-ish
    nested dict where missing (func,var) blocks are simply omitted.

    Parameters
    ----------
    jax_jac_fun : callable
        Function that returns a nested dict of jacobians:
        {func_name: {var_name: block}}
        E.g., produced by jax.jacrev on a dict-returning objective/constraint fn.
    func_order : list[tuple[str, int]]
        [("obj",1), ("c_X0",7), ...]  (sizes used only to reshape if present)
    var_order : list[tuple[str, int]]
        [("X0",7), ("Xf",7), ("controls",3*nodes), ...]

    Returns
    -------
    wrapped_fun : callable
        Same call signature as jax_jac_fun. Returns:
        {func_name: {var_name: jnp.ndarray of shape (m, n)}}
        but ONLY includes keys actually present in jax_jac_fun's output.
    """
    fsize = dict(func_order)
    vsize = dict(var_order)

    def reshape_block(A, m, n):
        A = jnp.asarray(A)
        # Try to coerce to (m,n) when possible; otherwise raise clearly.
        if A.ndim == 0:
            if m * n == 1:
                return jnp.reshape(A, (m, n))
            raise ValueError(f"Scalar block cannot reshape to ({m},{n}).")
        if A.ndim == 1:
            if A.size == n and m == 1:
                return A.reshape(1, n)
            if A.size == m and n == 1:
                return A.reshape(m, 1)
            if A.size == m * n:
                return A.reshape(m, n)
            raise ValueError(f"1D block of size {A.size} incompatible with ({m},{n}).")
        if A.ndim == 2:
            if A.shape == (m, n):
                return A
            if A.size == m * n:
                return A.reshape(m, n)
            raise ValueError(f"Got {A.shape}, expected ({m},{n}).")
        raise ValueError(f"Unexpected ndim {A.ndim} for jacobian block.")

    def wrapped_fun(*args, **kwargs):
        grad_raw = jax_jac_fun(*args, **kwargs)  # nested dict
        out = {}
        for fname, _ in func_order:
            row = grad_raw.get(fname, None)
            if row is None:
                continue  # skip missing function entirely
            shaped_row = {}
            for vname, _ in var_order:
                blk = row.get(vname, None)
                if blk is None:
                    continue  # skip missing variable block
                m, n = fsize[fname], vsize[vname]
                shaped_row[vname] = reshape_block(blk, m, n)
            if shaped_row:
                out[fname] = shaped_row
        return out

    return wrapped_fun

def process_sparsity(grad_nonsparse):
    grad_sparse = {}
    
    for key, val in grad_nonsparse.items():
        cur_obj_constr = val
        new_obj_constr = {}
        for key2, val2_jax in cur_obj_constr.items():
            """
            val2 = np.array(val2_jax)
            
            if len(val2.shape) != 2:
                if key == 'c_P_Xf':
                    new_obj_constr[key2] = val2.reshape(1,-1)
                else:
                    new_obj_constr[key2] = val2.reshape(-1,1)
            else:
                new_obj_constr[key2] = val2
            """
            new_obj_constr[key2] = val2_jax
            if jnp.all(new_obj_constr[key2] == 0):
                new_obj_constr.pop(key2, None)
        
        grad_sparse[key] = new_obj_constr
    
    return grad_sparse

def process_config(config, problem_type: str):
    """ This function processes the configuration file and returns the optimization arguments and boundary conditions
    """
    # Name
    cfg_name = config['name']

    # High Level Dyanamical Constants
    g0 = 9.81/1000 # standard gravitational acceleration [km/s^3]
    Sys = yaml_load(config['dynamics']['sys_param_file'])

    # Hot starter (to do later) **********

    # Optimization type
    det_or_stoch = problem_type

    # Integration
    int_save = config['integration']['int_save']
    a_tol = config['integration']['a_tol']
    r_tol = config['integration']['r_tol']

    # Segments
    N = config['segments']
    nodes = N-1

    # MC Trials
    N_trials = config['MC_trials']

    # Forward and backward indices
    indx_f = jnp.array(np.arange(0, nodes//2))
    indx_b = jnp.array(np.flip(np.arange(nodes//2,nodes)))

    # Process Boundary Conditions
    t0 = config['boundary_conditions']['t0']
    tf = config['boundary_conditions']['tf']
    t_node_bound = calc_t_elapsed_nd(t0, tf, N, Sys['Ts'])
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
    
    eps = config['uncertainty']['eps']
    
    # Covariance Terms
    init_cov = sig2cov(r_1sig, v_1sig, m_1sig, Sys, m0)
    targ_cov = sig2cov(r_1sig_t, v_1sig_t, m_1sig_t, Sys, m0)
    G_stoch = np.diag(np.array([a_err/Sys['As'], a_err/Sys['As'], a_err/Sys['As']])) # stochastic model error
    gates = np.array([fixed_mag, prop_mag, fixed_point, prop_point])

    # Collision Avoidance
    r_obs = jnp.array(config['constraints']['col_avoid']['parameters']['r_obs'])/Sys['Ls']
    d_safe = jnp.array(config['constraints']['col_avoid']['parameters']['safe_d'])/Sys['Ls']

    det_col_avoid = config['constraints']['col_avoid']['det']['bool']
    stat_col_avoid = config['constraints']['col_avoid']['stat']['bool']

    # Stat chance constrained col avoidance terms
    alpha_UT = config['constraints']['col_avoid']['stat']['UT']['alpha']
    beta_UT = config['constraints']['col_avoid']['stat']['UT']['beta']
    kappa_UT = config['constraints']['col_avoid']['stat']['UT']['kappa']
    
    optOptions = {"Major optimality tolerance": config['SNOPT']['major_opt_tol'],
                  "Major feasibility tolerance": config['SNOPT']['major_feas_tol'],
                  "Minor feasibility tolerance": config['SNOPT']['minor_feas_tol'],
                  'Major iterations limit': config['SNOPT']['major_iter_limit'],
                  'Partial price': config['SNOPT']['partial_price'],
                  'Linesearch tolerance': config['SNOPT']['linesearch_tol'],
                  'Function precision': config['SNOPT']['function_prec'],
                  'Verify level': -1,
                  'Nonderivative linesearch': 0}

    # Choose Dynamics
    if config['dynamics']['type'] == 'CR3BP':
        dyn_safe = 1737.5/Sys['Ls']

        eom_eval, Aprop_eval, Bprop_eval = CR3BPDynamics(U_Acc_min_nd, ve, Sys['mu'],dyn_safe)
        dynamics = {'eom_eval': eom_eval,
                 'Aprop_eval': Aprop_eval, 
                 'Bprop_eval': Bprop_eval}
    
    # Static configuration arguments
    @dataclass()
    class args_static:
        cfg_name: str
        det_or_stoch: str
        r_tol: float
        a_tol: float
        nodes: int
        N: int
        N_trials: int
        ve: float
        T_max_nd: float
        int_save: int
        free_phasing: bool
        det_col_avoid: bool 
        stat_col_avoid: bool
        alpha_UT: float
        beta_UT: float
        kappa_UT: float
        eps_UT: float
        mx: float
    cfg_args = args_static(cfg_name, det_or_stoch, r_tol, a_tol, nodes, N, N_trials, ve, U_Acc_min_nd, int_save, 
                           True if phasing == 'free' else False, det_col_avoid, stat_col_avoid, 
                           alpha_UT, beta_UT, kappa_UT, eps, np.sqrt(chi2.ppf(1-eps,3)))

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

    return Sys, dynamics, Boundary_Conds, cfg_args, dyn_args, optOptions
    
def prepare_prop_funcs(eoms_gen, dynamics, propagator_gen, propagator_gen_fin, dyn_args, cfg_args):
    
    # EOM given time, states and arguments
    eom_e = lambda t, states, args: eoms_gen(t, states, args, dynamics, cfg_args)

    # Propagator given intial condition and arguments over span
    propagator_e = lambda X0, U, t0, t1, cfg_args: propagator_gen(X0, U, t0, t1, eom_e, cfg_args)
    propagators = {'propagator_e': propagator_e}
    iterators = {}
    # Forward ode iteration given index and input_dict
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        propagator_e_fin = lambda X0, U, t0, t1, cfg_args: propagator_gen_fin(X0, U, t0, t1, eom_e, cfg_args)
        propagator_dX0_e = jax.jacfwd(propagator_e_fin, argnums=0)
        propagator_dU_e = jax.jacfwd(propagator_e_fin, argnums=1)

        propagators['propagator_dX0_e'] = propagator_dX0_e
        propagators['propagator_dU_e'] = propagator_dU_e

        cov_propagation_iterate_e = lambda i, input_dict: forward_propagation_cov_iterate(i, input_dict, dyn_args, cfg_args)
        iterators['cov_propagation_iterate_e'] = cov_propagation_iterate_e

    forward_propagation_iterate_e = lambda i, input_dict: forward_propagation_iterate(i, input_dict, propagators, dyn_args, cfg_args)
    backward_propagation_iterate_e = lambda i, input_dict: backward_propagation_iterate(i, input_dict, propagators, dyn_args, cfg_args)

    iterators['forward_propagation_iterate_e'] = forward_propagation_iterate_e
    iterators['backward_propagation_iterate_e'] = backward_propagation_iterate_e
    return eom_e, propagators, iterators

def prepare_opt_funcs(Boundary_Conds, iterators, var_order, objcon_order, dyn_args, cfg_args):

    func = lambda inputs: objective_and_constraints(inputs, Boundary_Conds, iterators, dyn_args, cfg_args)
    vals = jax.jit(func, backend='cpu')
    grad = jax.jit(jax.jacfwd(func), backend='cpu')
    grad_ordered = process_jac_ordered(grad, objcon_order, var_order)
    sens = jax.jit(lambda inputs, cvals: grad_ordered(inputs), backend='cpu')

    return vals, grad, grad_ordered, sens

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

    output['Det'] = sim_Det_traj(solution, Sys, propagators, dyn_args, cfg_args)

    # Run Monte Carlo Simulations
    if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        inputs = {'t_node_bound': t_node_bound,
                  'det_X_node_hst': output['Det']['X_node_hst'],
                  'det_U_node_hst': output['Det']['U_node_hst'],
                  'K_ks': output['Det']['K_ks']}
        
        rng_seed = 0

        output['MC_Runs'] = sim_MC_trajs(inputs, rng_seed, dyn_args, cfg_args, propagators['propagator_e'], n_jobs = 8)

    return output
    
def save_sol(output, Sys, save_loc: str, dyn_args, cfg_args):
    with h5py.File(save_loc+"data.h5", "w") as f:
        f.create_dataset("Name", data=output['Name'])
        f.create_dataset("Det_or_stoch", data=cfg_args.det_or_stoch)
        f.create_dataset("Orb0_X_hst", data=output['Orb0']['X_hst'])
        f.create_dataset("Orb0_t_hst", data=output['Orb0']['t_hst'])
        f.create_dataset("Orbf_X_hst", data=output['Orbf']['X_hst'])
        f.create_dataset("Orbf_t_hst", data=output['Orbf']['t_hst'])
        f.create_dataset("Det_X_hst", data=output['Det']['X_hst'])
        f.create_dataset("Det_t_hst", data=output['Det']['t_hst'])
        f.create_dataset("Det_X_node_hst", data=output['Det']['X_node_hst'])
        f.create_dataset("Det_t_node_hst", data=output['Det']['t_node_hst'])
        f.create_dataset("Det_U_hst", data=output['Det']['U_hst'])
        f.create_dataset("Det_U_hst_sph", data=output['Det']['U_hst_sph'])
        f.create_dataset("Det_dV", data=output['Det']['dV'])

        if cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
            f.create_dataset("Det_A_ks", data=output['Det']['A_ks'])
            f.create_dataset("Det_B_ks", data=output['Det']['B_ks'])
            f.create_dataset("Det_K_ks", data=output['Det']['K_ks'])
            f.create_dataset("Det_P_ks", data=output['Det']['P_ks'])
            f.create_dataset("Det_P_Xf_targ", data=dyn_args['targ_cov'])
            f.create_dataset("MC_X_hsts", data=output['MC_Runs']['X_hsts'])
            f.create_dataset("MC_t_hsts", data=output['MC_Runs']['t_hsts'])
            f.create_dataset("MC_U_hsts", data=output['MC_Runs']['U_hsts'])
            f.create_dataset("MC_U_hsts_sph", data=output['MC_Runs']['U_hsts_sph'])

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