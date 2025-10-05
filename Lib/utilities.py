import yaml
import numpy as np

import jax
from jax import numpy as jnp
import diffrax as dfx

from Lib.math import calc_t_elapsed_nd, sig2cov
from Lib.dynamics import CR3BPDynamics, forward_propagation_iterate, backward_propagation_iterate, objective_and_constraints

import functools as ft

import scipy as sp

from dataclasses import dataclass

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

def process_config(config):
    """ This function processes the configuration file and returns the optimization arguments and boundary conditions
    """
    # Name
    cfg_name = config['name']

    # High Level Dyanamical Constants
    g0 = 9.81/1000 # standard gravitational acceleration [km/s^3]
    Sys = yaml_load(config['dynamics']['sys_param_file'])

    # Hot starter (to do later) **********

    # Optimization type
    det_or_stoch = config['det_or_stoch']

    # Integration
    int_save = config['integration']['int_save']
    a_tol = config['integration']['a_tol']
    r_tol = config['integration']['r_tol']

    # Segments
    N = config['segments']
    nodes = N-1

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
    Boundary_Conds['Orb0']['t_hst'] =Orb0_t_hst

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

    # Collision Avoidance
    detdet = config['constraints']['deterministic']['det_col_avoid']['bool']
    stochdet = config['constraints']['stochastic']['det_col_avoid']['bool']
    stochstoch = config['constraints']['stochastic']['stat_col_avoid']['bool']

    # Unscented Transform
    alpha_UT = config['UT']['alpha']
    beta_UT = config['UT']['beta']
    kappa_UT = config['UT']['kappa']
    
    # Collision Avoidance
    r_obs = jnp.array(config['constraints']['deterministic']['det_col_avoid']['parameters']['r_obs'])/Sys['Ls']
    d_safe = jnp.array(config['constraints']['deterministic']['det_col_avoid']['parameters']['safe_d'])/Sys['Ls']
    
    optOptions = {"Major optimality tolerance": config['SNOPT']['major_opt_tol'],
                  "Major feasibility tolerance": config['SNOPT']['major_feas_tol'],
                  "Minor feasibility tolerance": config['SNOPT']['minor_feas_tol'],
                  'Major iterations limit': config['SNOPT']['major_iter_limit'],
                  'Partial price': config['SNOPT']['partial_price'],
                  'Linesearch tolerance': config['SNOPT']['linesearch_tol'],
                  'Function precision': config['SNOPT']['function_prec'],
                  'Verify level': -1,
                  'Nonderivative linesearch': 1}

    # Covariance Terms
    init_cov = sig2cov(r_1sig, v_1sig, m_1sig, Sys)
    targ_cov = sig2cov(r_1sig_t, v_1sig_t, m_1sig_t, Sys)

    G_stoch = np.diag(np.array([0, 0, 0, a_err/Sys['As'], a_err/Sys['As'], a_err/Sys['As'], 0])) # stochastic model error

    gates = np.array([fixed_mag, prop_mag, fixed_point, prop_point])

    
    # Choose Dynamics
    if config['dynamics']['type'] == 'CR3BP':
        dyn_safe = 1737.5/Sys['Ls']

        eom_eval, Aprop_eval, Bprop_eval = CR3BPDynamics(U_Acc_min_nd, ve, Sys['mu'],dyn_safe)
        dynamics = {'eom_eval': eom_eval,
                 'Aprop_eval': Aprop_eval, 
                 'Bprop_eval': Bprop_eval}


    # Optimization problem arguments
    
    # Static configuration arguments
    @dataclass()
    class args_static:
        cfg_name: str
        det_or_stoch: str
        r_tol: float
        a_tol: float
        nodes: int
        N: int
        ve: float
        T_max_nd: float
        int_save: int
        free_phasing: bool
        det_col_avoid: bool 
    cfg_args = args_static(cfg_name, det_or_stoch, r_tol, a_tol, nodes, N, ve, U_Acc_min_nd, int_save, True if phasing == 'free' else False, detdet)

    # Dynamic arguments
    dyn_args = {'t_node_bound': t_node_bound,
                'indx_f': indx_f, 
                'indx_b': indx_b,
                'r_obs': r_obs, 
                'd_safe': d_safe}

    return Sys, dynamics, Boundary_Conds, cfg_args, dyn_args, optOptions
    
def prepare_prop_funcs(eoms_gen, dynamics, propagator_gen, dyn_args, cfg_args):
    
    # EOM given time, states and arguments
    eom_e = lambda t, states, args: eoms_gen(t, states, args, dynamics, cfg_args)

    # Propagator given intial condition and arguments over span
    propagator_e = lambda X0, U, t0, t1, cfg_args: propagator_gen(X0, U, t0, t1, eom_e, cfg_args)
    propagators = {'propagator_e': propagator_e}

    # Forward ode iteration given index and input_dict
    if cfg_args.det_or_stoch.lower() == 'deterministic':
        forward_propagation_iterate_e = lambda i, input_dict: forward_propagation_iterate(i, input_dict, propagator_e, dyn_args, cfg_args)
        backward_propagation_iterate_e = lambda i, input_dict: backward_propagation_iterate(i, input_dict, propagator_e, dyn_args, cfg_args)
    elif cfg_args.det_or_stoch.lower() == 'stochastic_gauss_zoh':
        propagator_dX0_e = jax.jacrev(propagator_e.ys[-1,:].flatten(), argnums=0)
        propagator_dU_e = jax.jacrev(propagator_e.ys[-1,:].flatten(), argnums=1)
        propagators['propagator_dX0_e'] = propagator_dX0_e
        propagators['propagator_dU_e'] = propagator_dU_e

        forward_propagation_iterate_e = lambda i, input_dict: forward_propagation_iterate(i, input_dict, propagator_e, propagator_dX0_e, propagator_dU_e, dyn_args, cfg_args)
        backward_propagation_iterate_e = lambda i, input_dict: backward_propagation_iterate(i, input_dict, propagator_e, propagator_dX0_e, propagator_dU_e, dyn_args, cfg_args)

    return eom_e, propagators, forward_propagation_iterate_e, backward_propagation_iterate_e

def prepare_opt_funcs(Boundary_Conds, for_prop_iter, back_prop_iter, dyn_args, cfg_args):

    func = lambda inputs: objective_and_constraints(inputs, Boundary_Conds, for_prop_iter, back_prop_iter, dyn_args, cfg_args)
    vals = jax.jit(jax.block_until_ready(func), backend='cpu')
    grad = jax.jit(jax.block_until_ready(jax.jacrev(func)), backend='cpu')
    sens = jax.jit(jax.block_until_ready(lambda inputs, cvals: grad(inputs)), backend='cpu')

    return vals, grad, sens

def prepare_sol(solution, Sys, Boundary_Conds, popagator, dyn_args, cfg_args):
    
    t_node_bound = dyn_args['t_node_bound']

    # Set up ouput dictionary
    output = {}
    output["Name"] = cfg_args.cfg_name
    output['Orb0'] = {}
    output['Orbf'] = {}
    output['DetTraj'] = {}
    output['dV_det'] = {}

    # Store departure and arrival orbits
    output['Orb0']['X_hst'] = Boundary_Conds['Orb0']['X_hst']
    output['Orb0']['t_hst'] = Boundary_Conds['Orb0']['t_hst']
    output['Orbf']['X_hst'] = Boundary_Conds['Orbf']['X_hst']
    output['Orbf']['t_hst'] = Boundary_Conds['Orbf']['t_hst']

    if cfg_args.det_or_stoch == 'Deterministic':
        controls = solution.xStar['controls'].reshape(cfg_args.nodes, 3)

        X0 = solution.xStar['X0']
        optTraj_X_hst = np.ndarray((1,7))
        optTraj_X_hst[0,:] = X0
        optTraj_t_hst = np.array([t_node_bound[0]])
        for k in range(cfg_args.nodes):
            prop_inputs = {'X0': jnp.array(optTraj_X_hst[-1,:]),
                           'U': jnp.array(controls[k,:]),
                           't0': t_node_bound[k],
                           't1': t_node_bound[k+1]}
            
            sol_fwd = popagator(jnp.array(optTraj_X_hst[-1,:]), jnp.array(controls[k,:]), t_node_bound[k], t_node_bound[k+1], cfg_args)
            optTraj_X_hst = np.vstack([optTraj_X_hst, np.array(sol_fwd.ys[1:,:])])
            optTraj_t_hst = np.hstack([optTraj_t_hst, np.array(sol_fwd.ts[1:])])
    
        
        output['DetTraj']['X_hst'] = optTraj_X_hst
        output['DetTraj']['t_hst'] = optTraj_t_hst
        output['DetTraj']['controls'] = sp.interpolate.interp1d(t_node_bound[:-1], controls.T, kind='previous', fill_value="extrapolate")(optTraj_t_hst).T
        
        dt = t_node_bound[1]-t_node_bound[0]
        dV_det = jnp.sum(jnp.linalg.norm(controls, axis=1)*cfg_args.T_max_nd*dt / optTraj_X_hst[::cfg_args.int_save,-1])*Sys['Vs']
        output['dV_det'] = dV_det

    return output

def save_sol(output, Sys, save_loc: str):
    with h5py.File(save_loc+"data.h5", "w") as f:
        f.create_dataset("Name", data=output['Name'])
        f.create_dataset("Orb0_X_hst", data=output['Orb0']['X_hst'])
        f.create_dataset("Orb0_t_hst", data=output['Orb0']['t_hst'])
        f.create_dataset("Orbf_X_hst", data=output['Orbf']['X_hst'])
        f.create_dataset("Orbf_t_hst", data=output['Orbf']['t_hst'])
        f.create_dataset("DetTraj_X_hst", data=output['DetTraj']['X_hst'])
        f.create_dataset("DetTraj_t_hst", data=output['DetTraj']['t_hst'])
        f.create_dataset("DetTraj_U_hst", data=output['DetTraj']['controls'])
        f.create_dataset("DetTraj_dV", data=output['dV_det'])

    savemat(save_loc+"Sys.mat",Sys)
    return