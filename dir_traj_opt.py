# Direct Trajectory Optimization

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from pyoptsparse.pySNOPT.pySNOPT import SNOPT
from pyoptsparse import Optimization

import time

from dataclasses import replace

# Utilities
from Lib.utilities import yaml_load, process_config, process_sparsity, prepare_prop_funcs, prepare_opt_funcs, prepare_sol, save_sol, save_OptimizerSol, load_OptimizerSol

# Math

# Dynamics
from Lib.dynamics import eoms_gen, propagator_gen, propagator_gen_fin


if __name__ == "__main__":

    # Configuration Files -------------------------------------------------------
    folder_name = "L1_N-HO_to_L2_N-HO"
    Problem_Type = "stochastic_gauss_zoh"       # "deterministic" or "stochastic_gauss_zoh"
    hot_start = True                            # True or False

    # ---------------------------------------------------------------------------
    config_file = r"Scenarios/"+folder_name+"/config.yaml"
    hot_start_file = r"Scenarios/"+folder_name+"/deterministic_sol.h5"

    save_file = r"Plotting/Scenarios/"+folder_name+"/"+Problem_Type+"/"
    OptimSol_save_file = r"Scenarios/"+folder_name+"/"+Problem_Type+"_sol.h5"
    # ---------------------------------------------------------------------------


    # Process Configuration - System Constants, optimization arguments, boundary conditions, dynamical eoms, and optimization type
    config = yaml_load(config_file)
    Sys, dynamics, Boundary_Conds, cfg_args, dyn_args, optOptions = process_config(config, Problem_Type)

    # Propagation functions
    eom_e, propagators, iterators = prepare_prop_funcs(eoms_gen, dynamics, propagator_gen, propagator_gen_fin, dyn_args, replace(cfg_args, int_save=2))

    # Optimization functions
    vals, grad, sens = prepare_opt_funcs(Boundary_Conds, iterators, dyn_args, replace(cfg_args, int_save=2))
    
    # Create Sparse Jacobians
    print("Calculating SNOPT Gradient Sparsity")
    sparse_inputs = {'X0': 1*jnp.ones(7),
                     'Xf': 2*jnp.ones(7),
                    'controls': .01*jnp.ones(3*cfg_args.nodes)}
    if cfg_args.free_phasing:
        sparse_inputs['alpha'] = 0.1
        sparse_inputs['beta'] = 0.1
    if Problem_Type == 'stochastic_gauss_zoh':
        sparse_inputs['xis'] = 0.01*jnp.ones(2*cfg_args.nodes)
    grad_nonsparse = grad(sparse_inputs)
    grad_proc_sparse = process_sparsity(grad_nonsparse)


    # Load Hot Starter Solution
    print("Setting Up Initial Guess")
    init_guess = {'controls': 0.01*jnp.ones(3*cfg_args.nodes), 
                  'X0': jnp.hstack([Boundary_Conds['X0_init'],1]), 
                  'Xf': jnp.hstack([Boundary_Conds['Xf_init'],0.95])}
    if cfg_args.free_phasing:
        init_guess['alpha'] = Boundary_Conds['alpha_min']
        init_guess['beta'] = Boundary_Conds['beta_min']
    if Problem_Type == 'stochastic_gauss_zoh':
        init_guess['xis'] = 1e-4*jnp.ones(2*cfg_args.nodes)
    if hot_start:
        sol_hot = load_OptimizerSol(hot_start_file)
        for key in sol_hot.keys():
            init_guess[key] = sol_hot[key] 
    

    # Optimal Control Problem
    optprop = Optimization("Forward Backward Direct Trajectory Optimization", vals)
    optprop.addVarGroup('X0', 7, "c", value = init_guess['X0'], lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    optprop.addVarGroup('Xf', 7, "c", value = init_guess['Xf'], lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    optprop.addVarGroup('controls', 3*cfg_args.nodes, "c", value = init_guess['controls'], lower = -1, upper = 1)
    if Problem_Type == 'stochastic_gauss_zoh':
        optprop.addVarGroup('xis', 2*cfg_args.nodes, "c", value = init_guess['xis'], lower = 1e-5)
    if cfg_args.free_phasing:
        optprop.addVarGroup('alpha', 1, "c", value = init_guess['alpha'], lower = Boundary_Conds['alpha_min'], upper = Boundary_Conds['alpha_max'])
        optprop.addVarGroup('beta', 1, "c", value = init_guess['beta'], lower = Boundary_Conds['beta_min'], upper = Boundary_Conds['beta_max'])

    optprop.addObj('o_mf')

    optprop.addConGroup('c_Us', cfg_args.nodes, upper = 1, jac = grad_proc_sparse['c_Us'])
    if Problem_Type == 'stochastic_gauss_zoh':
        optprop.addConGroup('c_P_Xf', 1, upper = 0, jac = grad_proc_sparse['c_P_Xf'])
    optprop.addConGroup('c_X0', 7, lower = 0, upper = 0, jac = grad_proc_sparse['c_X0'])
    optprop.addConGroup('c_Xf', 6, lower = 0, upper = 0, jac = grad_proc_sparse['c_Xf'])
    optprop.addConGroup('c_X_mp', 7, lower = 0, upper = 0, jac = grad_proc_sparse['c_X_mp'])
    if cfg_args.det_col_avoid:
        optprop.addConGroup('c_det_col_avoid', cfg_args.N, upper = 0, jac = grad_proc_sparse['c_det_col_avoid'])
    
    print('SNOPT Starting')
    start_time = time.time()
    optSNOPT = SNOPT(options = optOptions)
    sol = optSNOPT(optprop, sens = sens, timeLimit = None)
    print('SNOPT Finished: %s'%(sol.optInform['text']))
    print("Elapsed Time: %.3f" % (time.time() - start_time))

    # Save Optimization Solution
    save_OptimizerSol(sol, cfg_args, OptimSol_save_file)

    # Analyze and Save Results
    allData = prepare_sol(sol, Sys, Boundary_Conds, propagators, dyn_args, cfg_args)
    save_sol(allData, Sys, save_file,dyn_args, cfg_args)


