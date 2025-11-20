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
from Lib.dynamics import eoms_gen, propagator_gen


if __name__ == "__main__":

    # Configuration Files and Problem Type --------------------------------------
    # Scenario Folder Names:
    #   L2_S-NRHO_to_L2_N-NRHO
    #   L1_N-HO_to_L2_N-HO
    #   Sandbox
    # 
    # Problem Types:
    #   deterministic
    #   stochastic_gauss_zoh
    # 
    # Feedback Controller Types:
    #   true_state
    #   estimated_state
    # ---------------------------------------------------------------------------
    folder_name = "Sandbox"
    Problem_Type = "stochastic_gauss_zoh"
    Feedback_Control_Type = "true_state"

    hot_start = True
    hot_start_sol = "deterministic"
    # ---------------------------------------------------------------------------
    file_name = Problem_Type
    if Problem_Type.lower() == 'stochastic_gauss_zoh': 
        file_name += "_" + Feedback_Control_Type
    config_file = r"Scenarios/"+folder_name+"/config.yaml"
    hot_start_file = r"Scenarios/"+folder_name+"/"+hot_start_sol+"_sol.h5"
    save_file = r"Plotting/Scenarios/"+folder_name+"/"+file_name+"/"
    OptimSol_save_file = r"Scenarios/"+folder_name+"/"+file_name+"_sol.h5"
    # ---------------------------------------------------------------------------

    # SNOPT Options -------------------------------------------------------------
    optOptions = {'Major optimality tolerance': 1e-5,  # Pretty much always keep this at 1.e-5 (linesearch_tol is more important)
                  'Major feasibility tolerance': 1e-6,  
                  'Minor feasibility tolerance': 1e-5,
                  'Major iterations limit': 1000, 
                  'Partial prince': 10,                 # Maybe just keep at 1
                  'Linesearch tolerance': .1,           # .5 for deterministic, .01 for stochastic
                  'Function precision': 1e-12,
                  'Verify level': -1,
                  'Nonderivative linesearch': 0,
                  'Elastic weight': 1.e6}
    # ---------------------------------------------------------------------------


    # Process Configuration - System Constants, optimization arguments, boundary conditions, dynamical eoms, and optimization type
    config = yaml_load(config_file)
    Sys, models, Boundary_Conds, cfg_args, dyn_args = process_config(config, Problem_Type, Feedback_Control_Type)

    # Propagation functions
    eom_e, propagators, iterators = prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, replace(cfg_args, N_save=2))

    # Optimization functions
    vals, grad, sens = prepare_opt_funcs(Boundary_Conds, iterators, propagators, dyn_args, replace(cfg_args, N_save=2))

    # Set up initial guess and hot starter
    print("Setting Up Initial Guess")
    init_guess = {'U_arc_hst': 1e-2*jnp.ones(3*cfg_args.N_arcs), 
                  'X0': jnp.hstack([Boundary_Conds['X0_init'],1]), 
                  'Xf': jnp.hstack([Boundary_Conds['Xf_init'],0.95])}
    if cfg_args.free_phasing:
        init_guess['alpha'] = Boundary_Conds['alpha_min']
        init_guess['beta'] = Boundary_Conds['beta_min']
    if Problem_Type == 'stochastic_gauss_zoh':
        init_guess['xi_arc_hst'] = 1e-1*jnp.ones(2*cfg_args.N_arcs)
    if hot_start:
        sol_hot = load_OptimizerSol(hot_start_file)
        for key in sol_hot.keys():
            init_guess[key] = sol_hot[key] 

    # Process Sparsity for SNOPT
    print("Processing SNOPT Gradient Sparsity")
    grad_nonsparse = grad(init_guess)
    grad_proc_sparse = process_sparsity(grad_nonsparse)
    

    # Optimal Control Problem
    optprop = Optimization("Forward Backward Direct Trajectory Optimization", vals)
    optprop.addVarGroup('U_arc_hst', 3*cfg_args.N_arcs, "c", value = init_guess['U_arc_hst'], lower = -1, upper = 1)
    optprop.addVarGroup('X0', 7, "c", value = init_guess['X0'], lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    optprop.addVarGroup('Xf', 7, "c", value = init_guess['Xf'], lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    if Problem_Type == 'stochastic_gauss_zoh':
        optprop.addVarGroup('xi_arc_hst', 2*cfg_args.N_arcs, "c", value = init_guess['xi_arc_hst'], lower = 1e-5)
    if cfg_args.free_phasing:
        optprop.addVarGroup('alpha', 1, "c", value = init_guess['alpha'], lower = Boundary_Conds['alpha_min'], upper = Boundary_Conds['alpha_max'])
        optprop.addVarGroup('beta', 1, "c", value = init_guess['beta'], lower = Boundary_Conds['beta_min'], upper = Boundary_Conds['beta_max'])

    optprop.addObj('o_mf')

    optprop.addConGroup('c_Us', cfg_args.N_arcs, upper = 1, jac = grad_proc_sparse['c_Us'])
    if Problem_Type == 'stochastic_gauss_zoh':
        optprop.addConGroup('c_P_Xf', 1, upper = 0, jac = grad_proc_sparse['c_P_Xf'])
    optprop.addConGroup('c_X0', 7, lower = 0, upper = 0, jac = grad_proc_sparse['c_X0'])
    optprop.addConGroup('c_Xf', 6, lower = 0, upper = 0, jac = grad_proc_sparse['c_Xf'])
    optprop.addConGroup('c_X_mp', 7, lower = 0, upper = 0, jac = grad_proc_sparse['c_X_mp'])
    if cfg_args.det_col_avoid and not cfg_args.stat_col_avoid:
        optprop.addConGroup('c_det_col_avoid', cfg_args.N_nodes, upper = 0, jac = grad_proc_sparse['c_det_col_avoid'])
    if cfg_args.stat_col_avoid and Problem_Type != 'deterministic':
        optprop.addConGroup('c_stat_col_avoid', cfg_args.N_nodes, upper = 0, jac = grad_proc_sparse['c_stat_col_avoid'])
    
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


