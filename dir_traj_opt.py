# Direct Trajectory Optimization

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import numpy as np

from pyoptsparse.pySNOPT.pySNOPT import SNOPT
from pyoptsparse import Optimization

import time
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.stats import chi2, norm

from joblib import Parallel, delayed

import h5py

# Utilities
from Lib.utilities import yaml_load, process_config, process_sparsity, prepare_prop_funcs, prepare_opt_funcs, prepare_sol, save_sol

# Math

# Dynamics
from Lib.dynamics import eoms_gen, propagator_gen


if __name__ == "__main__":

    N_trials = 128

    # Configuration Files
    folder_name = "L2_S-NRHO_to_L2_N-NRHO"
    config_file = r"Scenarios/"+folder_name+"/config.yaml"
    save_file = r"Plotting/Scenarios/"+folder_name+"/deterministic/"

    # Process Configuration - System Constants, optimization arguments, boundary conditions, dynamical eoms, and optimization type
    config = yaml_load(config_file)
    Sys, dynamics, Boundary_Conds, cfg_args, dyn_args, optOptions = process_config(config)

    # Propagation functions
    eom_e, propagators, forward_propagation_iterate_e, backward_propagation_iterate_e = prepare_prop_funcs(eoms_gen, dynamics, propagator_gen, dyn_args, cfg_args)

    # Optimization functions
    vals, grad, sens = prepare_opt_funcs(Boundary_Conds, forward_propagation_iterate_e, backward_propagation_iterate_e, dyn_args, cfg_args)
    
    # Create Sparse Jacobians
    print("Calculating SNOPT Gradient Sparsity")
    grad_nonsparse = grad({'X0': 1*jnp.ones(7), 'Xf': 2*jnp.ones(7), 'controls': .001*jnp.ones(3*cfg_args.nodes), 'alpha': 0.1, 'beta': 0.1})
    grad_proc_sparse = process_sparsity(grad_nonsparse)

    # Optimal Control Problem
    optprop = Optimization("Forward Backward Direct Trajectory Optimization", vals)
    optprop.addVarGroup('controls', 3*cfg_args.nodes, "c", value = .01, lower = -1, upper = 1)
    optprop.addVarGroup('X0', 7, "c", value = jnp.hstack([Boundary_Conds['X0_init'],1]), lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    optprop.addVarGroup('Xf', 7, "c", value = jnp.hstack([Boundary_Conds['Xf_init'],0.95]), lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    if config['boundary_conditions']['type'] == 'free':
        optprop.addVarGroup('alpha', 1, "c", value = Boundary_Conds['alpha_min'], lower = Boundary_Conds['alpha_min'], upper = Boundary_Conds['alpha_max'])
        optprop.addVarGroup('beta', 1, "c", value = Boundary_Conds['beta_min'], lower = Boundary_Conds['beta_min'], upper = Boundary_Conds['beta_max'])

    optprop.addObj('o_mf')

    optprop.addConGroup('c_Us', cfg_args.nodes, upper = 1, jac = grad_proc_sparse['c_Us'])
    optprop.addConGroup('c_X0', 7, lower = 0, upper = 0, jac = grad_proc_sparse['c_X0'])
    optprop.addConGroup('c_Xf', 6, lower = 0, upper = 0, jac = grad_proc_sparse['c_Xf'])
    optprop.addConGroup('c_X_mp', 7, lower = 0, upper = 0, jac = grad_proc_sparse['c_X_mp'])
    if config['constraints']['deterministic']['det_col_avoid']['bool']:
        optprop.addConGroup('c_det_col_avoid', cfg_args.N, upper = 0, jac = grad_proc_sparse['c_det_col_avoid'])
    
    print('SNOPT Starting')
    start_time = time.time()
    optSNOPT = SNOPT(options = optOptions)
    sol = optSNOPT(optprop, sens = sens, timeLimit = None)
    print('SNOPT Finished: %s'%(sol.optInform['text']))
    print("Elapsed Time: %.3f" % (time.time() - start_time))

    optInputs = dict(sol.xStar)
    optObjConst = sol.fStar

    # Re-propagate optimal trajectory for storage
    cfg_args.int_save = 128
    propagator_det = lambda inputs: propagator_gen(inputs, eom_e, dyn_args, cfg_args)
    save_sol(prepare_sol(sol, Sys, Boundary_Conds, propagators['propagator_e'], dyn_args, cfg_args), Sys, save_file)


