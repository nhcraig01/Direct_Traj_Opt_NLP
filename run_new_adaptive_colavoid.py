# New-code (traj_opt_oop_rework) SNOPT run with adaptive_fixedtof mesh and
# deterministic collision avoidance (c_det_col_avoid, enabled via Sandbox
# config's constraints.col_avoid.det.bool: True), saving detailed trajectory
# data for MATLAB plotting.

import os
import time
from dataclasses import replace

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

from pyoptsparse.pySNOPT.pySNOPT import SNOPT

from Lib.utilities import yaml_load, process_config, prepare_prop_funcs, prepare_sol, save_sol, save_OptimizerSol
from Lib.dynamics import eoms_gen, propagator_gen

from traj_opt_oop_rework.init_guess import random_init_guess
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.problems.deterministic_top import DeterministicTOP


FOLDER_NAME = "Sandbox"
CONFIG_FILE = f"Scenarios/{FOLDER_NAME}/config.yaml"
MEASUREMENTS = ("range", "range-rate", "angles")
SEED = 1
SAVE_DIR = f"Plotting/Scenarios/{FOLDER_NAME}/deterministic_adaptive_colavoid/"

optOptions = {'Major optimality tolerance': 1e-5,
               'Major feasibility tolerance': 1e-6,
               'Minor feasibility tolerance': 1e-6,
               'Major iterations limit': 10000,
               'Partial prince': 1,
               'Linesearch tolerance': .99,
               'Function precision': 1e-10,
               'Verify level': -1,
               'Nonderivative linesearch': 0,
               'Major step limit': 1e0,
               'Elastic weight': 1.e4}


print("Building new-code problem (adaptive_fixedtof mesh, det_col_avoid)...")
config = yaml_load(CONFIG_FILE)
toggles = Toggles(
    problem_type="deterministic", measurements=MEASUREMENTS,
    alpha_rng=(0.0, 1.0), beta_rng=(0.0, 1.0), adaptive_mesh_type="adaptive_fixedtof",
)
problem_def = build_problem_def(config, toggles)
top = DeterministicTOP(problem_def)
print("toggles:", problem_def.toggles)
print("variables:", [v.name for v in top.variables()])
print("constraints:", [c.name for c in top.constraints()])

init_guess = random_init_guess(top.variables(), jax.random.PRNGKey(SEED))
optprob, sens = top.to_pyoptsparse(init_guess)

print("\nSNOPT starting...")
optSNOPT = SNOPT(options=dict(optOptions))
t0 = time.time()
sol = optSNOPT(optprob, sens=sens, timeLimit=None)
total_time = time.time() - t0

print(f"\noptInform: {sol.optInform}")
print(f"fStar: {sol.fStar}")
print(f"total SNOPT time: {total_time:.3f} s")
n_minor = sol.userObjCalls
n_major = sol.userSensCalls
print(f"minor iterations (userObjCalls): {n_minor}  ->  {total_time / n_minor:.4f} s/iter")
print(f"major iterations (userSensCalls): {n_major}  ->  {total_time / n_major:.4f} s/major iter")

if sol.optInform['value'] not in (1,):
    print("\nWARNING: optimization did not report full convergence.")

# Build legacy propagation infrastructure for prepare_sol/save_sol
print("\nBuilding detailed-propagation infrastructure...")
legacy_config = yaml_load(CONFIG_FILE)
legacy_config['boundary_conditions']['type'] = 'free'
legacy_config['boundary_conditions']['alpha'] = {'min': 0.0, 'max': 1.0}
legacy_config['boundary_conditions']['beta'] = {'min': 0.0, 'max': 1.0}

Sys, models, Boundary_Conds, cfg_args, dyn_args = process_config(
    legacy_config, "deterministic", "true_state", "fulltraj_lqr", "adaptive_fixedtof", MEASUREMENTS,
)
cfg_args_opt = replace(cfg_args, N_save=2)
_, propagators, _ = prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, cfg_args_opt)

os.makedirs(SAVE_DIR, exist_ok=True)
allData = prepare_sol(sol, Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args)
save_sol(allData, Sys, SAVE_DIR, dyn_args, cfg_args)
save_OptimizerSol(sol, cfg_args, dyn_args, SAVE_DIR + "adaptive_colavoid_sol.h5")
print(f"\nSaved to {SAVE_DIR}")
