# Deterministic SNOPT benchmark: new traj_opt_oop_rework vs legacy dir_traj_opt
#
# Both pipelines are given the SAME non-converged initial guess (generated via
# traj_opt_oop_rework.init_guess.random_init_guess) and run through SNOPT with
# the same options as dir_traj_opt.py. Reports total/major/minor iteration
# timing, compares the resulting solutions, and (if they match) runs the
# legacy detailed-propagation post-processing on both solutions for MATLAB
# plotting.
#
# Usage: python benchmark_deterministic.py legacy|new
#   - "legacy" generates the shared init guess (saved to /tmp/benchmark_init_guess.npz),
#     runs the legacy SNOPT problem, and saves its solution to /tmp/benchmark_legacy_sol.npz.
#   - "new" loads the saved init guess, runs the new-code SNOPT problem, and
#     compares against the saved legacy solution (if present).
#
# NOTE: the new DeterministicTOP does not yet implement the
# c_det_col_avoid constraint group (Sandbox config has
# constraints.col_avoid.det.bool: True). For an apples-to-apples comparison,
# this benchmark omits c_det_col_avoid from the legacy SNOPT problem too.

import os
import sys
import time
from dataclasses import replace

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

import numpy as np
from pyoptsparse import Optimization
from pyoptsparse.pySNOPT.pySNOPT import SNOPT

from Lib.utilities import (
    yaml_load, process_config, process_sparsity, prepare_prop_funcs, prepare_opt_funcs,
    prepare_sol, save_sol, save_OptimizerSol,
)
from Lib.dynamics import eoms_gen, propagator_gen

from traj_opt_oop_rework.init_guess import random_init_guess
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.problems.deterministic_top import DeterministicTOP


FOLDER_NAME = "Sandbox"
CONFIG_FILE = f"Scenarios/{FOLDER_NAME}/config.yaml"
MEASUREMENTS = ("range", "range-rate", "angles")
SEED = 1

INIT_GUESS_FILE = "/tmp/benchmark_init_guess.npz"
LEGACY_SOL_FILE = "/tmp/benchmark_legacy_sol.npz"
NEW_SOL_FILE = "/tmp/benchmark_new_sol.npz"

# Same SNOPT options as dir_traj_opt.py, with hot_start_file is None,
# Major iterations limit raised to 10000 per user request.
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

VAR_NAMES = ['U_arc_hst', 'X0', 'Xf', 'alpha', 'beta']


def get_init_guess():
    if os.path.exists(INIT_GUESS_FILE):
        data = np.load(INIT_GUESS_FILE)
        return {name: jax.numpy.asarray(data[name]) for name in VAR_NAMES}

    config = yaml_load(CONFIG_FILE)
    toggles = Toggles(
        problem_type="deterministic", measurements=MEASUREMENTS,
        alpha_rng=(0.0, 1.0), beta_rng=(0.0, 1.0),
    )
    problem_def = build_problem_def(config, toggles)
    top = DeterministicTOP(problem_def)

    key = jax.random.PRNGKey(SEED)
    init_guess = random_init_guess(top.variables(), key)

    np.savez(INIT_GUESS_FILE, **{name: np.asarray(init_guess[name]) for name in VAR_NAMES})
    return init_guess


def report(label, sol, total_time):
    n_minor = sol.userObjCalls
    n_major = sol.userSensCalls
    print(f"\n--- {label} ---")
    print(f"optInform: {sol.optInform}")
    print(f"fStar: {sol.fStar}")
    print(f"total SNOPT time: {total_time:.3f} s")
    print(f"minor iterations (userObjCalls): {n_minor}  ->  {total_time / n_minor:.4f} s/iter")
    print(f"major iterations (userSensCalls): {n_major}  ->  {total_time / n_major:.4f} s/major iter")


def save_sol_xstar(path, sol):
    np.savez(path, **{name: np.asarray(sol.xStar[name]) for name in VAR_NAMES},
             fStar=np.asarray(sol.fStar), total_time=sol._total_time,
             userObjCalls=sol.userObjCalls, userSensCalls=sol.userSensCalls,
             optInform_value=sol.optInform['value'])


def build_legacy_data():
    legacy_config = yaml_load(CONFIG_FILE)
    legacy_config['boundary_conditions']['type'] = 'free'
    legacy_config['boundary_conditions']['alpha'] = {'min': 0.0, 'max': 1.0}
    legacy_config['boundary_conditions']['beta'] = {'min': 0.0, 'max': 1.0}

    Sys, models, Boundary_Conds, cfg_args, dyn_args = process_config(
        legacy_config, "deterministic", "true_state", "fulltraj_lqr", "fixed", MEASUREMENTS,
    )

    cfg_args_opt = replace(cfg_args, N_save=2)
    _, propagators, iterators = prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, cfg_args_opt)
    vals, grad, sens_legacy = prepare_opt_funcs(Boundary_Conds, iterators, propagators, models, Sys, dyn_args, cfg_args_opt)

    return Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args, vals, grad, sens_legacy


def run_legacy(init_guess):
    Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args, vals, grad, sens_legacy = build_legacy_data()

    print("\nProcessing legacy SNOPT gradient sparsity...")
    grad_proc_sparse = process_sparsity(grad(init_guess))

    optprop_legacy = Optimization("Forward Backward Direct Trajectory Optimization", vals)
    optprop_legacy.addVarGroup('U_arc_hst', 3 * cfg_args.N_arcs, "c", value=init_guess['U_arc_hst'], lower=-1, upper=1)
    optprop_legacy.addVarGroup('X0', 7, "c", value=init_guess['X0'], lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    optprop_legacy.addVarGroup('Xf', 7, "c", value=init_guess['Xf'], lower=[-10, -10, -10, -10, -10, -10, 1e-1], upper=[10, 10, 10, 10, 10, 10, 1])
    optprop_legacy.addVarGroup('alpha', 1, "c", value=init_guess['alpha'], lower=Boundary_Conds['alpha_min'], upper=Boundary_Conds['alpha_max'])
    optprop_legacy.addVarGroup('beta', 1, "c", value=init_guess['beta'], lower=Boundary_Conds['beta_min'], upper=Boundary_Conds['beta_max'])

    optprop_legacy.addObj('o')

    optprop_legacy.addConGroup('c_Us', cfg_args.N_arcs, upper=1, jac=grad_proc_sparse['c_Us'])
    optprop_legacy.addConGroup('c_X0', 7, lower=0, upper=0, jac=grad_proc_sparse['c_X0'])
    optprop_legacy.addConGroup('c_Xf', 6, lower=0, upper=0, jac=grad_proc_sparse['c_Xf'])
    optprop_legacy.addConGroup('c_X_mp', 7, lower=0, upper=0, jac=grad_proc_sparse['c_X_mp'])

    print("Legacy SNOPT starting...")
    optSNOPT_legacy = SNOPT(options=dict(optOptions))
    t0 = time.time()
    sol_legacy = optSNOPT_legacy(optprop_legacy, sens=sens_legacy, timeLimit=None)
    legacy_time = time.time() - t0
    sol_legacy._total_time = legacy_time

    report("Legacy (dir_traj_opt)", sol_legacy, legacy_time)
    save_sol_xstar(LEGACY_SOL_FILE, sol_legacy)
    print(f"\nSaved legacy solution to {LEGACY_SOL_FILE}")

    return sol_legacy, (Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args)


def run_new(init_guess):
    config = yaml_load(CONFIG_FILE)
    toggles = Toggles(
        problem_type="deterministic", measurements=MEASUREMENTS,
        alpha_rng=(0.0, 1.0), beta_rng=(0.0, 1.0),
    )
    problem_def = build_problem_def(config, toggles)
    top = DeterministicTOP(problem_def)

    print("\nBuilding new-code optimization problem...")
    optprob_new, sens_new = top.to_pyoptsparse(init_guess)

    print("New code SNOPT starting...")
    optSNOPT_new = SNOPT(options=dict(optOptions))
    t0 = time.time()
    sol_new = optSNOPT_new(optprob_new, sens=sens_new, timeLimit=None)
    new_time = time.time() - t0
    sol_new._total_time = new_time

    report("New (traj_opt_oop_rework)", sol_new, new_time)
    save_sol_xstar(NEW_SOL_FILE, sol_new)
    print(f"\nSaved new-code solution to {NEW_SOL_FILE}")

    return sol_new


def compare_and_propagate(sol_new, legacy_data=None):
    if not os.path.exists(LEGACY_SOL_FILE):
        print("No legacy solution found to compare against.")
        return

    legacy_xstar = np.load(LEGACY_SOL_FILE)

    print("\n--- Solution comparison ---")
    match = True
    for name in VAR_NAMES:
        a = np.asarray(sol_new.xStar[name])
        b = legacy_xstar[name]
        max_diff = np.max(np.abs(a - b))
        close = np.allclose(a, b, rtol=1e-6, atol=1e-6)
        match &= close
        print(f"{name}: max|diff| = {max_diff:.3e}, close = {close}")

    f_new = float(np.asarray(sol_new.fStar).flatten()[0])
    f_legacy = float(legacy_xstar['fStar'].flatten()[0])
    print(f"Objective: new = {f_new:.8f}, legacy = {f_legacy:.8f}, diff = {abs(f_new - f_legacy):.3e}")
    print(f"\nSolutions match: {match}")

    if match and legacy_data is not None:
        print("\nSolutions match - running detailed propagation for both...")
        Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args = legacy_data

        class _SolLike:
            def __init__(self, xStar):
                self.xStar = xStar

        legacy_sol_like = _SolLike({name: legacy_xstar[name] for name in VAR_NAMES})

        for label, sol in [("new", sol_new), ("legacy", legacy_sol_like)]:
            save_dir = f"Plotting/Scenarios/{FOLDER_NAME}/deterministic_{label}/"
            os.makedirs(save_dir, exist_ok=True)
            allData = prepare_sol(sol, Sys, Boundary_Conds, propagators, models, dyn_args, cfg_args)
            save_sol(allData, Sys, save_dir, dyn_args, cfg_args)
            save_OptimizerSol(sol, cfg_args, dyn_args, save_dir + f"{label}_sol.h5")
            print(f"  {label}: saved to {save_dir}")
    elif legacy_data is None:
        print("\n(legacy propagators not rebuilt in this run - rerun with both legacy_data available to propagate)")
    else:
        print("\nSolutions do not match - skipping detailed propagation.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "legacy"

    print("Loading/generating shared initial guess...")
    init_guess = get_init_guess()
    print("Initial guess:")
    for k, v in init_guess.items():
        print(f"  {k}: shape={np.asarray(v).shape}")

    if which == "legacy":
        run_legacy(init_guess)
    elif which == "new":
        sol_new = run_new(init_guess)
        legacy_data = build_legacy_data()[:6]
        compare_and_propagate(sol_new, legacy_data=legacy_data)
    else:
        raise ValueError(f"unknown mode: {which!r}, expected 'legacy' or 'new'")
