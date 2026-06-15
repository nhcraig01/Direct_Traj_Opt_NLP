import numpy as np

from Lib.utilities import process_config, prepare_prop_funcs, prepare_opt_funcs, yaml_load
from Lib.dynamics import eoms_gen, propagator_gen
from traj_opt_oop_rework.init_guess import hot_start_init_guess
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.problems.deterministic_top import DeterministicTOP

CONFIG_FILE = "Scenarios/L1_N-HO_to_L2_N-HO/config.yaml"
SOL_FILE = "Scenarios/L1_N-HO_to_L2_N-HO/deterministic_sol.h5"


def test_deterministic_evaluate_matches_legacy_at_full_scale():
    """Production config (N_arcs=40, a_tol=r_tol=1e-12), evaluated at a real
    converged solution - the scale at which max_steps/x64 issues actually
    bite, unlike the reduced-scale parity tests."""
    import jax

    config = yaml_load(CONFIG_FILE)
    toggles = Toggles(
        problem_type="deterministic", measurements=("range", "range-rate", "angles"),
        alpha_rng=(0.0, 1.0), beta_rng=(0.0, 1.0),
    )
    problem_def = build_problem_def(config, toggles)
    top = DeterministicTOP(problem_def)

    init_guess = hot_start_init_guess(SOL_FILE, top.variables())
    eval_point = {v.name: init_guess.get(v.name, v.value) for v in top.variables()}

    vals = jax.jit(top.evaluate, backend='cpu')
    out = vals(eval_point)

    Sys, models, Boundary_Conds, cfg_args, dyn_args = process_config(
        config, "deterministic", "true_state", "fulltraj_lqr", "fixed", ("range", "range-rate", "angles"),
    )
    _, propagators, iterators = prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, cfg_args)
    legacy_vals, _, _ = prepare_opt_funcs(Boundary_Conds, iterators, propagators, models, Sys, dyn_args, cfg_args)

    legacy_out = legacy_vals(eval_point)

    for key in ['o', 'c_X0', 'c_Xf', 'c_X_mp', 'c_Us']:
        np.testing.assert_allclose(np.asarray(out[key]), np.asarray(legacy_out[key]), rtol=1e-8, atol=1e-8)
