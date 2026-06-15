import jax
import jax.numpy as jnp
import numpy as np

from Lib.dynamics import eoms_gen, objective_and_constraints, propagator_gen
from Lib.utilities import prepare_prop_funcs, process_config, yaml_load
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.problems.deterministic_top import DeterministicTOP

CONFIG_FILE = "Scenarios/Sandbox/config.yaml"


def test_deterministic_top_evaluate_matches_legacy():
    config = yaml_load(CONFIG_FILE)
    # Small N_arcs + loose tolerances to keep this (un-jitted, eager) diffrax
    # integration fast - the evaluate() logic being tested doesn't depend on
    # either.
    config['traj_parameters']['control_arcs'] = 2
    config['integration']['a_tol'] = 1e-6
    config['integration']['r_tol'] = 1e-6

    alpha, beta = 0.3, 0.7
    toggles = Toggles(
        problem_type="deterministic",
        measurements=("range", "range-rate", "angles"),
        alpha_rng=(alpha, alpha),
        beta_rng=(beta, beta),
    )
    problem_def = build_problem_def(config, toggles)
    top = DeterministicTOP(problem_def)

    bc = problem_def.boundary_conditions

    key = jax.random.PRNGKey(0)
    U_arc_hst = 1e-2 * jax.random.normal(key, (problem_def.dims.N_arcs, 3))
    X0 = jnp.concatenate([bc.X0_interp.evaluate(alpha), jnp.array([1.0])])
    Xf = jnp.concatenate([bc.Xf_interp.evaluate(beta), jnp.array([0.95])])

    inputs = {'X0': X0, 'Xf': Xf, 'U_arc_hst': U_arc_hst.flatten(), 'alpha': alpha, 'beta': beta}
    out = top.evaluate(inputs)

    assert set(out.keys()) == {'o', 'c_X0', 'c_Xf', 'c_X_mp', 'c_Us'}

    # Legacy
    legacy_config = yaml_load(CONFIG_FILE)
    legacy_config['traj_parameters']['control_arcs'] = 2
    legacy_config['integration']['a_tol'] = 1e-6
    legacy_config['integration']['r_tol'] = 1e-6
    legacy_config['boundary_conditions']['type'] = 'free'
    legacy_config['boundary_conditions']['alpha'] = {'min': 0.0, 'max': 1.0}
    legacy_config['boundary_conditions']['beta'] = {'min': 0.0, 'max': 1.0}

    Sys, models, Boundary_Conds, cfg_args, dyn_args = process_config(
        legacy_config, "deterministic", "true_state", "fulltraj_lqr", "fixed", ("range", "range-rate", "angles"),
    )
    _, propagators, iterators = prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, cfg_args)

    legacy_inputs = {'X0': X0, 'Xf': Xf, 'U_arc_hst': U_arc_hst.flatten(), 'alpha': alpha, 'beta': beta}
    legacy_out = objective_and_constraints(legacy_inputs, Boundary_Conds, iterators, propagators, models, Sys, dyn_args, cfg_args)

    np.testing.assert_allclose(np.asarray(out['o']), np.asarray(legacy_out['o']), rtol=1e-10, atol=1e-10)
    for key in ['c_X0', 'c_Xf', 'c_X_mp', 'c_Us']:
        np.testing.assert_allclose(np.asarray(out[key]), np.asarray(legacy_out[key]), rtol=1e-10, atol=1e-10)


def test_deterministic_top_variables():
    config = yaml_load(CONFIG_FILE)
    toggles = Toggles(problem_type="deterministic", measurements=("range", "range-rate", "angles"))
    problem_def = build_problem_def(config, toggles)
    top = DeterministicTOP(problem_def)

    variables = top.variables()
    names = [v.name for v in variables]
    assert names == ['U_arc_hst', 'X0', 'Xf', 'alpha', 'beta']

    by_name = {v.name: v for v in variables}
    assert by_name['U_arc_hst'].size == 3 * problem_def.dims.N_arcs
    assert by_name['X0'].size == problem_def.dims.state_dim
    assert by_name['Xf'].size == problem_def.dims.state_dim
    assert by_name['alpha'].size == 1
    assert by_name['beta'].size == 1
