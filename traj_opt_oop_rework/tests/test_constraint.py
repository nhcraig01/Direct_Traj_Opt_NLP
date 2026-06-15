import jax
import jax.numpy as jnp
import numpy as np

from Lib.dynamics import eoms_gen, objective_and_constraints, propagator_gen
from Lib.utilities import prepare_prop_funcs, process_config, yaml_load
from traj_opt_oop_rework.constraints.constraint import (
    BoundaryConstraint,
    ControlNormConstraint,
    MatchPointConstraint,
)
from traj_opt_oop_rework.dynamics.equations_of_motion import CR3BPDynamics
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.propagators.propagator import StatePropagator

CONFIG_FILE = "Scenarios/Sandbox/config.yaml"


def test_deterministic_constraints_match_legacy():
    config = yaml_load(CONFIG_FILE)
    # Small N_arcs + loose tolerances to keep this (un-jitted, eager) diffrax
    # integration fast - the constraint logic being tested doesn't depend on
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
    dims = problem_def.dims
    bc = problem_def.boundary_conditions

    propagator = StatePropagator(problem_def, CR3BPDynamics(problem_def))

    key = jax.random.PRNGKey(0)
    U_arc_hst = 1e-2 * jax.random.normal(key, (dims.N_arcs, 3))
    X0 = jnp.concatenate([bc.X0_interp.evaluate(alpha), jnp.array([1.0])])
    Xf = jnp.concatenate([bc.Xf_interp.evaluate(beta), jnp.array([0.95])])

    inputs = {'X0': X0, 'Xf': Xf, 'U_arc_hst': U_arc_hst.flatten(), 'alpha': alpha, 'beta': beta}
    traj_state, error_state = propagator.propagate(inputs, problem_def)

    c_X0 = BoundaryConstraint(problem_def, "initial").evaluate(inputs, traj_state, error_state)
    c_Xf = BoundaryConstraint(problem_def, "final").evaluate(inputs, traj_state, error_state)
    c_X_mp = MatchPointConstraint(problem_def).evaluate(inputs, traj_state, error_state)
    c_Us = ControlNormConstraint(problem_def).evaluate(inputs, traj_state, error_state)

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

    np.testing.assert_allclose(np.asarray(c_X0), np.asarray(legacy_out['c_X0']), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(c_Xf), np.asarray(legacy_out['c_Xf']), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(c_X_mp), np.asarray(legacy_out['c_X_mp']), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(c_Us), np.asarray(legacy_out['c_Us']), rtol=1e-10, atol=1e-10)
