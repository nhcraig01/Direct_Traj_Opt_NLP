import jax
import jax.numpy as jnp
import numpy as np

from Lib.utilities import yaml_load, process_config, prepare_prop_funcs
from traj_opt_oop_rework.dynamics.equations_of_motion import CR3BPDynamics
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.propagators.propagator import StatePropagator

CONFIG_FILE = "Scenarios/Sandbox/config.yaml"


def test_state_propagator_matches_legacy():
    config = yaml_load(CONFIG_FILE)
    # Use a small number of arcs and loose integration tolerances to keep
    # this (un-jitted, eager) diffrax integration fast - the propagation
    # logic being tested doesn't depend on either.
    config['traj_parameters']['control_arcs'] = 2
    config['integration']['a_tol'] = 1e-6
    config['integration']['r_tol'] = 1e-6

    toggles = Toggles(problem_type="deterministic", measurements=("range", "range-rate", "angles"))
    problem_def = build_problem_def(config, toggles)
    dims = problem_def.dims
    bc = problem_def.boundary_conditions

    # New
    propagator = StatePropagator(problem_def, CR3BPDynamics(problem_def))

    key = jax.random.PRNGKey(0)
    U_arc_hst = 1e-2 * jax.random.normal(key, (dims.N_arcs, 3))
    X0 = jnp.concatenate([bc.X0_init, jnp.array([1.0])])
    Xf = jnp.concatenate([bc.Xf_init, jnp.array([0.95])])

    inputs = {'X0': X0, 'Xf': Xf, 'U_arc_hst': U_arc_hst.flatten()}
    traj_state, error_state = propagator.propagate(inputs, problem_def)

    assert error_state is None
    assert traj_state.X_hst.shape == (dims.N_arcs, dims.N_subarcs + 1, dims.state_dim)
    assert traj_state.t_hst.shape == (dims.N_arcs, dims.N_subarcs + 1)

    # Legacy - process_config() still expects boundary_conditions.type, which
    # was removed from the Sandbox config.yaml in favor of toggles.alpha_rng/
    # beta_rng (see problem_definition.py); set it directly for this comparison.
    legacy_config = yaml_load(CONFIG_FILE)
    legacy_config['traj_parameters']['control_arcs'] = 2
    legacy_config['integration']['a_tol'] = 1e-6
    legacy_config['integration']['r_tol'] = 1e-6
    legacy_config['boundary_conditions']['type'] = 'fixed'

    Sys, models, Boundary_Conds, cfg_args, dyn_args = process_config(
        legacy_config, "deterministic", "true_state", "fulltraj_lqr", "fixed", ("range", "range-rate", "angles"),
    )
    from Lib.dynamics import eoms_gen, propagator_gen
    _, propagators, iterators = prepare_prop_funcs(eoms_gen, models, propagator_gen, dyn_args, cfg_args)

    X_hst_legacy = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, 7))
    t_node_bound = dyn_args['t_node_bound']

    forward_input_dict = {'X0_true_f': X0, 'X_hst': X_hst_legacy, 'U_arc_hst': U_arc_hst, 't_node_bound': t_node_bound}
    forward_out = jax.lax.fori_loop(0, len(dyn_args['indx_f']), iterators['forward_propagation_iterate_e'], forward_input_dict)
    X_hst_legacy = forward_out['X_hst']

    backward_input_dict = {'X0_true_b': Xf, 'X_hst': X_hst_legacy, 'U_arc_hst': U_arc_hst, 't_node_bound': t_node_bound}
    backward_out = jax.lax.fori_loop(0, len(dyn_args['indx_b']), iterators['backward_propagation_iterate_e'], backward_input_dict)
    X_hst_legacy = backward_out['X_hst']

    np.testing.assert_allclose(np.asarray(traj_state.X_hst), np.asarray(X_hst_legacy), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(traj_state.t_node_bound), np.asarray(t_node_bound), rtol=1e-12, atol=1e-12)
