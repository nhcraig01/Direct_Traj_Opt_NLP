import jax
import jax.numpy as jnp
import numpy as np

from Lib.utilities import yaml_load
from traj_opt_oop_rework.init_guess import hot_start_init_guess, random_init_guess
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.problems.deterministic_top import DeterministicTOP

CONFIG_FILE = "Scenarios/Sandbox/config.yaml"


def _build_top():
    config = yaml_load(CONFIG_FILE)
    config['traj_parameters']['control_arcs'] = 2
    config['integration']['a_tol'] = 1e-6
    config['integration']['r_tol'] = 1e-6

    toggles = Toggles(problem_type="deterministic", measurements=("range", "range-rate", "angles"))
    problem_def = build_problem_def(config, toggles)
    return problem_def, DeterministicTOP(problem_def)


def test_random_init_guess_shapes_and_bounds():
    problem_def, top = _build_top()
    variables = top.variables()
    bc = problem_def.boundary_conditions

    init_guess = random_init_guess(variables, jax.random.PRNGKey(0))

    assert set(init_guess.keys()) == {'U_arc_hst', 'X0', 'Xf', 'alpha', 'beta'}

    for v in variables:
        assert init_guess[v.name].shape == (v.size,) or init_guess[v.name].shape == ()

    assert bc.alpha_min <= init_guess['alpha'] <= bc.alpha_max
    assert bc.beta_min <= init_guess['beta'] <= bc.beta_max

    # X0/Xf land on the boundary orbits (with mass appended).
    np.testing.assert_allclose(init_guess['X0'][-1], 1.0)
    np.testing.assert_allclose(init_guess['Xf'][-1], 0.95)


def test_random_init_guess_is_deterministic_for_a_given_key():
    problem_def, top = _build_top()
    variables = top.variables()

    guess_1 = random_init_guess(variables, jax.random.PRNGKey(0))
    guess_2 = random_init_guess(variables, jax.random.PRNGKey(0))

    for name in guess_1:
        np.testing.assert_array_equal(np.asarray(guess_1[name]), np.asarray(guess_2[name]))


def test_hot_start_init_guess_round_trip(tmp_path):
    import h5py

    problem_def, top = _build_top()
    variables = top.variables()

    sol_path = tmp_path / "sol.h5"
    with h5py.File(sol_path, "w") as f:
        f.create_dataset("X0", data=np.array([1., 0., 0., 0., 0., 0., 1.]))
        f.create_dataset("U_arc_hst", data=np.zeros(3 * problem_def.dims.N_arcs))
        f.create_dataset("gain_weights", data=np.ones(4))  # not a DeterministicTOP variable

    init_guess = hot_start_init_guess(str(sol_path), variables)

    assert set(init_guess.keys()) == {'X0', 'U_arc_hst'}
    np.testing.assert_allclose(np.asarray(init_guess['X0']), np.array([1., 0., 0., 0., 0., 0., 1.]))
    np.testing.assert_allclose(np.asarray(init_guess['U_arc_hst']), np.zeros(3 * problem_def.dims.N_arcs))


def test_to_pyoptsparse_accepts_random_init_guess():
    problem_def, top = _build_top()
    variables = top.variables()

    init_guess = random_init_guess(variables, jax.random.PRNGKey(0))
    optprob, sens = top.to_pyoptsparse(init_guess)

    eval_point = {v.name: init_guess.get(v.name, v.value) for v in variables}
    sens(eval_point, {})
