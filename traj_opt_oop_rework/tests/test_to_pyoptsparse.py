from Lib.utilities import yaml_load
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def
from traj_opt_oop_rework.problems.deterministic_top import DeterministicTOP

CONFIG_FILE = "Scenarios/Sandbox/config.yaml"


def test_deterministic_to_pyoptsparse_structure():
    config = yaml_load(CONFIG_FILE)
    # Small N_arcs + loose tolerances to keep this (un-jitted, eager) diffrax
    # integration fast - the adapter structure being tested doesn't depend on
    # either.
    config['traj_parameters']['control_arcs'] = 2
    config['integration']['a_tol'] = 1e-6
    config['integration']['r_tol'] = 1e-6

    toggles = Toggles(problem_type="deterministic", measurements=("range", "range-rate", "angles"))
    problem_def = build_problem_def(config, toggles)
    top = DeterministicTOP(problem_def)

    optprob, sens = top.to_pyoptsparse(init_guess={})

    var_sizes = {name: len(vars_) for name, vars_ in optprob.variables.items()}
    assert var_sizes == {
        'U_arc_hst': 3 * problem_def.dims.N_arcs,
        'X0': problem_def.dims.state_dim,
        'Xf': problem_def.dims.state_dim,
        'alpha': 1,
        'beta': 1,
    }

    assert set(optprob.objectives.keys()) == {'o'}

    con_sizes = {name: con.ncon for name, con in optprob.constraints.items()}
    assert con_sizes == {
        'c_X0': problem_def.dims.state_dim,
        'c_Xf': problem_def.dims.state_dim - 1,
        'c_X_mp': problem_def.dims.state_dim,
        'c_Us': problem_def.dims.N_arcs,
    }

    eval_point = {v.name: v.value for v in top.variables()}
    sens(eval_point, {})
