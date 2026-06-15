import jax.numpy as jnp
import numpy as np

from Lib.dynamics import CR3BPDynamics as legacy_CR3BPDynamics
from Lib.utilities import yaml_load
from traj_opt_oop_rework.dynamics.equations_of_motion import CR3BPDynamics
from traj_opt_oop_rework.problem_definition import Toggles, build_problem_def

CONFIG_FILE = "Scenarios/Sandbox/config.yaml"


def _build_problem_def():
    config = yaml_load(CONFIG_FILE)
    return build_problem_def(config, Toggles())


def test_eom_matches_legacy():
    problem_def = _build_problem_def()

    new_dynamics = CR3BPDynamics(problem_def)

    dyn_safe = 1737.5 / problem_def.Sys['Ls']
    legacy_eom_eval, *_ = legacy_CR3BPDynamics(
        problem_def.spacecraft.U_Acc_min_nd,
        problem_def.spacecraft.ve,
        problem_def.Sys['mu'],
        dyn_safe,
    )

    test_states = [
        jnp.concatenate([problem_def.boundary_conditions.X0_init, jnp.array([1.0])]),
        jnp.concatenate([problem_def.boundary_conditions.Xf_init, jnp.array([0.8])]),
        jnp.array([0.8, 0.1, 0.05, 0.0, 0.2, 0.0, 0.9]),
    ]
    test_controls = [
        jnp.zeros((3,)),
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.3, -0.4, 0.5]),
    ]

    for X in test_states:
        for U in test_controls:
            new_Xdot = new_dynamics.eom(0.0, X, U)
            legacy_Xdot = legacy_eom_eval(0.0, X, U).reshape(-1)
            np.testing.assert_allclose(np.asarray(new_Xdot), np.asarray(legacy_Xdot), rtol=1e-12, atol=1e-12)
