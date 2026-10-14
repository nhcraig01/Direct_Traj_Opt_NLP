"""Generate detailed-trajectory + Monte Carlo DATA for a solved case.

Second stage of the pipeline (solve -> generate -> plot). generate_data() is
handed a solution dict + the case.yaml toggles that go with it, rebuilds the
exact ProblemDefinition/StochasticTOP, replays the detailed deterministic
trajectory + orbit histories (src/postprocess/detailed_trajectory.py), runs the
Monte Carlo trials with the runner the feedback type implies, and writes the
grouped data.h5 + Sys.mat into case_dir for plot_traj.py / MATLAB.

It does no loading of its own - reading sol.h5/case.yaml off disk is the
caller's job (see __main__ below). That lets dir_traj_opt.py pass the solution
it already has in memory, solved or not, and lets a solution be run through a
DIFFERENT formulation's data generation (e.g. a true_state solution through the
estimated_state Monte Carlo) without staging files to match.

Fully OOP - no legacy code: the detailed generator, MC runners and results
writer all live in src/. The MC dispatch and measurement model are automatic,
reconstructed from case.yaml via StochasticTOP.

Run directly (edit the config block to pick a case) or call generate_data().
"""

import os
import sys

# Ensure the repo root (this script's directory) is on sys.path and is the cwd
# so scenario yamls resolve their data/... paths and Results/... reads/writes
# are root-relative.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

from src.problem.case_manifest import load_case_spec
from src.postprocess.detailed_trajectory import generate_detailed_data
from src.utils.console import banner, saved, step
from src.utils.io import yaml_load, load_solution, save_results
from src.stochastic.mc_runner import EstimatedStateMCRunner, TrueStateMCRunner
from src.problem.problem_definition import build_problem_def
from src.optimization.trajectory_optimization_problem import DeterministicTOP, StochasticTOP


def generate_data(xStar, scenario, toggles, case_dir, seed=0, solve_stats=True):
    """Propagate + run Monte Carlo for `xStar` under `toggles`, and write the
    grouped data.h5 + Sys.mat into case_dir.

    xStar/toggles are passed in rather than read from case_dir, so the solution
    need not be the one sitting in that directory. solve_stats=False NaN-fills
    the solve_stats group, for a solution that wasn't produced by a solve there.
    """
    problem_def = build_problem_def(yaml_load(f"Scenarios/{scenario}.yaml"), toggles)

    stochastic = toggles.problem_type.lower() == 'stochastic_gauss_zoh'
    top = StochasticTOP(problem_def) if stochastic else DeterministicTOP(problem_def)

    print("\n  generating data")
    with step("detailed trajectory", indent=4):
        data = generate_detailed_data(problem_def, top, xStar)

    if stochastic:
        det = data['Det']
        runner_cls = (EstimatedStateMCRunner
                      if toggles.feedback_control_type.lower() in ('estimated_state', 'estimated_state_sqrt')
                      else TrueStateMCRunner)
        runner = runner_cls(problem_def, top._propagator, top._error_propagator, top._control_repr)
        data['MC_Runs'] = runner.run(det, seed=seed)  # prints its own progress bar

    # data.h5 + Sys.mat; the caller reports what was saved.
    save_results(case_dir, data, problem_def, solve_stats=solve_stats)
    return data


if __name__ == "__main__":

    # Case selection (same style as plot_traj.py). The toggles used to rebuild
    # the problem come from the case's case.yaml; these just locate the dir.
    scenario = "L2_S-NRHO_to_L2_N-NRHO"
    Problem_Type = "stochastic_gauss_zoh"          # deterministic | stochastic_gauss_zoh
    Feedback_Control_Type = "true_state"           # true_state | true_state_sqrt | estimated_state | estimated_state_sqrt
    Measurements = ("range", "range-rate", "angles")

    case = Problem_Type
    if Problem_Type.lower() == 'stochastic_gauss_zoh':
        case += "_" + Feedback_Control_Type
        if Feedback_Control_Type.lower() in ('estimated_state', 'estimated_state_sqrt'):
            case += "_" + "_".join(Measurements)
    case_dir = f"Results/{scenario}/{case}/"

    # The reading lives here, not in generate_data().
    scenario_name, toggles = load_case_spec(case_dir)
    xStar = load_solution(os.path.join(case_dir, 'sol.h5'))
    banner(scenario_name, toggles, [('data', "regenerating from the case's sol.h5")])
    generate_data(xStar, scenario_name, toggles, case_dir)
    saved(case_dir, ['data.h5', 'Sys.mat'])
