"""Generate detailed-trajectory + Monte Carlo DATA for a solved case.

Second stage of the pipeline (solve -> generate -> plot). Given a case directory
Results/<scenario>/<case>/ holding sol.h5 + case.yaml, it rebuilds the exact
ProblemDefinition/StochasticTOP from the manifest, replays the detailed
deterministic trajectory + orbit histories (src/postprocess/detailed_trajectory.py),
runs the Monte Carlo trials with the runner the feedback type implies, and writes
the grouped data.h5 + Sys.mat into the case dir for plot_traj.py / MATLAB.

Fully OOP - no legacy code: the detailed generator, MC runners and results
writer all live in src/. The MC dispatch and measurement model are automatic,
reconstructed from case.yaml via StochasticTOP.

Run directly (edit the config block to pick a case) or call generate_data(case_dir).
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
from src.utils.io import yaml_load, load_solution, save_results
from src.stochastic.mc_runner import EstimatedStateMCRunner, TrueStateMCRunner
from src.problem.problem_definition import build_problem_def
from src.optimization.trajectory_optimization_problem import DeterministicTOP, StochasticTOP


def generate_data(case_dir, seed=0):
    """Rebuild the problem from case.yaml, propagate + run Monte Carlo, and write
    the grouped data.h5 + Sys.mat into case_dir."""
    scenario, toggles = load_case_spec(case_dir)
    problem_def = build_problem_def(yaml_load(f"Scenarios/{scenario}.yaml"), toggles)

    stochastic = toggles.problem_type.lower() == 'stochastic_gauss_zoh'
    top = StochasticTOP(problem_def) if stochastic else DeterministicTOP(problem_def)
    xStar = load_solution(os.path.join(case_dir, 'sol.h5'))

    print("Evaluating detailed deterministic trajectory...")
    data = generate_detailed_data(problem_def, top, xStar)

    if stochastic:
        det = data['Det']
        t_node_bound = (xStar['t_node_bound'] if toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof'
                        else problem_def.boundary_conditions.t_node_bound)
        mc_sol_data = {'X_node_hst': det['X_node_hst'], 'U_arc_hst': det['U_arc_hst'],
                       'K_arc_hst': det['K_arc_hst'], 't_node_bound': t_node_bound,
                       'dV_mean': det['dV_mean']}
        runner_cls = (EstimatedStateMCRunner if toggles.feedback_control_type.lower() == 'estimated_state'
                      else TrueStateMCRunner)
        print(f"Running Monte Carlo ({problem_def.dims.N_trials} trials)...")
        runner = runner_cls(problem_def, top._propagator, top._error_propagator)
        data['MC_Runs'] = runner.run(mc_sol_data, seed=seed)

    save_results(case_dir, data, problem_def)
    print(f"Saved data.h5 + Sys.mat to {case_dir}")
    return data


if __name__ == "__main__":

    # Case selection (same style as plot_traj.py). The toggles used to rebuild
    # the problem come from the case's case.yaml; these just locate the dir.
    scenario = "L2_S-NRHO_to_L2_N-NRHO"
    Problem_Type = "stochastic_gauss_zoh"          # deterministic | stochastic_gauss_zoh
    Feedback_Control_Type = "true_state"           # true_state | true_state_sqrt | estimated_state
    Measurements = ("range", "range-rate", "angles")

    case = Problem_Type
    if Problem_Type.lower() == 'stochastic_gauss_zoh':
        case += "_" + Feedback_Control_Type
        if Feedback_Control_Type.lower() == 'estimated_state':
            case += "_" + "_".join(Measurements)
    case_dir = f"Results/{scenario}/{case}/"

    print(f"Generating data for: {case_dir}")
    generate_data(case_dir)
