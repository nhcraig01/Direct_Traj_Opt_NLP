# Direct Trajectory Optimization - solve entry point.
#
# Builds a ProblemDefinition + TrajectoryOptimizationProblem from a scenario
# yaml, runs SNOPT, and writes the optimizer solution (sol.h5) plus its case
# manifest (case.yaml) into Results/<scenario>/<case>/. That's the whole job:
# the detailed-trajectory + Monte Carlo DATA generation is a separate step
# (generate_data.py, which rebuilds the exact problem from case.yaml), and
# plotting is plot_traj.py. Set Generate_Data = True below to chain the data
# step right after solving.

import os
import sys
import time

# Ensure the repo root (this script's directory) is on sys.path and is the cwd
# so scenario yamls resolve their data/... paths and Results/... outputs land
# correctly regardless of how this is launched (terminal, IDE run button,
# interactive window).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

from pyoptsparse.pySNOPT.pySNOPT import SNOPT

from src.utils.io import yaml_load, save_solution
from src.problem.case_manifest import write_case_yaml
from src.optimization.init_guess import hot_start_init_guess, random_init_guess
from src.problem.problem_definition import Toggles, build_problem_def
from src.optimization.trajectory_optimization_problem import DeterministicTOP, StochasticTOP


if __name__ == "__main__":

    # Case selection ------------------------------------------------------------
    # Scenarios (Scenarios/<scenario>.yaml):
    #   L2_S-NRHO_to_L2_N-NRHO / L1_N-HO_to_L2_N-HO / L1_Lyap_to_L2_Lyap
    #   L2_S-HO_to_L1_Lyap / L2_S-HO_to_L4_N-Axial / Sandbox
    scenario = "L1_N-HO_to_L2_N-HO"

    # Problem Types:  deterministic | stochastic_gauss_zoh
    Problem_Type = "stochastic_gauss_zoh"

    # Adaptive Mesh:  fixed | adaptive_fixedtof
    Adaptive_Mesh_Type = "fixed"

    # Gain Parameterization (stochastic only):  arc_lqr | fulltraj_lqr
    Gain_Parametrization_Type = "fulltraj_lqr"

    # Feedback Controller Types (stochastic only):
    #   true_state        - raw covariance propagation (TrueStateCovPropagator)
    #   true_state_sqrt   - square-root covariance propagation (TrueStateSqrtCovPropagator)
    #   estimated_state   - EKF-augmented covariance (EstimatedStateCovPropagator)
    Feedback_Control_Type = "true_state"

    # Measurement Types (estimated_state only):  range | range-rate | angles
    Measurements = ("range", "range-rate", "angles")

    # alpha/beta phasing variable ranges - (a, a) pins alpha to a (fixed
    # phasing); widen e.g. to (0.0, 1.0) to let the optimizer pick a phase along
    # the whole orbit. Collision avoidance is toggled via the scenario yaml's
    # constraints.col_avoid block, not here.
    Alpha_Rng = (0.0, 0.0)
    Beta_Rng = (0.0, 0.0)

    # Hot Start: a CASE name to warm-start from (reads Results/<scenario>/<case>/
    # sol.h5), or None. Variable shapes must match this run's exactly (node
    # resampling isn't ported).
    Hot_Start_Case = "stochastic_gauss_zoh_true_state"

    SEED = 7

    # Chain generate_data.py after solving to also produce data.h5 for plotting.
    Generate_Data = False
    # ---------------------------------------------------------------------------

    # Case directory + files (Results/<scenario>/<case>/) -----------------------
    case = Problem_Type
    if Problem_Type.lower() == 'stochastic_gauss_zoh':
        case += "_" + Feedback_Control_Type
        if Feedback_Control_Type.lower() == 'estimated_state':
            case += "_" + "_".join(Measurements)

    config_file = f"Scenarios/{scenario}.yaml"
    hot_start_file = f"Results/{scenario}/{Hot_Start_Case}/sol.h5" if Hot_Start_Case else None
    case_dir = f"Results/{scenario}/{case}/"
    # ---------------------------------------------------------------------------

    # SNOPT Options -------------------------------------------------------------
    optOptions = {'Major optimality tolerance': 1e-5,
                  'Major feasibility tolerance': 1e-6,
                  'Minor feasibility tolerance': 1e-6,
                  'Major iterations limit': 10000,
                  'Partial prince': 1,
                  'Linesearch tolerance': .99,
                  'Function precision': 1e-10,
                  'Verify level': -1,
                  'Nonderivative linesearch': 0,
                  'Major step limit': 1e0 if hot_start_file is None else 1e-3,
                  'Elastic weight': 1.e4}
    # ---------------------------------------------------------------------------

    # Build the problem ---------------------------------------------------------
    config = yaml_load(config_file)
    toggles = Toggles(
        problem_type=Problem_Type,
        feedback_control_type=Feedback_Control_Type,
        measurements=Measurements,
        gain_param_type=Gain_Parametrization_Type,
        adaptive_mesh_type=Adaptive_Mesh_Type,
        alpha_rng=Alpha_Rng,
        beta_rng=Beta_Rng,
    )
    problem_def = build_problem_def(config, toggles)

    if Problem_Type.lower() == "deterministic":
        top = DeterministicTOP(problem_def)
    elif Problem_Type.lower() == "stochastic_gauss_zoh":
        top = StochasticTOP(problem_def)
    else:
        raise ValueError(f"Unknown Problem_Type: {Problem_Type!r}")

    print("variables:", [v.name for v in top.variables()])
    print("constraints:", [c.name for c in top.constraints()])

    # Initial guess (+ optional hot start) --------------------------------------
    print("Setting Up Initial Guess")
    init_guess = random_init_guess(top.variables(), jax.random.PRNGKey(SEED))
    if hot_start_file is not None:
        init_guess = {**init_guess, **hot_start_init_guess(hot_start_file, top.variables())}

    # Build the pyoptsparse problem (addVarGroup/addConGroup + sparsity) ---------
    print("Processing SNOPT Gradient Sparsity")
    optprob, sens = top.to_pyoptsparse(init_guess)

    # Solve ---------------------------------------------------------------------
    print('SNOPT Starting')
    start_time = time.time()
    sol = SNOPT(options=optOptions)(optprob, sens=sens, timeLimit=None)
    print('SNOPT Finished: %s' % (sol.optInform['text']))
    print("Elapsed Time: %.3f" % (time.time() - start_time))

    # Save the solution + its case manifest -------------------------------------
    t_node_bound = (sol.xStar['t_node_bound'] if Adaptive_Mesh_Type.lower() == 'adaptive_fixedtof'
                    else problem_def.boundary_conditions.t_node_bound)
    os.makedirs(case_dir, exist_ok=True)
    save_solution(os.path.join(case_dir, "sol.h5"), sol.xStar, t_node_bound)
    write_case_yaml(case_dir, scenario, toggles)
    print(f"\nSaved solution + case.yaml to {case_dir}")

    if Generate_Data:
        from generate_data import generate_data
        generate_data(case_dir)
