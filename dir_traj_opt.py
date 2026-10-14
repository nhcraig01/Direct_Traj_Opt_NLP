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

from src.utils.console import banner, saved, snopt_summary, step
from src.utils.io import yaml_load, save_solution, parse_snopt_stats, SOLVE_LOG_NAMES
from src.problem.case_manifest import write_case_yaml
from src.optimization.init_guess import hot_start_init_guess, random_init_guess
from src.problem.problem_definition import Toggles, build_problem_def
from src.optimization.trajectory_optimization_problem import DeterministicTOP, StochasticTOP


if __name__ == "__main__":

    # Case selection ------------------------------------------------------------
    # Scenarios (Scenarios/<scenario>.yaml):
    #   L2_S-NRHO_to_L2_N-NRHO / L1_N-HO_to_L2_N-HO / L1_Lyap_to_L2_Lyap
    #   L2_S-HO_to_L1_Lyap / L2_S-HO_to_L4_N-Axial / Sandbox 
    scenario = "L2_S-NRHO_to_L2_N-NRHO"

    # Problem Types:  deterministic | stochastic_gauss_zoh
    Problem_Type = "stochastic_gauss_zoh"

    # Control Representation :  cartesian | regularized
    Control_Representation = "cartesian"

    # Adaptive Mesh:  fixed | adaptive_fixedtof
    Adaptive_Mesh_Type = "fixed"

    # Integrator:  diffrax | heyoka
    Integrator = "heyoka"

    # Gain Parameterization (stochastic only):  arc_lqr | fulltraj_lqr
    Gain_Parametrization_Type = "arc_lqr"

    # Feedback Controller Types (stochastic only):
    #   true_state             - covariance propagation (TrueStateCovPropagator)
    #   true_state_sqrt        - square-root covariance propagation (TrueStateSqrtCovPropagator)
    #   estimated_state        - EKF-augmented covariance (EstimatedStateCovPropagator)
    #   estimated_state_sqrt   - square-root EKF-augmented covariance (EstimatedStateSqrtCovPropagator)
    Feedback_Control_Type = "true_state_sqrt"


    # Measurement Types (estimated_state / estimated_state_sqrt only):  range | range-rate | angles
    Measurements = ("range","range-rate","angles",)

    # Departure/Arrival orbit phasing ranges
    Alpha_Rng = (0.0, 0.1)
    Beta_Rng = (0.0, 0.1)

    # Hot Start: a CASE name to warm-start from (reads Results/<scenario>/<case>/
    # sol.h5), or None.
    Hot_Start_Case = "deterministic"

    SEED = 1

    # False skips SNOPT and takes the initial guess (i.e. the hot start) as the
    # solution 
    Run_Optimization = True

    # Chain generate_data.py after solving to also produce data.h5 for plotting.
    Generate_Data = True
    # ---------------------------------------------------------------------------

    # Case directory + files (Results/<scenario>/<case>/) -----------------------
    case = Problem_Type
    if Problem_Type.lower() == 'stochastic_gauss_zoh':
        case += "_" + Feedback_Control_Type
        if Feedback_Control_Type.lower() in ('estimated_state', 'estimated_state_sqrt'):
            case += "_" + "_".join(Measurements)

    config_file = f"Scenarios/{scenario}.yaml"
    hot_start_file = f"Results/{scenario}/{Hot_Start_Case}/sol.h5" if Hot_Start_Case else None
    case_dir = f"Results/{scenario}/{case}/"
    # ---------------------------------------------------------------------------

    # SNOPT Options -------------------------------------------------------------
    optOptions = {'Major optimality tolerance': 1e-5 if Problem_Type.lower() == 'deterministic' else 1e-4,
                  'Major feasibility tolerance': 1e-6,
                  'Minor feasibility tolerance': 1e-6,
                  'Major iterations limit': 20000,
                  'Partial price': 1,
                  'Linesearch tolerance': .99,
                  'Function precision': 1e-10,
                  'Verify level': -1,
                  'Nonderivative linesearch': 1,
                  'Major step limit': 1e-2,
                  'Elastic weight': 1.e4,
                  'Hessian': 'limited memory',}
    # ---------------------------------------------------------------------------

    # Build the problem ---------------------------------------------------------
    config = yaml_load(config_file)
    toggles = Toggles(
        problem_type=Problem_Type,
        control_representation=Control_Representation,
        feedback_control_type=Feedback_Control_Type,
        measurements=Measurements,
        gain_param_type=Gain_Parametrization_Type,
        adaptive_mesh_type=Adaptive_Mesh_Type,
        integrator=Integrator,
        alpha_rng=Alpha_Rng,
        beta_rng=Beta_Rng,
    )
    banner(scenario, toggles, [
        ('start', f"hot start from {Hot_Start_Case}" if Hot_Start_Case else f"cold start, seed {SEED}"),
        *([] if Run_Optimization else [('solve', 'skipped - generating data for the starting point')]),
    ])
    problem_def = build_problem_def(config, toggles)

    # Control_Representation is dispatched inside the TOP (via its
    # ControlRepresentation), so it doesn't branch here.
    if Problem_Type.lower() == "deterministic":
        top = DeterministicTOP(problem_def)
    elif Problem_Type.lower() == "stochastic_gauss_zoh":
        top = StochasticTOP(problem_def)
    else:
        raise ValueError(f"Unknown Problem_Type: {Problem_Type!r}")

    # Initial guess (+ optional hot start) --------------------------------------
    with step("initial guess"):
        init_guess = random_init_guess(top.variables(), jax.random.PRNGKey(SEED))
        if hot_start_file is not None:
            init_guess = {**init_guess, **hot_start_init_guess(hot_start_file, top.variables())}

    os.makedirs(case_dir, exist_ok=True)

    if not Run_Optimization:
        # The initial guess IS the solution. Same per-variable fallback
        # to_pyoptsparse() applies for anything the hot start didn't supply,
        # reshaped to each variable's declared size - guess_fn hands back a bare
        # scalar for alpha/beta, which would otherwise land in sol.h5 as a
        # scalar dataset load_solution() can't read.
        xStar = {v.name: jax.numpy.reshape(init_guess.get(v.name, v.value), (v.size,))
                 for v in top.variables()}
    else:
        # Build the pyoptsparse problem (addVarGroup/addConGroup + sparsity) -----
        with step("building SNOPT problem", timed=True):
            optprob, sens = top.to_pyoptsparse(init_guess)

        # Solve -----------------------------------------------------------------
        # SNOPT's print/summary files stay at their default location (repo root).
        # The pyoptsparse history file does go in the case dir - it's what carries
        # the per-major-iteration series (feasibility/optimality/merit) into
        # data.h5's solve_stats group, so it needs to travel with the solution.
        # The progress table starts here, not earlier: building the problem
        # above already evaluates once, for the sparsity pattern.
        top.progress.start()
        try:
            sol = SNOPT(options=optOptions)(
                optprob, sens=sens, timeLimit=None,
                storeHistory=os.path.join(case_dir, SOLVE_LOG_NAMES['history']),
                storeSens=False,
            )
        finally:
            top.progress.finish()
        xStar = sol.xStar
        snopt_summary(parse_snopt_stats(SOLVE_LOG_NAMES['summary']), SOLVE_LOG_NAMES['summary'])

    # Save the solution + its case manifest -------------------------------------
    # Only a real solve writes these; a bypass leaves whatever solution is
    # already in case_dir untouched.
    written = []
    if Run_Optimization:
        save_solution(os.path.join(case_dir, "sol.h5"), xStar)
        write_case_yaml(case_dir, scenario, toggles)
        written += ['sol.h5', 'case.yaml']

    if Generate_Data:
        from generate_data import generate_data
        # solve_stats only describes a solve that actually happened here.
        generate_data(xStar, scenario, toggles, case_dir, solve_stats=Run_Optimization)
        written += ['data.h5', 'Sys.mat']

    if written:
        saved(case_dir, written)
