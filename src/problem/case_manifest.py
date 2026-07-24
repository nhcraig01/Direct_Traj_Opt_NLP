"""Per-case manifest (case.yaml).

A solution is meaningless without the problem formulation that produced it, so
`dir_traj_opt.py` writes a small `case.yaml` next to `sol.h5` recording the
scenario + the Toggles seed. `generate_data.py` / plotting then rebuild the
exact ProblemDefinition/StochasticTOP from it - no folder-name parsing, and the
Monte Carlo dispatch (true_state vs estimated_state, measurement set) falls out
automatically because StochasticTOP is reconstructed from these toggles.

Only the user-chosen Toggles seed is stored; the config-derived switches
(det_col_avoid/stat_col_avoid) are re-folded in by build_problem_def() from the
scenario yaml, so they are intentionally not duplicated here.
"""

import os

import yaml

from .problem_definition import Toggles

CASE_FILENAME = 'case.yaml'


def write_case_yaml(case_dir: str, scenario: str, toggles: Toggles) -> str:
    """Write case_dir/case.yaml capturing the scenario + Toggles seed."""
    os.makedirs(case_dir, exist_ok=True)
    spec = {
        'scenario': scenario,
        'problem_type': toggles.problem_type,
        'control_representation': toggles.control_representation,
        'feedback_control_type': toggles.feedback_control_type,
        'measurements': list(toggles.measurements),
        'gain_param_type': toggles.gain_param_type,
        'adaptive_mesh_type': toggles.adaptive_mesh_type,
        'alpha_rng': list(toggles.alpha_rng),
        'beta_rng': list(toggles.beta_rng),
    }
    path = os.path.join(case_dir, CASE_FILENAME)
    with open(path, 'w') as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return path


def load_case_spec(case_dir: str) -> tuple[str, Toggles]:
    """Read case_dir/case.yaml -> (scenario, Toggles)."""
    with open(os.path.join(case_dir, CASE_FILENAME)) as f:
        spec = yaml.safe_load(f)
    toggles = Toggles(
        problem_type=spec['problem_type'],
        control_representation=spec['control_representation'],
        feedback_control_type=spec['feedback_control_type'],
        measurements=tuple(spec['measurements']),
        gain_param_type=spec['gain_param_type'],
        adaptive_mesh_type=spec['adaptive_mesh_type'],
        alpha_rng=tuple(spec['alpha_rng']),
        beta_rng=tuple(spec['beta_rng']),
    )
    return spec['scenario'], toggles
