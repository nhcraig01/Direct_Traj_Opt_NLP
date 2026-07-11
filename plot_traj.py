"""Quick-look plotting entry point - companion to dir_traj_opt.py.

Edit the config block below to pick a case, then run this file. It mirrors
Plotting.m / dir_traj_opt.py: choose scenario, problem type, feedback, and
(for estimated_state) measurements, and it plots the saved data.h5 for that
case. The actual plotting lives in src/postprocess/plotting.py.

Produces, saved as PNGs next to the case's data.h5:
  - Traj_Ctrl.png       : 3D trajectory + control profile (always)
  - Deviation_Hist.png  : true-state deviation history (stochastic only)
"""

import os
import sys

# Ensure the repo root (this script's directory) is on sys.path and is the cwd
# so the Scenarios/... and Results/... paths resolve correctly regardless of
# how this is launched.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from src.postprocess.plotting import plot_case


if __name__ == "__main__":

    # Case selection (same names/values as dir_traj_opt.py) --------------------
    # Scenario Folder Names:
    #   L2_S-NRHO_to_L2_N-NRHO / L1_N-HO_to_L2_N-HO / L1_Lyap_to_L2_Lyap
    #   L2_S-HO_to_L1_Lyap / L2_S-HO_to_L4_N-Axial / Sandbox
    folder_name = "L2_S-NRHO_to_L2_N-NRHO"

    # Problem Types:  deterministic | stochastic_gauss_zoh
    Problem_Type = "stochastic_gauss_zoh"

    # Feedback Controller Types (stochastic only):
    #   true_state | true_state_sqrt | estimated_state
    Feedback_Control_Type = "true_state"

    # Measurement Types (estimated_state only):  range | range-rate | angles
    Measurements = ("range", "range-rate", "angles")

    # Plot theme:  light | dark
    Theme = "light"

    # Open an interactive figure window (rotate the 3D trajectory, zoom) in
    # addition to saving PNGs.
    Show = True
    # -------------------------------------------------------------------------

    # Case directory - Results/<scenario>/<case>/, same case derivation as
    # dir_traj_opt.py.
    file_name = Problem_Type
    if Problem_Type.lower() == 'stochastic_gauss_zoh':
        file_name += "_" + Feedback_Control_Type
        if Feedback_Control_Type.lower() == 'estimated_state':
            file_name += "_" + "_".join(Measurements)
    case_dir = f"Results/{folder_name}/{file_name}/"

    print(f"Plotting case: {case_dir}")
    plot_case(case_dir, theme=Theme, save=True, show=Show)
