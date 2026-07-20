# Direct Trajectory Optimization Under Uncertainty (CR3BP)

Direct trajectory optimization in the Circular Restricted Three-Body Problem
(CR3BP), solved via SNOPT (through `pyoptsparse`) using a forward/backward
multiple-shooting formulation, with optional closed-loop feedback control
under uncertainty:

- **Deterministic** transfers — no uncertainty modeling.
- **Stochastic, true-state feedback** — LQR feedback on the true state, with
  covariance propagated either directly or in square-root form
  (`true_state` / `true_state_sqrt`).
- **Stochastic, estimated-state feedback** — closed-loop LQR feedback on an
  Extended Kalman Filter state estimate, with range / range-rate / angles
  ground measurements and trajectory-correction maneuvers.

Solutions are validated with a Monte Carlo trial engine (per-trial EKF for
the estimated-state case), and every stage is built in JAX (`jax.jacfwd`
throughout for sensitivities, `diffrax` for propagation), so the whole
pipeline is differentiable and runs equally well with or without a GPU.

## Prerequisites

- **Python ≥ 3.10**
- **A SNOPT license.** SNOPT is a commercial NLP solver (Stanford Business
  Software) — it is *not* included or pip-installable, and this repo cannot
  provide one. `pyoptsparse` (the Python wrapper this project calls into)
  has to be built locally against your own licensed SNOPT source. See
  [SNOPT / pyoptsparse setup](#snopt--pyoptsparse-setup) below.

## Install

Everything, including the solver, builds into one `.venv` — there's no
separate environment for SNOPT/`pyoptsparse`.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This gets you `jax`, `diffrax`, `numpy`, `scipy`, `h5py`, `pyyaml`,
`astropy`, `sympy`, `matplotlib`, and `tqdm` — enough to read/explore the
codebase, but not enough to actually run SNOPT yet.

`pyoptsparse` is intentionally **not** in `pyproject.toml`'s dependency
list: `pip install pyoptsparse` from PyPI builds a version with no SNOPT
bindings, which "succeeds" while leaving the solver completely
non-functional. It has to be built for real, against your licensed SNOPT,
into this same `.venv`, per the section below.

### SNOPT / pyoptsparse setup

This is written for WSL2/Ubuntu (apt); the same steps apply directly on
native Linux, and on macOS with Homebrew in place of apt. Do this after
[Install](#install) above — `.venv` should already exist and be activated.

1. **Get SNOPT.** Obtain a licensed SNOPT7 source distribution (e.g. through
   your institution) and place it somewhere accessible, e.g. `~/snopt7`.
2. **Install system build dependencies:**
   ```bash
   sudo apt install swig libblas-dev liblapack-dev   # macOS: brew install swig openblas lapack
   ```
3. **Build `pyoptsparse` against SNOPT**, into the active `.venv`, using MDO
   Lab's build tool:
   ```bash
   git clone --branch <tag> --depth 1 https://github.com/OpenMDAO/build_pyoptsparse.git /tmp/build_pyoptsparse
   pip install /tmp/build_pyoptsparse
   build_pyoptsparse -s ~/snopt7
   ```
   Pin `<tag>` to a specific released version (see the repo's
   [Releases page](https://github.com/OpenMDAO/build_pyoptsparse/releases))
   rather than tracking `main`, so the build is reproducible — the clone
   itself is scratch, nothing from it needs to stay around afterward.

   `build_pyoptsparse` will print an environment-variable command near the
   end (pointing the dynamic linker at the built SNOPT library) — run it,
   and add it to your shell profile so it persists across terminals.
4. **Verify:** `python -c "from pyoptsparse.pySNOPT.pySNOPT import SNOPT; print('OK')"`
   should import cleanly with no missing-library errors, in the same
   `.venv` used for everything else.

## Repo layout

```
Scenarios/<scenario>.yaml     # Input configs, one file per scenario (see below)
data/EarthMoon_System/        # System constants + precomputed periodic-orbit families
Results/<scenario>/<case>/    # Outputs: sol.h5 + case.yaml are committed (tiny); data.h5/Sys.mat/*.png are gitignored (regenerable, can run into the GB range)
src/
  problem/       ProblemDefinition, Toggles, OptimizationVariable, the case.yaml manifest
  dynamics/      CR3BP equations of motion, the diffrax-based propagator + sensitivities
  stochastic/    Covariance/EKF propagators, LQR gain parameterization, measurement model, control-noise model, Monte Carlo runners
  optimization/  TrajectoryOptimizationProblem (Deterministic/Stochastic), constraints, initial-guess construction
  postprocess/   Detailed-trajectory replay + Monte Carlo orchestration, matplotlib quick-look plots
  utils/         Generic math helpers, all file I/O (config/solution/results HDF5 schema)
dir_traj_opt.py   # Entry point 1: solve
generate_data.py  # Entry point 2: generate detailed trajectory + Monte Carlo data
plot_traj.py      # Entry point 3: plot
```

## Usage

The pipeline is three separate stages, each a standalone script with a
config block at the top you edit directly (no CLI flags):

1. **`dir_traj_opt.py`** — builds the problem from a `Scenarios/<scenario>.yaml`,
   runs SNOPT, and writes `Results/<scenario>/<case>/{sol.h5, case.yaml}`.
   Set `Hot_Start_Case` to warm-start from a previously solved case in the
   same scenario (variable shapes must match exactly — mesh resampling on
   hot-start isn't supported). Set `Generate_Data = True` to chain step 2
   automatically after solving.
2. **`generate_data.py`** — given a case directory, rebuilds the exact
   problem from its `case.yaml` (no need to re-specify anything), replays
   the solution at detailed resolution, runs the Monte Carlo trials (dispatched
   automatically to the right runner based on the case's feedback type), and
   writes `data.h5` + `Sys.mat`.
3. **`plot_traj.py`** — reads a case directory's `data.h5` and produces two
   quick-look matplotlib figures: a 3D trajectory + control profile
   (always), and a Monte Carlo true-state deviation history with 3σ
   envelope (stochastic cases only).

Because `sol.h5` + `case.yaml` for every scenario/case combination already
in `Results/` are committed, you can run steps 2–3 directly on any of them
without solving anything yourself first — e.g. edit `generate_data.py`'s
config block to point at
`Results/L2_S-NRHO_to_L2_N-NRHO/stochastic_gauss_zoh_true_state/` and run it.

## Scenario config format

`Scenarios/Sandbox.yaml` is fully annotated field-by-field — use it as the
reference when writing a new scenario. The other `Scenarios/*.yaml` files
use the identical schema without comments.

## Notes

- `docs/architecture/` has PlantUML class diagrams for the `src/` design
  (some are more current than others — check each diagram's own freshness
  against the actual code before trusting it for anything load-bearing).
- Statistical (uncertainty-aware) collision avoidance
  (`constraints.col_avoid.stat.bool` in a scenario config) is **not
  currently implemented** — leave it `False`. Setting it `True` silently
  disables collision avoidance entirely rather than raising an error.
