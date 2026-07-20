# Project context for Claude

This file is for Claude, not for end users of the tool — it travels with the
repo (unlike Claude's local memory, which is keyed off the working-directory
path and won't survive a move to a new machine) so context carries over
across machines and sessions. `README.md` is the user-facing doc for what
this project is and how to run it; read that first for the actual pipeline.
This file is about *state*: where things stand, what's been verified, what's
still open, and a few non-obvious conventions worth not re-deriving.

Keep this updated at natural checkpoints (a release, a big verification pass,
a environment migration) rather than after every small change — it's meant
to stay a small, high-signal document, not a running log.

## Current status (as of 2026-07-19)

Branch `pre_release_cleanup_and_restructure` is functionally complete and
verified — it's the sole codebase in the repo (`Lib/` and all legacy scripts
were deleted back on 2026-07-07/10). It is a clean fast-forward of `main`
(no divergence). **Not yet merged/tagged as v1.0.0** — the user wants to do
that only after:

1. A README/polish pass on this branch (in progress — see below for what's
   already done).
2. Confirming the whole setup runs cleanly on a **new Mac** the user is in
   the process of migrating to (currently developing on WSL2/Ubuntu +
   Windows). This is the actual gating step — nothing else is blocking.

**Do not merge to `main`, tag, or push to `origin` without the user's
explicit go-ahead in that specific moment.** This project's established norm
(and general good practice) is to treat merges/tags/pushes as needing fresh
confirmation each time, not something a prior approval covers going forward.

### What's been verified

A full controlled benchmark was just run comparing this branch against
legacy `main` (`Lib/`-based, run from an isolated `git worktree`, never
imported in-process alongside `src/`) on scenario `L2_S-NRHO_to_L2_N-NRHO`,
with identical SNOPT settings and an identical shared cold-start init guess
injected on both sides (bypassing each side's own RNG). Three cases:

| Case | Result |
|---|---|
| Deterministic | Both converge to the same optimum (`fStar` matches to ~1e-12). Only case where wall-clock time is a clean apples-to-apples comparison: legacy 34.73s vs new 36.28s (~4% slower, most likely JIT/tracing overhead on the OOP side's sparsity-processing step, not a real per-iteration solve-speed difference). |
| Stochastic true_state | Both hit a shared 1000-major-iteration cap without fully converging (`fStar` within ~0.8%). Wall time: legacy 1293.85s vs new 1246.37s. |
| Stochastic estimated_state (range/range-rate/angles, 2 measurements/arc) | Same story, capped at 1000 majors (`fStar` within ~1.4%). Wall time: legacy 4227.87s vs new 4371.83s. |

Net takeaway: the OOP rework (`src/`) is functionally equivalent to legacy
and not meaningfully slower. All scratch benchmark artifacts (driver
scripts, a temporary `git worktree` at
`Direct_Traj_Opt_NLP_legacy_bench`, generated `Results/*/_benchmark_*/`
dirs) were cleaned up afterward — this repo has no trace of that exercise
except this summary and the earlier, separately-run **full legacy-vs-new
functional audit** from 2026-07-10 (5 parallel agents covered ~55
legacy functions field-by-field; only gap found was `stat_col_avoid`, see
Known gaps below).

### README changes made this session

- Unified install onto a single `.venv` — the old instructions had two
  disconnected environments (a `.venv` from `pip install -e .`, then a
  separate, never-reconciled `conda create -n traj_opt` for the SNOPT
  build). Now `build_pyoptsparse` builds directly into the same `.venv`.
- Dropped conda from the SNOPT/pyoptsparse setup steps in favor of plain
  `apt`/`brew` system packages — conda wasn't actually load-bearing there
  (the build already used `apt install swig libblas-dev liblapack-dev`, not
  conda-forge).
- `build_pyoptsparse` clone is now pinned to a specific release tag rather
  than tracking `main`, for reproducibility.

**This `.venv`-only approach is unvalidated** — it was written based on
reasoning about what should work, not a real from-scratch build. The
existing, actually-working setup on the WSL machine is two conda
environments (`snopt_traj_opt` and `snopt_env`, both have `jax`+
`pyoptsparse`+SNOPT built in; `snopt_traj_opt` is the one actually used day
to day — it's first on `$PATH`). **The first real task on the new Mac should
be a genuine from-scratch install following the rewritten README**, and the
README should be corrected against whatever actually happens, not assumed
correct.

### A live gotcha hit this session (environment/IDE, not code)

Editing `.vscode/settings.json`'s `python-envs.defaultEnvManager` /
`defaultPackageManager` (from `conda` to `venv`, to match the new README) at
one point broke Pylance import resolution and running the file from the IDE
on the WSL machine, because the currently-*selected* interpreter for the
workspace is a separate, stickier bit of VS Code state (not fully reset just
by changing that default-manager setting) and no `.venv` actually exists
here. Root cause + fix: the setting is back to `conda` now (correct for this
WSL machine — leave it that way here), and the user had to explicitly
re-run **Python: Select Interpreter** and pick `snopt_traj_opt` to fully
clear the stale selection. Worth remembering: if this comes up again on the
Mac (once a real `.venv` exists there), the analogous settings should point
at `venv`/`pip`, and if Pylance still looks wrong after changing the
setting, re-running the interpreter picker (not just editing the JSON) is
often the actual fix.

## Environment / setup notes

- **This WSL machine**: two working conda envs exist, `snopt_traj_opt`
  (the one on `$PATH`, used for everything) and `snopt_env` (also has
  `jax`+`pyoptsparse`, less commonly used). Both have SNOPT-enabled
  `pyoptsparse` built in.
- **New Mac**: not set up yet. Plan is plain `.venv` + `pip install -e .` +
  `build_pyoptsparse` per the rewritten README — see "unvalidated" note
  above.
- `Plotting/Plotting.m` (MATLAB) currently points `repo` at a WSL UNC path
  (`\\wsl.localhost\Ubuntu-22.04\...`) with a comment already anticipating
  the Mac move — after migrating, just change `repo` to a local path.
- `.claude/settings.local.json` should be gitignored (personal/local
  permission allowlist, not meant to be shared) — check `.gitignore` still
  has this if it's ever missing.

## Conventions worth knowing

- **`sol_data` pattern**: pipeline stages in `src/` (`Propagator.propagate`,
  `GainParameterization.compute_gains`, `ErrorPropagator.propagate_cov`,
  `Constraint.evaluate`, etc.) all take/return a flat, untyped `sol_data`
  dict rather than a typed `NamedTuple`/`eqx.Module`. This is deliberate —
  the user is actively deriving new math (e.g. higher-order state
  transition tensors) and doesn't want a schema to pre-declare and maintain
  as that evolves. Follow the same `(problem_def, sol_data) -> dict` shape
  for any new producer/consumer. Exception: `OptimizationVariable` stays a
  `NamedTuple` (build-time, consumed once by `to_pyoptsparse()`, not part of
  the per-call data flow).
- **Verifying against legacy**: when comparing new-code behavior against
  legacy (`Lib/`, only reachable via `git worktree`/`main` now, never
  in-process), the pattern that's worked well: isolate legacy in its own
  worktree/process, inject one shared RNG-free init guess into both sides,
  match every SNOPT option exactly, and check both `fStar`/`optInform` *and*
  max\|diff\| per solution variable — not just "objective is close."
  A cross-compatible flat HDF5 schema (`src/utils/io.py`'s
  `save_solution`/`load_solution`, keys `X0,Xf,U_arc_hst,t_node_bound,
  alpha,beta,gain_weights`) makes hot-start files portable between legacy
  and new with zero conversion.
- **Data generation for comparisons**: when producing `data.h5`/`Sys.mat`
  for MATLAB plotting from a benchmark/comparison run (as opposed to a
  normal solve), generate it through the **new code's** `generate_data.py`
  machinery even for a legacy-produced solution — legacy's `sol.h5` output
  is schema-compatible, so it can be dropped into a case dir and run through
  `generate_data()` directly rather than needing legacy's own (nonexistent)
  equivalent tooling.
- Scratch/benchmark work (driver scripts, one-off comparison scripts,
  temporary worktrees) belongs outside the tracked repo — this project's
  norm is to keep `Results/` and the repo root free of anything that isn't
  either real source or a real, intentionally-committed case result.

## Known gaps / accepted limitations

- **No automated test suite.** `tests/` was deliberately removed during the
  v1.0 restructure — the user's call, on the basis that the README's usage
  walkthrough + manual/ad hoc verification (like the benchmark above) covers
  it for now. Worth a conscious decision if this ever changes, not
  something to silently reintroduce.
- **`stat_col_avoid`** (statistical/unscented-transform collision avoidance
  under uncertainty) is not implemented. The config toggle exists but is a
  silent no-op — setting `constraints.col_avoid.stat.bool: True` in a
  scenario config disables collision avoidance entirely rather than raising
  an error. Documented in the README and in `Scenarios/Sandbox.yaml`'s
  comments. The user has explicitly decided not to implement this right now.
- `docs/architecture/_rendered/*.svg` can drift stale relative to the
  `.puml` sources they're rendered from — check freshness before trusting
  a diagram for anything load-bearing (also called out in the README).
