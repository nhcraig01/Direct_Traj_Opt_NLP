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
(no divergence).

**Both original v1.0.0 gating items are now done**: the README/polish pass,
and a genuine from-scratch install + full pipeline validation on the new
Mac (see "macOS setup" below) — the Mac is now the user's primary machine
going forward, migrating off WSL2/Ubuntu + Windows. **Still not
merged/tagged as v1.0.0** — that's a separate, deliberate action for the
user to explicitly request, not something to do just because the checklist
above is now clear.

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

### README changes made on the WSL machine (README-polish session)

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

This was written based on reasoning about what should work, not a real
from-scratch build, at the time. It has since been **validated for real**
on the Mac migration session below — corrections made there are folded
into the README directly, and summarized in "macOS setup" next.

### macOS setup — validated (Mac migration session, 2026-07-19)

Full from-scratch install on a new M5 Pro MacBook Pro (Apple Silicon,
Homebrew at `/opt/homebrew`), done live and corrected against the README as
it went. The README now reflects all of this — this section is the "why"
behind those changes, worth keeping in case the README ever needs
re-deriving or something regresses:

- **Homebrew wasn't preinstalled**; Xcode Command Line Tools were. Its
  installer needs a real interactive `sudo` password prompt, so this step
  can't be run unattended by an agent — has to be done directly by the
  user in their own terminal.
- **System Python is 3.9.6** (`/usr/bin/python3`, Apple's bundled one, too
  old for this project's `>=3.10` floor). Used `brew install python@3.12`
  (a *versioned* formula, not the generic `python3` one) — lands at
  `/opt/homebrew/bin/python3.12` without touching the system Python or
  becoming the default `python3`/`pip3` on `PATH`.
- **`.venv/` was missing from `.gitignore`** — never existed in this repo
  before now (no `.venv` had ever been created here). Fixed.
- **`.vscode/settings.json`'s `python-envs.defaultEnvManager`/
  `defaultPackageManager`** flipped from `conda` (correct for the WSL
  machine — see the gotcha below) to `venv`. This is a committed/shared
  file, so the change affects any other clone too — deliberate, since the
  Mac is now primary.
- A clean VS Code install has **no Python extension by default** —
  `code --install-extension ms-python.python` (pulls in Pylance too) is
  required before "Python: Select Interpreter" even exists as a command.
- **`openblas`/`lapack` install "keg-only"** on macOS (not symlinked into
  `/opt/homebrew`, since macOS ships its own via `Accelerate.framework`) —
  needs `LDFLAGS`/`CPPFLAGS` pointed at
  `/opt/homebrew/opt/{openblas,lapack}` during the `pyoptsparse` build.
- **`gfortran` has no standalone Homebrew formula** — it ships bundled
  inside `gcc` (`brew install gcc`). Apple's Clang toolchain has no Fortran
  compiler, and SNOPT's source is Fortran 77, so this is a hard
  requirement the old apt-based wording didn't make obvious.
- **Real upstream bug hit in `build_pyoptsparse` v2.0.13**:
  `install_mumps_from_src()` discards `git_clone()`'s return value (a
  `tempfile.TemporaryDirectory`), so CPython garbage-collects it —
  deleting the freshly cloned MUMPS source, including `get.Mumps`, before
  the very next line runs. Confirmed by reproducing the clone+checkout
  manually (file's there and executable right after checkout) and by
  diffing against `install_metis_from_src()`, which *does* capture the
  return value and works fine right next to it. Workaround: the
  `-d`/`--no-delete` CLI flag switches to a `tempfile.mkdtemp()` path with
  no auto-cleanup finalizer, sidestepping the bug entirely. Worth
  rechecking on whatever version is current next time — may be fixed
  upstream by then.
- **`DYLD_LIBRARY_PATH`** (macOS's `LD_LIBRARY_PATH` equivalent) is needed
  at `pyoptsparse` *import* time, not just once during the build — scoped
  it into `.venv/bin/activate` itself (set on activate, restored on
  deactivate) rather than the shell profile, so it only applies when this
  venv is active.
- End-to-end verified working: `pip install -e .` (every dependency landed
  as a prebuilt arm64 wheel, no source builds needed, including usually-
  finicky ones like `scipy`/`numpy`/`astropy`), JAX reports a `CpuDevice`,
  `pyoptsparse`/SNOPT import cleanly, `generate_data.py` replay and a real
  cold-start SNOPT solve (`dir_traj_opt.py`, deterministic,
  `L2_S-NRHO_to_L2_N-NRHO`) both completed successfully, and MATLAB
  `Plotting.m` was repointed and plots correctly.
- **`Plotting/Plotting.m`'s `repo` variable was updated** — but only in the
  live OneDrive copy the user actually runs from
  (`OneDrive - purdue.edu/Grad School/Project Code/Direct_Traj_Opt_NLP/Plotting/Plotting.m`),
  **not** in the repo's own tracked `Plotting` symlink (still points at the
  old WSL UNC path, still broken on macOS). Fixing that symlink was
  explicitly deferred by the user this session — still open, see
  "Environment / setup notes" below.

### A gotcha hit on the WSL machine (README-polish session)

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

- **Old WSL machine**: two working conda envs exist, `snopt_traj_opt`
  (the one on `$PATH`, used for everything) and `snopt_env` (also has
  `jax`+`pyoptsparse`, less commonly used). Both have SNOPT-enabled
  `pyoptsparse` built in. Being migrated away from — the Mac below is now
  primary.
- **New Mac (now primary, set up + validated 2026-07-19)**: Homebrew at
  `/opt/homebrew`, `python@3.12`, a project-local `.venv` (gitignored) with
  `pip install -e .` plus `pyoptsparse`+SNOPT built directly in via
  `build_pyoptsparse`. Licensed SNOPT7 source lives long-term in OneDrive
  (`OneDrive - purdue.edu/Grad School/Project Code/snopt7.7/snopt7/`) with
  a local build copy at `~/snopt7`. See "macOS setup" above for the full
  list of gotchas hit getting here.
- `Plotting/Plotting.m` (MATLAB) — the live OneDrive copy the user actually
  runs has been repointed at the local Mac repo path and works correctly
  (see "macOS setup" above). The repo's own tracked `Plotting` symlink is a
  separate, still-open item: it still points at the old WSL UNC path
  (`\\wsl.localhost\Ubuntu-22.04\...`), still broken on macOS, and the user
  explicitly deferred fixing it. Don't assume it's resolved just because
  the OneDrive copy is.
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
- **`generate_data.py`/`plot_traj.py` need `pyoptsparse` importable even
  though neither ever calls SNOPT.** `src/utils/io.py` and
  `src/optimization/trajectory_optimization_problem.py` both import
  `pyoptsparse` at module load time rather than lazily inside the
  functions that actually use it. Discovered on the Mac while trying to
  validate the README's "replay/plot without re-solving" claim before
  `pyoptsparse` was built — it fails at import, not at any SNOPT call.
  README now notes this caveat. A lazy-import fix is possible but hasn't
  been made — the user's call, not done as part of this session.
