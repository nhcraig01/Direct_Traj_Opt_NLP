# Shared cases and utilities for the diffrax vs heyoka integration benchmark.
#
# Integration only - no evaluate(), no optimization problem. Each case is one
# orbit-library initial condition integrated from t = 0 for a fraction of that
# orbit's period, at three derivative levels:
#
#   L0   X_f                  (7,)
#   L1   J = dX_f/dz          (7, 10)
#   L2   H = d2X_f/dz2        (7, 10, 10)
#
# with z = [X0 (r, v, m), U]. J[:, :7] is production's A and J[:, 7:] its B.
# Each framework's script stores its outputs with save_run() under this
# convention, and compare.py diffs them case by case.

import os
import pickle
import sys
import time

# Repo root on sys.path and as cwd, same as dir_traj_opt.py, so src/ imports
# and Scenarios/ + data/ paths resolve however the script is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import h5py
import jax
import numpy as np

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load


# Config ----------------------------------------------------------------------
# label: (family file in data/EarthMoon_System/, Orb_ID). Defaults are the
# orbits the scenarios already use - swap in whatever you like.
ORBITS = {
    'Lyap': ('L1_Lyap', 20),
    'Halo': ('L2_S_Halo', 100),
    'NRHO': ('L2_S_Halo', 323),
}
FRACTIONS = (0.025, 0.05, 0.25, 0.5, 1.0)  # integration time, as a fraction of each orbit's period
# Production arcs are short (Sandbox dt ~0.03-0.06 T). Those starting at the
# orbit's first saved state are the easy ones (near apolune for the NRHO), so
# these durations are ALSO run centered on closest approach to the Moon - for
# the NRHO, a short arc straight through perilune, the hardest production case.
PERI_FRACTIONS = (0.025, 0.05)
CONTROL = np.zeros(3)         # constant thrust command over every case
TOL = 1e-12                   # diffrax rtol = atol, heyoka tol
REPEAT = 5                    # timed calls per case, after the first

SCENARIO = 'Sandbox'          # source of the spacecraft/system constants only
OUT_DIR = os.path.join('heyoka_benchmark', '_out')
# -----------------------------------------------------------------------------

_MOON_RADIUS_KM = 1737.5  # matches src/dynamics/equations_of_motion.py
_problem_def = None


def problem_def():
    """ProblemDefinition for SCENARIO. Only its constants are used here (and the
    diffrax side builds CR3BPDynamics from it) - nothing is propagated."""
    global _problem_def
    if _problem_def is None:
        config = yaml_load(f'Scenarios/{SCENARIO}.yaml')
        _problem_def = build_problem_def(config, Toggles(problem_type='deterministic'))
    return _problem_def


def eom_constants():
    """The four numbers _cr3bp_eom_lambdify is built from, for the heyoka port."""
    pd = problem_def()
    return {
        'U_Acc_min_nd': float(pd.spacecraft.U_Acc_min_nd),
        've': float(pd.spacecraft.ve),
        'mu': float(pd.Sys['mu']),
        'safe_d': _MOON_RADIUS_KM / float(pd.Sys['Ls']),
    }


def build_cases():
    """One case per (orbit, fraction): the orbit's first saved state with mass
    1.0 appended (the nondimensional m0), CONTROL, and dt = fraction * period.
    Plus, per (orbit, PERI_FRACTIONS entry), a '_peri' case starting dt/2 before
    closest approach to the Moon, so the arc passes straight through it."""
    mu = eom_constants()['mu']
    U = np.asarray(CONTROL, dtype=float)

    def case(key, X0_rv, dt, period):
        X0 = np.concatenate([X0_rv, [1.0]])
        return {'key': key, 'X0': X0, 'U': U, 'z': np.concatenate([X0, U]),
                'dt': dt, 'period': period}

    cases = []
    for label, (family, orb_id) in ORBITS.items():
        with h5py.File(f'data/EarthMoon_System/{family}.h5', 'r') as f:
            X_hst = np.array(f['X_hst'][orb_id])
            t_hst = np.array(f['t_hst'][orb_id])
        period = float(t_hst[-1])

        for frac in FRACTIONS:
            cases.append(case(f'{label}_{frac:g}T', X_hst[0], frac * period, period))

        # Start from the saved state nearest dt/2 before closest approach
        # (wrapping around the period) - the arc then straddles it to within
        # one save interval, T/2000.
        t_peri = t_hst[np.argmin(np.linalg.norm(X_hst[:, :3] - [1 - mu, 0, 0], axis=1))]
        for frac in PERI_FRACTIONS:
            dt = frac * period
            i0 = np.argmin(np.abs(t_hst - (t_peri - dt / 2) % period))
            cases.append(case(f'{label}_{frac:g}T_peri', X_hst[i0], dt, period))
    return cases


def _block(out):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, out,
    )


def time_it(fn, *args, repeat=REPEAT):
    """(output, first-call seconds, median steady-state seconds). For a jitted
    function the first call includes compilation; heyoka's compile happens at
    integrator construction instead, so time that separately."""
    t0 = time.perf_counter()
    out = _block(fn(*args))
    first = time.perf_counter() - t0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _block(fn(*args))
        times.append(time.perf_counter() - t0)
    return out, first, float(np.median(times))


def compare(a, b):
    """(max abs error, normwise relative error) of a against reference b.
    Normwise rather than elementwise, since J and H have exact-zero entries."""
    a, b = np.asarray(a), np.asarray(b)
    abs_err = float(np.max(np.abs(a - b)))
    return abs_err, abs_err / float(np.max(np.abs(b)))


def save_run(framework, records):
    """records: {(case key, level): {'X_f', 'J'?, 'H'?, 'first', 'steady', 'steps'?}}"""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'{framework}.pkl'), 'wb') as f:
        pickle.dump(records, f)


def load_run(framework):
    with open(os.path.join(OUT_DIR, f'{framework}.pkl'), 'rb') as f:
        return pickle.load(f)
