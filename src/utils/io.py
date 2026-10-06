"""File/data I/O for the rework: config + orbit-family loading, gradient-sparsity
processing, and (added in the detailed-data rewrite) the solution / results HDF5
readers-writers.

Ported from the legacy Lib/utilities.py I/O helpers so src/ no longer depends on
Lib. The results-data schema and its save/load land here alongside these.
"""

import os
import re

import h5py
import numpy as np
import jax.numpy as jnp
import yaml
from scipy.io import savemat


# --------------------------------------------------------------------------- #
# YAML config
# --------------------------------------------------------------------------- #

def yaml_load(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)


def yaml_save(config, filename):
    with open(filename, 'w') as file:
        yaml.safe_dump(config, file)


# --------------------------------------------------------------------------- #
# Orbit-family data
# --------------------------------------------------------------------------- #

def load_family(input_loc: str) -> dict:
    """Load a CR3BP periodic-orbit family (X/t histories, JCs, STMs, break values)."""
    with h5py.File(input_loc, "r") as f:
        return {
            "X_hst": f["X_hst"][:],     # (6, n, N)
            "t_hst": f["t_hst"][:],     # (1, n, N)
            "JCs": f["JCs"][:],         # (1, N)
            "STMs": f["STMs"][:],       # (6, 6, N)
            "BrkVals": f["BrkVals"][:],
        }


# --------------------------------------------------------------------------- #
# Optimizer gradient sparsity
# --------------------------------------------------------------------------- #

def process_sparsity(grad_nonsparse):
    """Reshape a dense jacfwd gradient dict into pyoptsparse's per-constraint
    sparse jac dict, dropping all-zero blocks."""
    grad_sparse = {}
    for key, cur_obj_constr in grad_nonsparse.items():
        new_obj_constr = {}
        for key2, val2_jax in cur_obj_constr.items():
            val2 = np.array(val2_jax)
            if len(val2.shape) != 2:
                if key == 'c_P_Xf':
                    new_obj_constr[key2] = val2.reshape(1, -1)
                else:
                    new_obj_constr[key2] = val2.reshape(-1, 1)
            else:
                new_obj_constr[key2] = val2
            if jnp.all(new_obj_constr[key2] == 0):
                new_obj_constr.pop(key2, None)
        grad_sparse[key] = new_obj_constr
    return grad_sparse


# --------------------------------------------------------------------------- #
# SNOPT solve statistics
# --------------------------------------------------------------------------- #

# Label (as it appears in the file) -> (output dict key, caster). SNOPT's own
# self-reported counts/timings, not pyoptsparse's separate Python-side call
# counter (sol.userObjCalls) - these come straight from SNOPT's fixed-format
# summary block, present verbatim in both SNOPT_print.out and the
# much-shorter SNOPT_summary.out.
_SNOPT_STAT_LABELS = {
    'No. of iterations': ('minor_iterations', int),
    'No. of major iterations': ('major_iterations', int),
    'User function calls (total)': ('user_func_calls', int),
    'Time for solving problem': ('solve_time_s', float),
    'Time for constraint functions': ('constraint_func_time_s', float),
    'Time for objective function': ('objective_func_time_s', float),
}

# Per-major-iteration series pySNOPT records into its history file. These are
# the raw numbers SNOPT hands the wrapper at each major - the same quantities
# its printed iteration table shows, but as floats rather than fixed-width
# text, so there is no column-slicing or parenthesised-value parsing to get
# wrong. See pySNOPT._snstop's iterDict.
_SNOPT_ITER_KEYS = ('nMajor', 'nMinor', 'step', 'feasibility', 'optimality',
                    'merit', 'condZHZ', 'penalty')


def parse_snopt_stats(path='SNOPT_summary.out') -> dict:
    """Pull iteration/call/timing stats out of SNOPT's summary block. Each
    label's line also has a second, unrelated label:value pair sharing the
    line (fixed-width columns), so match on the label prefix and take the
    first number after it."""
    with open(path, 'r') as f:
        text = f.read()

    stats = {}
    for label, (key, cast) in _SNOPT_STAT_LABELS.items():
        m = re.search(re.escape(label) + r'\s+([\d.]+)', text)
        if m:
            stats[key] = cast(m.group(1))

    # Final objective and the EXIT/INFO status pair, which only appear in the
    # trailing block rather than as label:value columns.
    m = re.search(r'Objective\s+(-?[\d.]+E[-+]\d+)', text)
    if m:
        stats['f_star'] = float(m.group(1))
    for tag, key in (('EXIT', 'exit'), ('INFO', 'info')):
        m = re.search(r'SNOPTC ' + tag + r'\s+(\d+)\s*--\s*(.+)', text)
        if m:
            stats[key + '_code'] = int(m.group(1))
            stats[key + '_text'] = m.group(2).strip()
    return stats


SOLVE_LOG_NAMES = {'print': 'SNOPT_print.out',
                   'summary': 'SNOPT_summary.out',
                   'history': 'solve_hist.hst'}


def parse_solve_stats(case_dir) -> dict:
    """Solver diagnostics for a case, as a {iterations, summary} dict ready to
    drop into data.h5.

    `iterations` holds the per-major-iteration series (feasibility, optimality,
    merit, ...) read from pyoptsparse's own history file - dir_traj_opt.py
    points SNOPT's print/summary/history outputs at the case directory so they
    travel with the solution rather than being left in the repo root.

    Every key is always present. Anything unavailable (a case solved before
    this existed, or data regenerated without a matching solve) is filled with
    NaN rather than omitted, so the HDF5 schema stays fixed and MATLAB can
    isnan() instead of testing for field existence.

    case_dir=None returns the NaN defaults without looking for any logs - for
    data generated without a solve of its own (see dir_traj_opt.py's
    Run_Optimization bypass), where any logs sitting in the case directory
    belong to a different solution."""
    summary = {k: np.nan for _, (k, _c) in _SNOPT_STAT_LABELS.items()}
    summary.update({'f_star': np.nan, 'exit_code': np.nan, 'info_code': np.nan,
                    'exit_text': '', 'info_text': ''})
    iterations = {k: np.full(1, np.nan) for k in _SNOPT_ITER_KEYS}

    if case_dir is None:
        return {'iterations': iterations, 'summary': summary}

    # SNOPT's summary file is written to the working directory by default.
    # Prefer a copy archived in the case dir if one is there, else fall back
    # to that default - which assumes data generation follows the solve it
    # belongs to, since nothing ties a root-level log to a particular case.
    summary_path = next(
        (p for p in (os.path.join(case_dir, SOLVE_LOG_NAMES['summary']),
                     SOLVE_LOG_NAMES['summary']) if os.path.exists(p)),
        None,
    )
    if summary_path is not None:
        parsed = parse_snopt_stats(summary_path)
        # exit_text/info_text are strings - keep them out of the numeric dict
        # and hang them off as attributes via 0-d writes.
        summary.update({k: v for k, v in parsed.items() if not k.endswith('_text')})
        for k in ('exit_text', 'info_text'):
            summary[k] = parsed.get(k, '')

    hist_path = os.path.join(case_dir, SOLVE_LOG_NAMES['history'])
    if os.path.exists(hist_path):
        try:
            from pyoptsparse import History
            hist = History(hist_path)
            vals = hist.getValues(names=list(_SNOPT_ITER_KEYS), major=True)
            for k in _SNOPT_ITER_KEYS:
                if k in vals:
                    iterations[k] = np.asarray(vals[k]).squeeze()
        except Exception as e:       # a truncated/locked .hst shouldn't kill data gen
            print(f"  (solve history unreadable, filling NaN: {e})")

    return {'iterations': iterations, 'summary': summary}


# --------------------------------------------------------------------------- #
# Solution (sol.h5) I/O
# --------------------------------------------------------------------------- #

_SOL_KEYS = ('X0', 'Xf', 'control_arc_hst', 'arc_length_hst', 'alpha', 'beta', 'gain_weights')


def save_solution(path, xStar):
    """Write the optimizer solution to sol.h5 - only optimization variables,
    nothing derived. Each key is written iff xStar actually has it, covering
    X0/Xf/control_arc_hst/arc_length_hst (always present) and the
    genuinely-optional alpha/beta/gain_weights with the same rule.

    Node times are NOT stored: they are the cumulative sum of arc_length_hst
    (see propagator.node_times), and storing a derived quantity alongside the
    variable it comes from is how the two drift apart.

    control_arc_hst is stored raw, in whatever units the solve actually
    optimized in - the reg->cart map isn't invertible in general, so a
    cartesian-ised copy could not be converted back for a hot start. Which
    units those are is recorded in the case's case.yaml
    (control_representation), not in sol.h5 itself."""
    with h5py.File(path, "w") as f:
        for k in _SOL_KEYS:
            if k in xStar:
                f.create_dataset(k, data=np.asarray(xStar[k]))


def load_solution(path) -> dict:
    """Read sol.h5 back into an xStar-style dict."""
    sol = {}
    with h5py.File(path, "r") as f:
        for k in _SOL_KEYS:
            if k in f:
                sol[k] = f[k][:]
    return sol


# --------------------------------------------------------------------------- #
# Results (data.h5) I/O - generic schema
# --------------------------------------------------------------------------- #
#
# /meta holds the same fixed set of attributes as always (name, scenario,
# feedback_type, measurements, stochastic, N_arcs, N_subarcs, N_save,
# length_transfer, length_arc). /orbits, /det, /mc mirror data['Orb0']/
# ['Orbf']/['Det']/['MC_Runs'] directly, one dataset/attribute/subgroup per
# dict key, named verbatim - no separate name-mapping table to keep in sync
# by hand every time a field is added on the Python side (see _write_group).

def _write_group(group, d):
    """Recursively write dict `d` into HDF5 `group`: nested dicts become
    subgroups, 0-d values become attributes, everything else becomes a
    dataset (gzip-compressed if 2D+). Names are the dict keys verbatim."""
    for k, v in d.items():
        if isinstance(v, dict):
            _write_group(group.create_group(k), v)
        else:
            arr = np.asarray(v)
            if arr.ndim == 0:
                group.attrs[k] = arr.item()
            else:
                kw = {'compression': 'gzip'} if arr.ndim >= 2 else {}
                group.create_dataset(k, data=arr, **kw)


def save_results(case_dir, data, problem_def, solve_stats=True):
    """Write the grouped results data.h5 (+ Sys.mat) for a case.

    `data` is the {Name, Orb0, Orbf, Det, MC_Runs?} dict from the detailed
    generator / MC runner; problem_def supplies the metadata + Sys.

    solve_stats=False NaN-fills the solve_stats group instead of reading the
    case directory's logs - for data whose solution did not come from a solve
    in that directory."""
    tg = problem_def.toggles
    det = data['Det']
    with h5py.File(os.path.join(case_dir, 'data.h5'), 'w') as f:
        meta = f.create_group('meta')
        meta.attrs['name'] = data['Name']
        meta.attrs['scenario'] = problem_def.name
        meta.attrs['feedback_type'] = tg.feedback_control_type
        meta.attrs['measurements'] = '_'.join(tg.measurements)
        meta.attrs['stochastic'] = int(tg.problem_type.lower() == 'stochastic_gauss_zoh')
        meta.attrs['N_arcs'] = problem_def.dims.N_arcs
        meta.attrs['N_subarcs'] = problem_def.dims.N_subarcs
        meta.attrs['N_save'] = problem_def.dims.N_save
        meta.attrs['length_transfer'] = int(det['length_transfer'])
        meta.attrs['length_arc'] = int(det['length_arc'])

        _write_group(f.create_group('orbits'), {'Orb0': data['Orb0'], 'Orbf': data['Orbf']})
        _write_group(f.create_group('det'), det)
        if 'MC_Runs' in data:
            _write_group(f.create_group('mc'), data['MC_Runs'])
        # Always written, NaN-filled when no matching solve artifacts exist,
        # so the schema MATLAB reads never changes shape.
        _write_group(f.create_group('solve_stats'),
                     parse_solve_stats(case_dir if solve_stats else None))

    savemat(os.path.join(case_dir, 'Sys.mat'), problem_def.Sys)


def load_results(case_dir) -> dict:
    """Read data.h5 back into a nested {meta, orbits, det, mc} dict (schema
    single-source-of-truth for the Python plotter)."""
    out = {'meta': {}, 'orbits': {}, 'det': {}, 'mc': {}}
    with h5py.File(os.path.join(case_dir, 'data.h5'), 'r') as f:
        for k, v in f['meta'].attrs.items():
            out['meta'][k] = v
        for k in f['orbits']:
            out['orbits'][k] = f['orbits'][k][()]
        for k in f['det']:
            out['det'][k] = f['det'][k][()]
        for k, v in f['det'].attrs.items():
            out['det'][k] = v
        if 'mc' in f:
            for k in f['mc']:
                out['mc'][k] = f['mc'][k][()]
    return out
