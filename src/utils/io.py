"""File/data I/O for the rework: config + orbit-family loading, gradient-sparsity
processing, and (added in the detailed-data rewrite) the solution / results HDF5
readers-writers.

Ported from the legacy Lib/utilities.py I/O helpers so src/ no longer depends on
Lib. The results-data schema and its save/load land here alongside these.
"""

import os

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
# Solution (sol.h5) I/O
# --------------------------------------------------------------------------- #

_SOL_KEYS = ('X0', 'Xf', 'U_arc_hst', 't_node_bound', 'alpha', 'beta', 'gain_weights')


def save_solution(path, xStar, t_node_bound):
    """Write the optimizer solution to sol.h5. t_node_bound is passed explicitly
    (it is a fixed-mesh constant, not an xStar variable, unless adaptive)."""
    with h5py.File(path, "w") as f:
        f.create_dataset('X0', data=np.asarray(xStar['X0']))
        f.create_dataset('Xf', data=np.asarray(xStar['Xf']))
        f.create_dataset('U_arc_hst', data=np.asarray(xStar['U_arc_hst']))
        f.create_dataset('t_node_bound', data=np.asarray(t_node_bound))
        for k in ('alpha', 'beta', 'gain_weights'):
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
# Results (data.h5) I/O - grouped schema
# --------------------------------------------------------------------------- #
#
#   /meta   : attributes (name, scenario, feedback_type, measurements,
#             stochastic, N_arcs, N_subarcs, N_save, length_transfer, length_arc)
#   /orbits : orb0_X orb0_t orbf_X orbf_t
#   /det    : X t X_node t_node U U_sph U_arc  (+ dV_mean attr)
#      +stoch: A B K K_arc gain_weights P U_bound tcm_bound U_dV tcm_dV P_u
#              P_targ P_Xf_targ P_XT_targ  (+ dV_stat, dV_bound attrs)
#      +est  : Phat Ptild Phattild H P_v
#   /mc     : X t U U_sph dV dV_tcm
#      +est  : Xhat P Phat Ptild Phattild P_mean Phat_mean Ptild_mean Phattild_mean

# Det dict key -> /det dataset name (arrays); scalars go to /det attrs below.
_DET_MAP = {
    'X_hst': 'X', 't_hst': 't', 'X_node_hst': 'X_node', 't_node_hst': 't_node',
    'U_hst': 'U', 'U_hst_sph': 'U_sph', 'U_arc_hst': 'U_arc',
    'A_hst': 'A', 'B_hst': 'B', 'K_hst': 'K', 'K_arc_hst': 'K_arc',
    'gain_weights_hst': 'gain_weights', 'P_hst': 'P', 'P_u_hst': 'P_u',
    'U_norm_bound_hst': 'U_bound', 'TCM_norm_bound_hst': 'tcm_bound',
    'U_norm_dV_hst': 'U_dV', 'TCM_norm_dV_hst': 'tcm_dV',
    'P_Targ_hst': 'P_targ', 'P_Xf_targ': 'P_Xf_targ', 'P_XT_targ': 'P_XT_targ',
    'Phat_hst': 'Phat', 'Ptild_hst': 'Ptild', 'Phattild_hst': 'Phattild',
    'H_hst': 'H', 'P_v_hst': 'P_v',
}
_DET_SCALARS = ('dV_mean', 'dV_stat', 'dV_bound')

# MC_Runs key -> /mc dataset name.
_MC_MAP = {
    'X_hsts': 'X', 't_hsts': 't', 'U_hsts': 'U', 'U_hsts_sph': 'U_sph',
    'dVs': 'dV', 'dV_tcms': 'dV_tcm', 'Xhat_hsts': 'Xhat', 'P_hsts': 'P',
    'Phat_hsts': 'Phat', 'Ptild_hsts': 'Ptild', 'Phattild_hsts': 'Phattild',
    'P_mean_hst': 'P_mean', 'Phat_mean_hst': 'Phat_mean',
    'Ptild_mean_hst': 'Ptild_mean', 'Phattild_mean_hst': 'Phattild_mean',
}


def save_results(case_dir, data, problem_def):
    """Write the grouped results data.h5 (+ Sys.mat) for a case.

    `data` is the {Name, Orb0, Orbf, Det, MC_Runs?} dict from the detailed
    generator / MC runner; problem_def supplies the metadata + Sys."""
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

        orb = f.create_group('orbits')
        orb.create_dataset('orb0_X', data=np.asarray(data['Orb0']['X_hst']))
        orb.create_dataset('orb0_t', data=np.asarray(data['Orb0']['t_hst']))
        orb.create_dataset('orbf_X', data=np.asarray(data['Orbf']['X_hst']))
        orb.create_dataset('orbf_t', data=np.asarray(data['Orbf']['t_hst']))

        dg = f.create_group('det')
        for src_k, dst_k in _DET_MAP.items():
            if src_k in det:
                dg.create_dataset(dst_k, data=np.asarray(det[src_k]))
        for s in _DET_SCALARS:
            if s in det:
                dg.attrs[s] = float(np.asarray(det[s]))

        if 'MC_Runs' in data:
            mc = f.create_group('mc')
            for src_k, dst_k in _MC_MAP.items():
                if src_k in data['MC_Runs']:
                    arr = np.asarray(data['MC_Runs'][src_k])
                    kw = {'compression': 'gzip'} if arr.ndim >= 2 else {}
                    mc.create_dataset(dst_k, data=arr, **kw)

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
