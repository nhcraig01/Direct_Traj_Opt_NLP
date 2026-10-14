# Stage 3 check: the stochastic heyoka propagator against production diffrax.
#
# For each scenario's deterministic hot start, at sub_arcs = 1 and 3: every
# StateSensitivityPropagator output and its jacobian wrt every input, reported
# per output with relative AND absolute differences. Plus a StochasticTOP
# smoke test: it builds on heyoka and evaluate() agrees. The full stochastic
# evaluate/gradient comparison and timing is stage 4.

from cases import compare, time_it

import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.heyoka_propagator import HeyokaStateSensitivityPropagator
from src.dynamics.propagator import StateSensitivityPropagator
from src.optimization.trajectory_optimization_problem import StochasticTOP
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load

SCENARIOS = ('Sandbox', 'L2_S-NRHO_to_L2_N-NRHO')
SUB_ARCS = (1, 3)
OUTPUTS = ('X_hst', 'A_hst', 'B_hst', 'A_arc_hst', 'B_arc_hst', 'A_postinsert')


def diffs(a, b):
    """(normwise relative, max absolute) over all leaves of two pytrees."""
    rel, ab = 0.0, 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        ab = max(ab, float(np.max(np.abs(x - y))) if x.size else 0.0)
        if x.size and np.max(np.abs(y)) > 0:
            rel = max(rel, compare(x, y)[1])
    return rel, ab


def jit_time(fn, *args):
    compiled = jax.jit(fn).lower(*args).compile()
    out, _, steady = time_it(compiled, *args)
    return out, steady


if __name__ == "__main__":
    for scenario in SCENARIOS:
        with h5py.File(f'Results/{scenario}/deterministic/sol.h5', 'r') as f:
            sol = {k: jnp.asarray(f[k][:]) for k in f.keys()}
        prop_in = {'X0': sol['X0'], 'Xf': sol['Xf'], 'U_arc_hst': sol['control_arc_hst'],
                   'arc_length_hst': sol['arc_length_hst']}

        for n_sub in SUB_ARCS:
            config = yaml_load(f'Scenarios/{scenario}.yaml')
            config['traj_parameters']['sub_arcs'] = n_sub
            pds = {name: build_problem_def(config, Toggles(problem_type='stochastic_gauss_zoh', integrator=name))
                   for name in ('diffrax', 'heyoka')}
            props = {'diffrax': StateSensitivityPropagator(pds['diffrax'], CR3BPDynamics(pds['diffrax'])),
                     'heyoka': HeyokaStateSensitivityPropagator(pds['heyoka'], CR3BPDynamics(pds['heyoka']))}

            res = {}
            for name, prop in props.items():
                F = lambda inp, prop=prop, pd=pds[name]: {k: v for k, v in prop.propagate(pd, dict(inp)).items() if k in OUTPUTS}
                vals, vs = jit_time(F, prop_in)
                sens, ss = jit_time(jax.jacfwd(F), prop_in)
                res[name] = dict(vals=vals, sens=sens, vs=vs, ss=ss)
            d, h = res['diffrax'], res['heyoka']

            print(f"{scenario}, sub_arcs = {n_sub}")
            print(f"    {'output':14s} {'shape':>16s}   {'value rel / abs':>20s}   {'jacobian rel / abs':>20s}")
            for k in OUTPUTS:
                vr, va = diffs(h['vals'][k], d['vals'][k])
                sr, sa = diffs(h['sens'][k], d['sens'][k])
                print(f"    {k:14s} {str(tuple(d['vals'][k].shape)):>16s}   {vr:9.1e} / {va:8.1e}   {sr:9.1e} / {sa:8.1e}")
            print(f"    propagate timing: vals {d['vs'] * 1e3:6.2f} -> {h['vs'] * 1e3:5.2f} ms ({d['vs'] / h['vs']:4.1f}x)"
                  f"   jacobian {d['ss'] * 1e3:7.2f} -> {h['ss'] * 1e3:6.2f} ms ({d['ss'] / h['ss']:4.1f}x)")

            # StochasticTOP builds on heyoka and its evaluate() agrees -------------
            tops = {name: StochasticTOP(pd) for name, pd in pds.items()}
            x = {v.name: jnp.reshape(sol[v.name] if v.name in sol else v.value, (v.size,))
                 for v in tops['diffrax'].variables()}
            outs = {name: jax.jit(top.evaluate)(x) for name, top in tops.items()}
            worst = {k: diffs(outs['heyoka'][k], outs['diffrax'][k]) for k in outs['diffrax']}
            k_rel = max(worst, key=lambda k: worst[k][0])
            print(f"    StochasticTOP.evaluate on heyoka: {len(worst)} outputs, max abs diff "
                  f"{max(a for _, a in worst.values()):.1e}; largest relative {worst[k_rel][0]:.1e} ({k_rel})")
        print()
