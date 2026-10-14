# Stage 4: the WHOLE stochastic evaluate() on heyoka vs diffrax.
#
# What SNOPT actually calls - StochasticTOP.evaluate (propagation, gains,
# covariance propagation, constraints) and jacfwd of it - for every feedback
# type, at each scenario's deterministic solution with default gain weights.
# Reports agreement (relative AND absolute, per output) and timing, so the
# end-to-end speedup includes everything heyoka doesn't touch.

from cases import compare, time_it

import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from src.optimization.trajectory_optimization_problem import StochasticTOP
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load

SCENARIOS = ('Sandbox', 'L2_S-NRHO_to_L2_N-NRHO')
FEEDBACK = ('true_state', 'true_state_sqrt', 'estimated_state', 'estimated_state_sqrt')
MEASUREMENTS = ('range', 'range-rate', 'angles')
GAIN_PARAM = 'fulltraj_lqr'


def diffs(a, b):
    """(normwise relative, max absolute) over all leaves of two pytrees."""
    rel, ab = 0.0, 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        x, y = np.asarray(x), np.asarray(y)
        if not x.size:
            continue
        ab = max(ab, float(np.max(np.abs(x - y))))
        if np.max(np.abs(y)) > 0:
            rel = max(rel, compare(x, y)[1])
    return rel, ab


def jit_time(fn, *args):
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t0
    out, _, steady = time_it(compiled, *args)
    return out, compile_s, steady


if __name__ == "__main__":
    summary = []
    for scenario in SCENARIOS:
        with h5py.File(f'Results/{scenario}/deterministic/sol.h5', 'r') as f:
            sol = {k: jnp.asarray(f[k][:]) for k in f.keys()}
        config = yaml_load(f'Scenarios/{scenario}.yaml')

        for fb in FEEDBACK:
            res = {}
            for integ in ('diffrax', 'heyoka'):
                toggles = Toggles(problem_type='stochastic_gauss_zoh', feedback_control_type=fb,
                                  measurements=MEASUREMENTS if fb.startswith('estimated') else (),
                                  gain_param_type=GAIN_PARAM, integrator=integ)
                top = StochasticTOP(build_problem_def(config, toggles))
                x = {v.name: jnp.reshape(sol[v.name] if v.name in sol else v.value, (v.size,))
                     for v in top.variables()}
                vals, vc, vs = jit_time(top.evaluate, x)
                sens, sc, ss = jit_time(jax.jacfwd(top.evaluate), x)
                res[integ] = dict(vals=vals, sens=sens, vc=vc, vs=vs, sc=sc, ss=ss,
                                  n_var=sum(v.size for v in top.variables()))
            d, h = res['diffrax'], res['heyoka']

            print(f"{scenario}  {fb}  ({d['n_var']} variables)")
            print(f"    {'':22s} {'diffrax':>10s} {'heyoka':>10s} {'speedup':>8s}")
            print(f"    {'evaluate (vals)':22s} {d['vs'] * 1e3:7.2f} ms {h['vs'] * 1e3:7.2f} ms {d['vs'] / h['vs']:7.1f}x")
            print(f"    {'gradient (sens)':22s} {d['ss'] * 1e3:7.1f} ms {h['ss'] * 1e3:7.1f} ms {d['ss'] / h['ss']:7.1f}x")
            print(f"    {'compile vals / sens':22s} {d['vc']:4.1f}/{d['sc']:4.1f} s {h['vc']:4.1f}/{h['sc']:4.1f} s")
            for k in d['vals']:
                vr, va = diffs(h['vals'][k], d['vals'][k])
                gr, ga = diffs(h['sens'][k], d['sens'][k])
                print(f"    {k:14s} value rel {vr:8.1e} abs {va:8.1e}   gradient rel {gr:8.1e} abs {ga:8.1e}")
            summary.append((scenario, fb, d['vs'] + d['ss'], h['vs'] + h['ss']))
            print()

    print("per SNOPT major iteration (one vals + one sens):")
    for scenario, fb, dt, ht in summary:
        print(f"    {scenario:24s} {fb:22s} {dt * 1e3:7.1f} -> {ht * 1e3:6.1f} ms  ({dt / ht:4.1f}x)")
