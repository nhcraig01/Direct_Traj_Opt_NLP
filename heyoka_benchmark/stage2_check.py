# Stage 2 check: the deterministic heyoka propagator against production diffrax.
#
# For each scenario's deterministic hot start, at sub_arcs = 1 and 3:
#   - propagate() outputs and jacfwd(propagate) wrt every input agree,
#   - DeterministicTOP.evaluate() and its full SNOPT gradient agree,
#   - timings for both.
# Plus: case.yaml round-trips the integrator toggle, old manifests load as
# diffrax, and heyoka + stochastic says clearly it isn't wired up yet.

from cases import compare, time_it

import os
import tempfile
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.heyoka_propagator import HeyokaStatePropagator
from src.dynamics.propagator import StatePropagator
from src.optimization.trajectory_optimization_problem import DeterministicTOP, StochasticTOP
from src.problem.case_manifest import load_case_spec, write_case_yaml
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load

SCENARIOS = ('Sandbox', 'L2_S-NRHO_to_L2_N-NRHO')
SUB_ARCS = (1, 3)
INPUTS = ('X0', 'Xf', 'U_arc_hst', 'arc_length_hst')


def worst_rel(a, b):
    """Max normwise relative difference over matching pytree leaves, skipping
    blocks that are exactly zero in the reference."""
    out = 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if np.max(np.abs(y)) > 0:
            out = max(out, compare(x, y)[1])
    return out


def jit_time(fn, *args):
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t0
    out, _, steady = time_it(compiled, *args)
    return out, compile_s, steady


if __name__ == "__main__":
    for scenario in SCENARIOS:
        with h5py.File(f'Results/{scenario}/deterministic/sol.h5', 'r') as f:
            sol = {k: jnp.asarray(f[k][:]) for k in f.keys()}
        prop_in = {'X0': sol['X0'], 'Xf': sol['Xf'], 'U_arc_hst': sol['control_arc_hst'],
                   'arc_length_hst': sol['arc_length_hst']}

        for n_sub in SUB_ARCS:
            config = yaml_load(f'Scenarios/{scenario}.yaml')
            config['traj_parameters']['sub_arcs'] = n_sub
            pd_d = build_problem_def(config, Toggles(integrator='diffrax'))
            pd_h = build_problem_def(config, Toggles(integrator='heyoka'))
            print(f"{scenario}, sub_arcs = {n_sub}")

            # propagate() and its jacobian wrt every input --------------------
            props = {'diffrax': StatePropagator(pd_d, CR3BPDynamics(pd_d)),
                     'heyoka': HeyokaStatePropagator(pd_h, CR3BPDynamics(pd_h))}
            res = {}
            for name, prop in props.items():
                pd = pd_d if name == 'diffrax' else pd_h
                F = lambda inp, prop=prop, pd=pd: prop.propagate(pd, dict(inp))['X_hst']
                vals, vc, vs = jit_time(F, prop_in)
                sens, sc, ss = jit_time(jax.jacfwd(F), prop_in)
                res[name] = dict(vals=vals, sens=sens, vc=vc, vs=vs, sc=sc, ss=ss)
            d, h = res['diffrax'], res['heyoka']
            print(f"    propagate      X_hst {tuple(d['vals'].shape)}   vals rel diff {worst_rel(h['vals'], d['vals']):.1e}"
                  f"   jacobian rel diff {worst_rel(h['sens'], d['sens']):.1e}")
            print(f"                   vals {d['vs'] * 1e3:7.2f} -> {h['vs'] * 1e3:6.2f} ms ({d['vs'] / h['vs']:4.1f}x)"
                  f"   jacobian {d['ss'] * 1e3:7.2f} -> {h['ss'] * 1e3:6.2f} ms ({d['ss'] / h['ss']:4.1f}x)")

            # The whole deterministic evaluate() and SNOPT's gradient of it ----
            tops = {'diffrax': DeterministicTOP(pd_d), 'heyoka': DeterministicTOP(pd_h)}
            x = {v.name: jnp.reshape(sol[v.name] if v.name in sol else v.value, (v.size,))
                 for v in tops['diffrax'].variables()}
            tres = {}
            for name, top in tops.items():
                vals, _, vs = jit_time(top.evaluate, x)
                sens, _, ss = jit_time(jax.jacfwd(top.evaluate), x)
                tres[name] = dict(vals=vals, sens=sens, vs=vs, ss=ss)
            d, h = tres['diffrax'], tres['heyoka']
            print(f"    evaluate       outputs rel diff {worst_rel(h['vals'], d['vals']):.1e}"
                  f"   SNOPT gradient rel diff {worst_rel(h['sens'], d['sens']):.1e}")
            print(f"                   vals {d['vs'] * 1e3:7.2f} -> {h['vs'] * 1e3:6.2f} ms ({d['vs'] / h['vs']:4.1f}x)"
                  f"   gradient {d['ss'] * 1e3:7.2f} -> {h['ss'] * 1e3:6.2f} ms ({d['ss'] / h['ss']:4.1f}x)")
        print()

    # Plumbing ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        write_case_yaml(tmp, 'Sandbox', Toggles(integrator='heyoka'))
        print("case.yaml round trip:        integrator =", load_case_spec(tmp)[1].integrator)
    old = 'Results/Sandbox/deterministic'
    print(f"pre-toggle manifest ({old}): integrator =", load_case_spec(old)[1].integrator)
    try:
        StochasticTOP(build_problem_def(yaml_load('Scenarios/Sandbox.yaml'),
                                        Toggles(problem_type='stochastic_gauss_zoh', integrator='heyoka')))
        print("heyoka + stochastic:         constructed (wired up in stage 3)")
    except NotImplementedError as e:
        print("heyoka + stochastic:         NotImplementedError -", e)
