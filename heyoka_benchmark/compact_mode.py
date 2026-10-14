# Compact vs non-compact heyoka integrators.
#
# compact_mode=True (what heyoka_runs.py uses) generates loops over similar
# operations - small code, quick to compile. compact_mode=False unrolls the full
# Taylor recursion into straight-line code - typically faster per step, but the
# code (and so compile time and memory) grows with the number of elementary
# operations x the Taylor order, which is large for the variational systems.
# In production an integrator would be compiled once per process, so a long
# compile is acceptable if every call gets faster.

from cases import build_cases, compare, time_it

import time

import numpy as np

from heyoka_runs import N, build, derivative_map, unpack

if __name__ == "__main__":
    cases = build_cases()

    # L0/L1 only: a non-compact L2 at full optimization didn't finish compiling
    # in 17+ minutes - l2_compile_variants.py covers L2 with capped subprocesses.
    for level in ('L0', 'L1'):
        results = {}
        for compact in (True, False):
            label = 'compact' if compact else 'non-compact'
            print(f"{level} {label}: building...", flush=True)
            t0 = time.perf_counter()
            ta = build(level, cases[0], compact_mode=compact)
            compile_s = time.perf_counter() - t0
            J_map, H_map = derivative_map(ta)
            var_ic = ta.state[N:].copy()

            def run(c):
                ta.time = 0.0
                ta.state[:N] = c['X0']
                ta.state[N:] = var_ic
                ta.pars[:] = c['U']
                return ta.propagate_until(c['dt'])

            per_case = {}
            for c in cases:
                res, _, steady = time_it(run, c)
                per_case[c['key']] = (steady, int(res[3]), unpack(ta.state, level, J_map, H_map))
            results[compact] = (compile_s, len(ta.decomposition), per_case)

        (c_compile, c_ops, c_cases), (n_compile, n_ops, n_cases) = results[True], results[False]
        print(f"\n{level}   compile: compact {c_compile:6.2f} s   non-compact {n_compile:6.2f} s"
              f"   ({c_ops} elementary operations in the decomposition)")
        print(f"    {'case':12s} {'compact':>11s} {'non-compact':>12s} {'speedup':>8s} {'steps':>7s}"
              f" {'max rel diff':>13s}")
        for c in cases:
            cs, csteps, cout = c_cases[c['key']]
            ns, nsteps, nout = n_cases[c['key']]
            diff = max(compare(nout[f], cout[f])[1] for f in cout)
            steps = f"{csteps}" if csteps == nsteps else f"{csteps}/{nsteps}"
            print(f"    {c['key']:12s} {cs * 1e3:8.3f} ms {ns * 1e3:9.3f} ms {cs / ns:7.2f}x {steps:>7s} {diff:13.1e}")
        print(flush=True)
