# High-accuracy reference for the integration benchmark.
#
# heyoka in 113-bit multiprecision (~34 significant digits, MPFR via hy.real)
# at its default tolerance for that precision, so round-off is out of the
# picture entirely. The model constants are the same doubles both frameworks
# use, so this is the exact solution of the very ODE they both integrate, to
# far below double precision. Stored as float64 in _out/reference.pkl.
#
# Then a convergence check, so the reference isn't just heyoka grading itself:
# both frameworks rerun at several tolerances should close in on it.

from cases import build_cases, compare, save_run

import time

import heyoka as hy
import jax
import numpy as np

from diffrax_runs import make_levels
from heyoka_eom import sys_
from heyoka_runs import ARGS, N, build, derivative_map, unpack

PREC = 113                    # bits of mantissa: quad-precision equivalent
TOLS = (1e-10, 1e-12, 1e-14)  # convergence-check tolerances, both frameworks


def _mp(x):
    return [hy.real(float(v), PREC) for v in np.ravel(x)]


def reference(level, c):
    """One multiprecision integration. Built fresh per case - heyoka caches the
    compiled code, so only the first build of each level actually compiles."""
    sys_level = sys_ if level == 'L0' else hy.var_ode_sys(sys_, ARGS, order=int(level[1]))
    ta = hy.taylor_adaptive(sys_level, _mp(c['X0']), pars=_mp(c['U']),
                            fp_type=hy.real, prec=PREC, compact_mode=True)
    ta.propagate_until(hy.real(float(c['dt']), PREC))
    state = np.array([float(s) for s in ta.state])
    J_map, H_map = derivative_map(ta)
    return unpack(state, level, J_map, H_map), ta.order


if __name__ == "__main__":
    cases = build_cases()

    # Reference -----------------------------------------------------------------
    ref = {}
    print(f"reference: heyoka, {PREC}-bit multiprecision\n")
    for level in ('L0', 'L1', 'L2'):
        t0 = time.perf_counter()
        for c in cases:
            ref[(c['key'], level)], order = reference(level, c)
        print(f"  {level}  order {order}   {time.perf_counter() - t0:7.1f} s for {len(cases)} cases")
    save_run('reference', ref)
    print("  saved heyoka_benchmark/_out/reference.pkl\n")

    # Convergence check ---------------------------------------------------------
    # Each framework at three tolerances, scored against the reference. Errors
    # are normwise relative, max over the three durations of each orbit.
    orbits = sorted({c['key'].split('_')[0] for c in cases}, key=[c['key'].split('_')[0] for c in cases].index)
    print("convergence check: max normwise rel error vs reference (over durations)\n")
    print(f"  {'tol':>7s} {'level':5s} {'field':5s}" + "".join(f"{o + '  dfx / hy':>26s}" for o in orbits))
    for tol in TOLS:
        dfx_levels, _ = make_levels(tol)
        for level in ('L1', 'L2'):
            dfx_fn = jax.jit(dfx_levels[level])
            ta = build(level, cases[0], tol=tol)
            J_map, H_map = derivative_map(ta)
            var_ic = ta.state[N:].copy()
            fields = ('X_f', 'J') if level == 'L1' else ('X_f', 'J', 'H')
            errs = {(o, f): [0.0, 0.0] for o in orbits for f in fields}
            for c in cases:
                dfx_out = dict(zip(('X_f', 'J', 'H'), (np.asarray(v) for v in dfx_fn(c['z'], c['dt']))))
                ta.time = 0.0
                ta.state[:N] = c['X0']
                ta.state[N:] = var_ic
                ta.pars[:] = c['U']
                ta.propagate_until(c['dt'])
                hy_out = unpack(ta.state, level, J_map, H_map)
                o = c['key'].split('_')[0]
                for f in fields:
                    r = ref[(c['key'], level)][f]
                    errs[(o, f)][0] = max(errs[(o, f)][0], compare(dfx_out[f], r)[1])
                    errs[(o, f)][1] = max(errs[(o, f)][1], compare(hy_out[f], r)[1])
            for f in fields:
                print(f"  {tol:7.0e} {level:5s} {f:5s}"
                      + "".join(f"{errs[(o, f)][0]:12.1e} / {errs[(o, f)][1]:8.1e}  " for o in orbits))
        print()
