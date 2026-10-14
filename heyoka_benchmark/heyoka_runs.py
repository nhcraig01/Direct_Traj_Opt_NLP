# heyoka side of the integration benchmark.
#
# Same cases and levels as diffrax_runs.py. One integrator per level is built
# (= compiled, by LLVM) once, then reset and reused for every case - heyoka's
# equivalent of jit-compiling once and calling many times. Outputs land in
# _out/heyoka.pkl under the same convention as _out/diffrax.pkl.

from cases import TOL, build_cases, compare, load_run, save_run, time_it

import time

import heyoka as hy
import numpy as np

from heyoka_eom import X, sys_

N = len(X)  # 7 physical states
# Variables the variational equations differentiate with respect to, in the
# z = [X0 (r, v, m), U] order that J/H use: the 7 initial states, then the
# three thrust parameters.
ARGS = list(X) + [hy.par[0], hy.par[1], hy.par[2]]
NZ = len(ARGS)  # 10

# Per-level compile mode, from compact_mode.py / l2_compile_variants.py:
# non-compact runs 1.4-2.3x (L0) and ~1.6x (L1) faster for 0.2 s / 8 s of
# compile; L2 stays compact - non-compact there either never finishes
# compiling or, with optimization off, runs 4x slower.
COMPACT = {'L0': False, 'L1': False, 'L2': True}


def build(level, c, tol=TOL, compact_mode=True, **kwargs):
    """Construct (and so compile) the integrator for one level. Passing only the
    7 physical states lets heyoka fill in the variational initial conditions
    itself (identity for d/dX0, zero for d/dU and every second derivative).

    compact_mode=True generates loops over similar operations: small code, fast
    to compile. False unrolls the whole Taylor recursion into straight-line
    code: faster to run, but code size grows with (operations x order).
    kwargs go straight to taylor_adaptive (e.g. opt_level, parjit)."""
    sys_level = sys_ if level == 'L0' else hy.var_ode_sys(sys_, ARGS, order=int(level[1]))
    return hy.taylor_adaptive(sys_level, c['X0'], pars=c['U'], tol=tol,
                              compact_mode=compact_mode, **kwargs)


def derivative_map(ta):
    """Where each derivative sits in ta.state. get_mindex(idx) gives
    [component, n_0, ..., n_9] - the derivative order wrt each of ARGS - so the
    first-order entries scatter into J[comp, j] and the second-order ones into
    H[comp, j, k]. Mixed second derivatives are stored once; H gets both halves."""
    J_map, H_map = [], []
    for idx in range(N, len(ta.state)):
        mi = ta.get_mindex(idx)
        comp, nz = mi[0], np.nonzero(mi[1:])[0]
        if sum(mi[1:]) == 1:
            J_map.append((idx, comp, nz[0]))
        elif len(nz) == 1:   # pure, e.g. d2/dx2 (exponent 2)
            H_map.append((idx, comp, nz[0], nz[0]))
        else:                # mixed (two exponents of 1)
            H_map.append((idx, comp, nz[0], nz[1]))
    return np.array(J_map, dtype=int).reshape(-1, 3), np.array(H_map, dtype=int).reshape(-1, 4)


def unpack(state, level, J_map, H_map):
    out = {'X_f': state[:N].copy()}
    if level != 'L0':
        J = np.zeros((N, NZ))
        J[J_map[:, 1], J_map[:, 2]] = state[J_map[:, 0]]
        out['J'] = J
    if level == 'L2':
        H = np.zeros((N, NZ, NZ))
        H[H_map[:, 1], H_map[:, 2], H_map[:, 3]] = state[H_map[:, 0]]
        H[H_map[:, 1], H_map[:, 3], H_map[:, 2]] = state[H_map[:, 0]]
        out['H'] = H
    return out


if __name__ == "__main__":
    cases = build_cases()
    try:
        dfx_records = load_run('diffrax')
    except FileNotFoundError:
        dfx_records = None

    records = {}
    print(f"heyoka  Taylor  tol = {TOL:g}\n")
    for level in ('L0', 'L1', 'L2'):
        t0 = time.perf_counter()
        ta = build(level, cases[0], compact_mode=COMPACT[level])
        compile_s = time.perf_counter() - t0
        J_map, H_map = derivative_map(ta)

        # heyoka only sets the variational initial conditions at construction,
        # so keep them and restore them before every run.
        var_ic = ta.state[N:].copy()

        def run(c):
            # Reset BEFORE propagating, every call: time_it calls this several
            # times, and without the reset the second call would integrate
            # from t = dt to t = dt and time nothing.
            ta.time = 0.0
            ta.state[:N] = c['X0']
            ta.state[N:] = var_ic
            ta.pars[:] = c['U']
            return ta.propagate_until(c['dt'])

        mode = 'compact' if COMPACT[level] else 'non-compact'
        print(f"{level}  compile {compile_s:7.2f} s   ({mode}, order {ta.order}, {len(ta.state)} states)")
        print(f"    {'case':12s} {'steady':>11s} {'steps':>7s} {'diffrax/heyoka':>16s}")
        for c in cases:
            res, first, steady = time_it(run, c)
            if res[0] != hy.taylor_outcome.time_limit:
                print(f"    WARNING {c['key']}: integration stopped with {res[0]}")
            rec = unpack(ta.state, level, J_map, H_map)
            rec.update(compile=compile_s, first=first, steady=steady, steps=int(res[3]))
            records[(c['key'], level)] = rec
            ratio = (f"{dfx_records[(c['key'], level)]['steady'] / steady:13.1f}x"
                     if dfx_records else "")
            print(f"    {c['key']:12s} {steady * 1e3:8.3f} ms {rec['steps']:7d} {ratio:>16s}")
        print()

    # Sanity checks ------------------------------------------------------------
    # Lower levels recomputed at higher ones should agree (they are separate
    # integrators here, so this is a real check, unlike in diffrax_runs.py).
    worst_lvl = 0.0
    for c in cases:
        r0, r1, r2 = (records[(c['key'], l)] for l in ('L0', 'L1', 'L2'))
        worst_lvl = max(worst_lvl, compare(r1['X_f'], r0['X_f'])[1],
                        compare(r2['X_f'], r0['X_f'])[1], compare(r2['J'], r1['J'])[1])
    print(f"cross-level X_f / J agreement  max rel err {worst_lvl:.2e}")

    # Quick look against diffrax - both at TOL, so this is agreement, not error.
    # compare.py scores both against a common high-accuracy reference.
    if dfx_records:
        for level, fields in (('L0', ('X_f',)), ('L1', ('X_f', 'J')), ('L2', ('X_f', 'J', 'H'))):
            for f in fields:
                worst = max(compare(records[(c['key'], level)][f], dfx_records[(c['key'], level)][f])[1]
                            for c in cases)
                print(f"vs diffrax  {level} {f:4s}             max rel diff {worst:.2e}")

    save_run('heyoka', records)
    print("\nsaved heyoka_benchmark/_out/heyoka.pkl")
