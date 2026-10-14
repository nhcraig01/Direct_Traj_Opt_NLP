# diffrax side of the integration benchmark.
#
# The integrator exactly as StateSensitivityPropagator._propagate_final sets it
# up (Dopri8, PID step-size controller, ForwardMode, endpoint-only save), run at
# L0/L1/L2 over every case in cases.py. Outputs land in _out/diffrax.pkl.

from cases import (
    TOL, build_cases, compare, problem_def, save_run, time_it,
)

import time

import diffrax as dfx
import jax
import numpy as np

from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.propagator import StateSensitivityPropagator

pd = problem_def()
eom = CR3BPDynamics(pd)
term = dfx.ODETerm(eom.eom)
solver = dfx.Dopri8()


def make_levels(tol):
    """(levels, solve) at a given rtol = atol. A factory so reference.py can
    rerun the same setup at other tolerances."""
    stepsize_controller = dfx.PIDController(rtol=tol, atol=tol)

    def solve(z, dt):
        return dfx.diffeqsolve(
            term, solver, 0.0, dt, None, z[:7],
            args=z[7:], stepsize_controller=stepsize_controller,
            adjoint=dfx.ForwardMode(),
            saveat=dfx.SaveAt(t1=True),
            max_steps=16 ** 5,
        )

    def flow(z, dt):
        return solve(z, dt).ys[-1]

    # Each level returns everything below it from ONE solve - has_aux carries
    # the lower-order outputs through the tangent pass rather than re-solving,
    # the same way heyoka's variational system hands back X_f and J together.
    def state_aux(z, dt):
        X_f = flow(z, dt)
        return X_f, X_f

    def jac_aux(z, dt):
        J, X_f = jax.jacfwd(state_aux, has_aux=True)(z, dt)
        return J, (X_f, J)

    def L0(z, dt):
        return (flow(z, dt),)

    def L1(z, dt):
        _, (X_f, J) = jac_aux(z, dt)
        return X_f, J

    def L2(z, dt):
        H, (X_f, J) = jax.jacfwd(jac_aux, has_aux=True)(z, dt)
        return X_f, J, H

    return {'L0': L0, 'L1': L1, 'L2': L2}, solve


FIELDS = ('X_f', 'J', 'H')

if __name__ == "__main__":
    cases = build_cases()
    z0, dt0 = cases[0]['z'], cases[0]['dt']
    LEVELS, solve = make_levels(TOL)

    # Steps are a property of the primal solve - tangents don't feed the
    # step-size controller - so one count per case covers all three levels.
    num_steps = jax.jit(lambda z, dt: solve(z, dt).stats['num_steps'])

    records = {}
    print(f"diffrax  Dopri8  rtol = atol = {TOL:g}\n")
    for level, fn in LEVELS.items():
        t0 = time.perf_counter()
        compiled = jax.jit(fn).lower(z0, dt0).compile()
        compile_s = time.perf_counter() - t0
        print(f"{level}  compile {compile_s:7.2f} s")
        print(f"    {'case':12s} {'steady':>11s} {'steps':>7s}")
        for c in cases:
            out, first, steady = time_it(compiled, c['z'], c['dt'])
            rec = {k: np.asarray(v) for k, v in zip(FIELDS, out)}
            rec.update(compile=compile_s, first=first, steady=steady,
                       steps=int(num_steps(c['z'], c['dt'])))
            records[(c['key'], level)] = rec
            print(f"    {c['key']:12s} {steady * 1e3:8.2f} ms {rec['steps']:7d}")
        print()

    # Sanity checks ------------------------------------------------------------
    # L1 against production's own _arc_dAB, which needs the same tolerance to be
    # a like-for-like check (production reads it from the scenario yaml).
    if pd.integration.r_tol == TOL and pd.integration.a_tol == TOL:
        prop = StateSensitivityPropagator(pd, eom)
        arc_dAB = jax.jit(prop._arc_dAB)
        worst = 0.0
        for c in cases:
            A, B = arc_dAB(c['X0'], c['U'], 0.0, c['dt'])
            J = records[(c['key'], 'L1')]['J']
            worst = max(worst, compare(J[:, :7], A)[1], compare(J[:, 7:], B)[1])
        print(f"L1 vs production _arc_dAB      max rel err {worst:.2e}")
    else:
        print("L1 vs production _arc_dAB      skipped (TOL differs from the scenario's)")

    # Lower levels recomputed at higher ones should agree, and H is symmetric.
    worst_lvl, worst_sym = 0.0, 0.0
    for c in cases:
        r0, r1, r2 = (records[(c['key'], l)] for l in ('L0', 'L1', 'L2'))
        worst_lvl = max(worst_lvl, compare(r1['X_f'], r0['X_f'])[1],
                        compare(r2['X_f'], r0['X_f'])[1], compare(r2['J'], r1['J'])[1])
        H = r2['H']
        worst_sym = max(worst_sym, compare(H, H.transpose(0, 2, 1))[1])
    print(f"cross-level X_f / J agreement  max rel err {worst_lvl:.2e}")
    print(f"H symmetry                     max rel err {worst_sym:.2e}")

    save_run('diffrax', records)
    print("\nsaved heyoka_benchmark/_out/diffrax.pkl")
