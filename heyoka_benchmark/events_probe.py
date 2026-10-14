# Event-based gravity softening for heyoka - investigation, no src/ changes.
#
# heyoka's select() fixes the softening branch at the START of each Taylor
# step and nothing stops a step at the switch radius, so arcs that cross the
# softened core integrate wrong (found on the NRHO cold start, arc 16). Three
# variants on that exact arc, against a tight-tolerance diffrax reference:
#   (a) select()                       - what src/ does today
#   (b) select() + terminal events     - steps end at the switch, but the next
#                                        step's branch is decided by d <= sw AT
#                                        the boundary, i.e. by round-off
#   (c) flag parameters + terminal     - branch is a runtime parameter that the
#       events                           event callback sets from the CROSSING
#                                        DIRECTION (inward = softened)
# Then the cost of (c) vs (a) on the 21 benchmark arcs, which never enter the
# core: event detection runs on every step whether or not anything triggers.

from cases import build_cases, time_it

import jax
import jax.numpy as jnp
import numpy as np
import heyoka as hy
import sympy as sp

from diffrax_runs import make_levels
from src.dynamics.equations_of_motion import cr3bp_eom_constants, cr3bp_eom_sympy
from src.optimization.init_guess import random_init_guess
from src.optimization.trajectory_optimization_problem import DeterministicTOP
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load

TOL = 1e-12
N = 7


def build_system(pd, mode):
    """mode: 'select' (today), or 'flag' (branch = par[3] for Earth, par[4] for
    Moon). Returns (sys, diff args, event expressions d^2 - sw^2 per primary)."""
    _, states, controls, eoms, softening = cr3bp_eom_sympy(*cr3bp_eom_constants(pd))
    X = hy.make_vars(*[str(s) for s in states])
    base = dict(zip([*states, *controls], [*X, *(hy.par[i] for i in range(3))]))
    to_hy = lambda e: hy.from_sympy(sp.sympify(e), base)
    s_dict, events = dict(base), []
    for k, (ph, pw) in enumerate(sorted(softening.items(), key=lambda kv: str(kv[0]))):  # dtmp (Earth), rtmp (Moon)
        (inner, cond), (outer, _) = pw.args
        if mode == 'select':
            s_dict[ph] = hy.select(hy.lte(to_hy(cond.lhs), to_hy(cond.rhs)), to_hy(inner), to_hy(outer))
        else:
            flag = hy.par[3 + k]
            s_dict[ph] = flag * to_hy(inner) + (1.0 - flag) * to_hy(outer)
        # squared distance - same root, no sqrt in the event's Taylor series
        events.append(to_hy(cond.lhs) ** 2 - to_hy(cond.rhs) ** 2)
    sys_ = list(zip(X, (hy.from_sympy(e, s_dict) for e in eoms)))
    return sys_, [*X, *(hy.par[i] for i in range(3))], events


class Integ:
    """Minimal version of src's _Integrator, optionally with terminal events."""

    def __init__(self, pd, mode, order, events_on):
        sys_, args, ev = build_system(pd, mode)
        self.mode, self.order = mode, order
        if order:
            sys_ = hy.var_ode_sys(sys_, args, order=order)
        t_events = []
        if events_on:
            for k, e in enumerate(ev):
                cb = (lambda k: lambda ta, d_sgn: self._flip(ta, k, d_sgn))(k) if mode == 'flag' else (lambda ta, d_sgn: True)
                # Explicit cooldown: heyoka's automatic one scales with max|state| INCLUDING the
                # variational entries, which at order 2 made it longer than a core crossing and
                # the exit was ignored. The callback sets the flag absolutely from the crossing
                # direction, so a spurious re-trigger at the same crossing is harmless.
                t_events.append(hy.t_event(e, callback=cb, cooldown=1e-9))
        X_init = np.zeros(N); X_init[-1] = 1.0
        npars = 5 if mode == 'flag' else 3
        self.ta = hy.taylor_adaptive(sys_, X_init, pars=np.zeros(npars), tol=TOL,
                                     compact_mode=(order == 2), t_events=t_events)
        self.var_ic = self.ta.state[N:].copy()
        self.mu, self.sw2 = pd.Sys['mu'], (0.75 * 1737.5 / pd.Sys['Ls']) ** 2
        self.n_events = 0
        self.events_on = events_on
        if order:
            self.J_idx = [(i, self.ta.get_mindex(i)) for i in range(N, len(self.ta.state)) if sum(self.ta.get_mindex(i)[1:]) == 1]

    def _flip(self, ta, k, d_sgn):
        # d_sgn is the sign of d(event)/dt in TIME, not along the integration -
        # backward arcs would get the flags inverted. Entering the core means
        # the event decreases along the integration: d_sgn * direction < 0.
        ta.pars[3 + k] = 1.0 if d_sgn * self.direction < 0 else 0.0
        self.n_events += 1
        return True  # keep integrating

    def run(self, X0, U, tau):
        ta = self.ta
        self.direction = 1.0 if tau >= 0 else -1.0
        ta.time = 0.0
        ta.state[:N] = X0
        if self.order:
            ta.state[N:] = self.var_ic
        ta.pars[:3] = U
        if self.mode == 'flag':  # branch at the arc's start, from where it starts
            for k, c in enumerate(([-self.mu, 0, 0], [1 - self.mu, 0, 0])):
                ta.pars[3 + k] = 1.0 if np.sum((X0[:3] - c) ** 2) <= self.sw2 else 0.0
        if self.events_on:
            ta.reset_cooldowns()
        res = ta.propagate_until(float(tau))
        return res

    def J(self):
        J = np.zeros((N, 10))
        for i, mi in self.J_idx:
            J[mi[0], int(np.nonzero(mi[1:])[0][0])] = self.ta.state[i]
        return J


if __name__ == "__main__":
    # The failing arc: NRHO cold start (seed 1, alpha/beta free), arc 16
    cfg = yaml_load('Scenarios/L2_S-NRHO_to_L2_N-NRHO.yaml')
    pd = build_problem_def(cfg, Toggles(alpha_rng=(0.0, 1.0), beta_rng=(0.0, 1.0)))
    top = DeterministicTOP(pd)
    vars_ = top.variables(); ig = random_init_guess(vars_, jax.random.PRNGKey(1))
    x = {v.name: jnp.reshape(jnp.asarray(ig.get(v.name, v.value), dtype=jnp.float64), (v.size,)) for v in vars_}
    inp = dict(x); inp['U_arc_hst'] = top._control_repr.to_cartesian(pd, x['control_arc_hst'])
    X_hst = np.asarray(top._propagator.propagate(pd, dict(inp))['X_hst'])
    k = 16
    X0 = X_hst[k, 0]; U = np.asarray(inp['U_arc_hst']).reshape(-1, 3)[k]; dt = float(x['arc_length_hst'][k])
    z = np.concatenate([X0, U])

    ref_L1, _ = make_levels(1e-14)
    X_ref, J_ref = (np.asarray(a) for a in jax.jit(ref_L1['L1'])(z, dt))
    dfx_L1, _ = make_levels(TOL)
    X_dfx, J_dfx = (np.asarray(a) for a in jax.jit(dfx_L1['L1'])(z, dt))
    rel = lambda a, b: np.max(np.abs(a - b)) / np.max(np.abs(b))

    print(f"Failing arc (NRHO cold start, arc {k}), against diffrax at tol 1e-14")
    print(f"    {'variant':34s} {'X_f error':>10s} {'J error':>10s} {'events':>7s} {'steps':>6s}")
    print(f"    {'diffrax @1e-12 (production)':34s} {rel(X_dfx, X_ref):10.1e} {rel(J_dfx, J_ref):10.1e}")
    for label, mode, ev in (('(a) select, no events (src today)', 'select', False),
                            ('(b) select + terminal events', 'select', True),
                            ('(c) flag + terminal events', 'flag', True)):
        it = Integ(pd, mode, 1, ev)
        res = it.run(X0, U, dt)
        print(f"    {label:34s} {rel(it.ta.state[:N], X_ref):10.1e} {rel(it.J(), J_ref):10.1e}"
              f" {it.n_events if mode == 'flag' else ('on' if ev else '-'):>7} {int(res[3]):6d}")

    # finite-difference check of (c)'s first derivatives: J consistent with its own X_f?
    it = Integ(pd, 'flag', 1, True); it.run(X0, U, dt); J_c = it.J()
    it0 = Integ(pd, 'flag', 0, True); fd = np.zeros((N, 10)); h = 1e-6
    for j in range(10):
        zp, zm = z.copy(), z.copy(); zp[j] += h; zm[j] -= h
        it0.run(zp[:N], zp[N:], dt); xp = it0.ta.state[:N].copy()
        it0.run(zm[:N], zm[N:], dt); xm = it0.ta.state[:N].copy()
        fd[:, j] = (xp - xm) / (2 * h)
    print(f"    (c) analytic J vs its own central FD (h=1e-6): {rel(J_c, fd):.1e}")

    # Cost on arcs that never enter the core (the 21 benchmark cases)
    print("\nCost of (c) on the 21 benchmark arcs (none enter the core), production compile modes")
    print(f"    {'order':6s} {'compile a / c':>16s} {'median time a / c':>20s} {'ratio':>6s} {'steps a / c (sum)':>18s}")
    cases = build_cases()
    for order in (0, 1, 2):
        out = {}
        for label, mode, ev in (('a', 'select', False), ('c', 'flag', True)):
            import time
            t0 = time.perf_counter(); it = Integ(pd, mode, order, ev); comp = time.perf_counter() - t0
            times, steps = [], 0
            for c in cases:
                res, _, steady = time_it(lambda: it.run(c['X0'], c['U'], c['dt']))
                times.append(steady); steps += int(res[3])
            out[label] = (comp, np.array(times), steps)
        (ca, ta_, sa), (cc, tc, sc) = out['a'], out['c']
        print(f"    {order:<6d} {ca:6.2f} / {cc:6.2f} s {np.median(ta_) * 1e3:8.3f} / {np.median(tc) * 1e3:7.3f} ms"
              f" {np.median(tc / ta_):6.2f} {sa:8d} / {sc:<8d}")
