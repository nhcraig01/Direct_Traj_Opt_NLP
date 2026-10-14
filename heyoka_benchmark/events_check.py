# Check of the event-based gravity softening in src/dynamics/heyoka_propagator.py,
# at the NRHO cold-start point (seed 1, alpha/beta free) whose arc 16 crosses
# the Moon's softened core - where select() softening failed SNOPT's gradient
# check. Derivatives are checked against central finite differences of the
# same heyoka function (directional: jvp along a random direction), so a
# crossing that diffrax itself differentiates inaccurately can't hide an error.

from cases import compare

import jax
import jax.numpy as jnp
import numpy as np

from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.heyoka_propagator import HeyokaStateSensitivityPropagator
from src.dynamics.propagator import StateSensitivityPropagator
from src.optimization.init_guess import random_init_guess
from src.optimization.trajectory_optimization_problem import DeterministicTOP
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load

SCENARIO, SEED, H = 'L2_S-NRHO_to_L2_N-NRHO', 1, 1e-6


def directional_check(F, x, key):
    """max relative difference between jvp(F)(x; v) and the central FD of F
    along the same random direction v, over all outputs."""
    leaves, tree = jax.tree_util.tree_flatten(x)
    keys = jax.random.split(key, len(leaves))
    v = jax.tree_util.tree_unflatten(tree, [jax.random.normal(k, l.shape) * jnp.maximum(jnp.abs(l), 1.0)
                                            for k, l in zip(keys, leaves)])
    shift = lambda s: jax.tree_util.tree_map(lambda a, b: a + s * b, x, v)
    _, jvp = jax.jvp(F, (x,), (v,))
    fd = jax.tree_util.tree_map(lambda p, m: (p - m) / (2 * H), F(shift(H)), F(shift(-H)))
    return {k: compare(jvp[k], fd[k])[1] for k in jvp if np.max(np.abs(np.asarray(fd[k]))) > 0}


if __name__ == "__main__":
    cfg = yaml_load(f'Scenarios/{SCENARIO}.yaml')
    toggles = lambda integ, **kw: Toggles(integrator=integ, alpha_rng=(0.0, 1.0), beta_rng=(0.0, 1.0), **kw)
    tops = {i: DeterministicTOP(build_problem_def(cfg, toggles(i))) for i in ('diffrax', 'heyoka')}
    vars_ = tops['diffrax'].variables()
    ig = random_init_guess(vars_, jax.random.PRNGKey(SEED))
    x = {v.name: jnp.reshape(jnp.asarray(ig.get(v.name, v.value), dtype=jnp.float64), (v.size,)) for v in vars_}

    # 1. Deterministic, at the cold start ----------------------------------------
    pd = tops['diffrax']._problem_def
    inp = {'X0': x['X0'], 'Xf': x['Xf'], 'arc_length_hst': x['arc_length_hst'],
           'U_arc_hst': tops['diffrax']._control_repr.to_cartesian(pd, x['control_arc_hst'])}
    X = {i: np.asarray(t._propagator.propagate(t._problem_def, dict(inp))['X_hst']) for i, t in tops.items()}
    d = np.max(np.abs(X['heyoka'][:, -1] - X['diffrax'][:, -1]), axis=1)
    print(f"deterministic, {SCENARIO} cold start (arc 16 crosses the Moon's core)")
    print(f"    arc end states, heyoka vs diffrax: arcs 0-15 max {d[:16].max():.1e}, arc 16 {d[16]:.1e}, all arcs max {d.max():.1e}")

    ev = {i: jax.jit(t.evaluate) for i, t in tops.items()}
    an = {i: np.asarray(jax.jit(jax.jacfwd(t.evaluate))(x)['c_X_mp']['X0'])[:, 0] for i, t in tops.items()}
    xp, xm = dict(x), dict(x)
    xp['X0'], xm['X0'] = x['X0'].at[0].add(H), x['X0'].at[0].add(-H)
    fd = (np.asarray(ev['heyoka'](xp)['c_X_mp']) - np.asarray(ev['heyoka'](xm)['c_X_mp'])) / (2 * H)
    print(f"    the failed SNOPT check, dc_X_mp/dX0[0]:  heyoka analytic vs its FD {compare(an['heyoka'], fd)[1]:.1e},"
          f"  vs diffrax analytic {compare(an['heyoka'], an['diffrax'])[1]:.1e}")
    worst = directional_check(tops['heyoka'].evaluate, x, jax.random.PRNGKey(0))
    print(f"    evaluate(), jvp vs FD along a random direction: worst {max(worst.values()):.1e} ({max(worst, key=worst.get)})")

    # 2. Stochastic propagator (orders 1 and 2) at the same point -----------------
    pds = {i: build_problem_def(cfg, toggles(i, problem_type='stochastic_gauss_zoh')) for i in ('diffrax', 'heyoka')}
    props = {'diffrax': StateSensitivityPropagator(pds['diffrax'], CR3BPDynamics(pds['diffrax'])),
             'heyoka': HeyokaStateSensitivityPropagator(pds['heyoka'], CR3BPDynamics(pds['heyoka']))}
    OUT = ('X_hst', 'A_arc_hst', 'B_arc_hst', 'A_postinsert')
    F = {i: (lambda p, pd: lambda a: {k: v for k, v in p.propagate(pd, dict(a)).items() if k in OUT})(p, pds[i])
         for i, p in props.items()}
    vals = {i: jax.jit(f)(inp) for i, f in F.items()}
    print("\nstochastic propagator, same point")
    print("    values heyoka vs diffrax: " + ", ".join(
        f"{k} {compare(vals['heyoka'][k], vals['diffrax'][k])[1]:.1e}" for k in OUT))
    worst = directional_check(F['heyoka'], inp, jax.random.PRNGKey(1))
    print("    jvp vs FD along a random direction: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
