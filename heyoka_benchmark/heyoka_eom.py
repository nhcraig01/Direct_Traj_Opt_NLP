# CR3BP equations of motion ported to heyoka.
#
# Same dynamics as src/dynamics/equations_of_motion.py's _cr3bp_eom_lambdify.
# The sympy construction is copied here (src/ stays untouched), but stopped
# BEFORE the .subs() that swaps in the Piecewise gravity softening: heyoka's
# from_sympy cannot convert Piecewise or relationals. Instead, from_sympy's
# s_dict maps the dtmp/rtmp placeholders straight to heyoka select()
# expressions built by hand.
#
# heyoka builds the ODE symbolically: variables come from make_vars, plain
# Python numbers (mu, a, b, ...) are baked in as constants, and hy.par[i] are
# runtime parameters that can change between integrations without recompiling
# - which is what the thrust command U is here.
#
# Running this file runs the RHS check against the production JAX EOM.

from cases import eom_constants

import heyoka as hy
import sympy as sp

_k = eom_constants()
U_Acc, ve, mu, safe_d = _k['U_Acc_min_nd'], _k['ve'], _k['mu'], _k['safe_d']

# sympy EOM, as in _cr3bp_eom_lambdify up to (not including) .subs(subs_dict) -
r_x, r_y, r_z, v_x, v_y, v_z = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z', real=True)
m = sp.symbols("m", positive=True)
u1, u2, u3 = sp.symbols('u1, u2, u3', real=True)
dtmp, rtmp = sp.symbols('dtmp, rtmp', positive=True)

eta = .75
alpha = 2
a = (3 * (eta * safe_d) ** (-3 - alpha)) / alpha
b = (eta * safe_d) ** (-3) + a * (eta * safe_d) ** alpha

u_norm = sp.sqrt(u1 ** 2 + u2 ** 2 + u3 ** 2 + 1e-12)

term1 = -(1 - mu) * (r_x + mu) * dtmp - mu * (r_x - 1 + mu) * rtmp + r_x
term2 = -(1 - mu) * r_y * dtmp - mu * r_y * rtmp + r_y
term3 = -(1 - mu) * r_z * dtmp - mu * r_z * rtmp

eoms = [v_x,
        v_y,
        v_z,
        term1 + 2 * v_y + u1 * U_Acc / m,
        term2 - 2 * v_x + u2 * U_Acc / m,
        term3 + u3 * U_Acc / m,
        -u_norm * U_Acc / ve]

# heyoka side -----------------------------------------------------------------
X = hy.make_vars("x", "y", "z", "vx", "vy", "vz", "m")

d = hy.sqrt((X[0] + mu) ** 2 + X[1] ** 2 + X[2] ** 2)
r = hy.sqrt((X[0] - 1 + mu) ** 2 + X[1] ** 2 + X[2] ** 2)

# Gravity softening. Python's <= doesn't work on heyoka expressions - the
# relationals are functions (lt, lte, gt, gte, eq). select() picks its branch
# from the state at the START of each Taylor step and uses that branch for the
# whole step; feasible trajectories never enter the core, so it never flips.
dmod = hy.select(hy.lte(d, eta * safe_d), -a * d ** alpha + b, d ** -3)
rmod = hy.select(hy.lte(r, eta * safe_d), -a * r ** alpha + b, r ** -3)

s_dict = {r_x: X[0], r_y: X[1], r_z: X[2], v_x: X[3], v_y: X[4], v_z: X[5], m: X[6],
          u1: hy.par[0], u2: hy.par[1], u3: hy.par[2],
          dtmp: dmod, rtmp: rmod}
rhs = [hy.from_sympy(e, s_dict) for e in eoms]

# heyoka's ODE format: a list of (variable, rhs) pairs
sys_ = list(zip(X, rhs))


if __name__ == "__main__":
    # RHS check ----------------------------------------------------------------
    # cfunc compiles plain expressions into a callable - no integration involved.
    import jax.numpy as jnp
    import numpy as np

    from cases import problem_def
    from src.dynamics.equations_of_motion import CR3BPDynamics

    cf = hy.cfunc(rhs, vars=X)
    jax_eom = CR3BPDynamics(problem_def())
    rng = np.random.default_rng(0)
    switch = eta * safe_d

    def random_state(center=None, rmin=None, rmax=None):
        if center is None:
            pos = rng.uniform(-1.5, 1.5, 3)
        else:
            dirn = rng.normal(size=3)
            pos = np.asarray(center) + dirn / np.linalg.norm(dirn) * rng.uniform(rmin, rmax)
        return np.concatenate([pos, rng.normal(size=3), [rng.uniform(0.5, 1.0)]])

    def random_control():
        return rng.normal(size=3) * rng.choice([0.0, 1e-6, 1.0])

    def truth(s, u):
        """40-digit evaluation of the same EOM, softening branch picked exactly."""
        v = [sp.Float(float(q), 40) for q in (*s, *u)]
        mu_, a_, b_, sw_ = (sp.Float(q, 40) for q in (mu, a, b, switch))

        def soft(dist):
            return -a_ * dist ** 2 + b_ if dist <= sw_ else dist ** -3

        D = sp.sqrt((v[0] + mu_) ** 2 + v[1] ** 2 + v[2] ** 2)
        R = sp.sqrt((v[0] - 1 + mu_) ** 2 + v[1] ** 2 + v[2] ** 2)
        vals = dict(zip((r_x, r_y, r_z, v_x, v_y, v_z, m, u1, u2, u3), v))
        vals.update({dtmp: soft(D), rtmp: soft(R)})
        return np.array([float(e.evalf(40, subs=vals)) for e in eoms])

    def rel(a_, b_):
        return np.max(np.abs(a_ - b_)) / np.max(np.abs(b_))

    print("RHS check: heyoka cfunc vs production JAX EOM vs 40-digit truth\n")

    states = [random_state() for _ in range(500)]
    worst = max(rel(cf(s, pars=u), np.asarray(jax_eom.eom(0.0, jnp.asarray(s), jnp.asarray(u))))
                for s, u in ((s, random_control()) for s in states))
    print(f"  broad random states (500)      heyoka vs JAX   {worst:.1e}")

    # Inside the softened core heyoka and JAX disagree at ~1e-11 - the truth
    # comparison shows that's the lambdified JAX EOM losing digits, not heyoka.
    for label, center in (("Moon", [1 - mu, 0, 0]), ("Earth", [-mu, 0, 0])):
        for where, (lo, hi) in (("inside core", (0.05, 0.999)), ("just outside", (1.001, 3.0))):
            hy_err, jax_err = 0.0, 0.0
            for _ in range(20):
                s, u = random_state(center, lo * switch, hi * switch), random_control()
                t = truth(s, u)
                hy_err = max(hy_err, rel(cf(s, pars=u), t))
                jax_err = max(jax_err, rel(np.asarray(jax_eom.eom(0.0, jnp.asarray(s), jnp.asarray(u))), t))
            print(f"  {label:5s} {where:12s} (20)    vs truth: heyoka {hy_err:.1e}   JAX {jax_err:.1e}")
