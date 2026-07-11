"""Equations of motion - 03_propagator.puml: EquationsOfMotion <|-- CR3BPDynamics."""

import sympy as sp
from jax import Array

from src.problem.problem_definition import ProblemDefinition

# Moon radius [km], used to smooth the CR3BP singularity near the secondary body.
_MOON_RADIUS_KM = 1737.5


def _cr3bp_eom_lambdify(U_Acc_min_nd: float, ve: float, mu: float, safe_d: float):
    """Build the jax-lambdified CR3BP equations of motion.

    Ported from Lib.dynamics.CR3BPDynamics(), dropping the Aprop_eval/
    Bprop_eval/U_st_eval/JC_eval outputs - confirmed unused by the
    sensitivity pipeline (A_hst/B_hst come from jax.jacfwd of
    propagate_arc(), and U_st_eval/JC_eval are unused entirely).
    """
    r_x, r_y, r_z, v_x, v_y, v_z = sp.symbols('r_x, r_y, r_z, v_x, v_y, v_z', real=True)
    t = sp.symbols("t")
    m = sp.symbols("m", positive=True)
    u1, u2, u3 = sp.symbols('u1, u2, u3', real=True)
    dtmp, rtmp = sp.symbols('dtmp, rtmp', positive=True)

    eta = .75
    alpha = 2
    a = (3 * (eta * safe_d) ** (-3 - alpha)) / alpha
    b = (eta * safe_d) ** (-3) + a * (eta * safe_d) ** alpha

    dval = sp.sqrt((r_x + mu) ** 2 + r_y ** 2 + r_z ** 2)
    rval = sp.sqrt((r_x - 1 + mu) ** 2 + r_y ** 2 + r_z ** 2)

    u_norm = sp.sqrt(u1 ** 2 + u2 ** 2 + u3 ** 2 + 1e-12)

    states = sp.Matrix([[r_x], [r_y], [r_z], [v_x], [v_y], [v_z], [m]])
    controls = sp.Matrix([[u1], [u2], [u3]])

    term1 = -(1 - mu) * (r_x + mu) * dtmp - mu * (r_x - 1 + mu) * rtmp + r_x
    term2 = -(1 - mu) * r_y * dtmp - mu * r_y * rtmp + r_y
    term3 = -(1 - mu) * r_z * dtmp - mu * r_z * rtmp

    dmod_term = -a * dval ** alpha + b
    rmod_term = -a * rval ** alpha + b

    dmod = sp.Piecewise((dmod_term, dval <= eta * safe_d), (1 / dval ** 3, dval > eta * safe_d))
    rmod = sp.Piecewise((rmod_term, rval <= eta * safe_d), (1 / rval ** 3, rval > eta * safe_d))
    subs_dict = {rtmp: rmod, dtmp: dmod}

    eoms = sp.Matrix([[v_x],
                      [v_y],
                      [v_z],
                      [term1 + 2 * v_y + u1 * U_Acc_min_nd / m],
                      [term2 - 2 * v_x + u2 * U_Acc_min_nd / m],
                      [term3 + u3 * U_Acc_min_nd / m],
                      [-u_norm * U_Acc_min_nd / ve]]).subs(subs_dict)

    return sp.lambdify((t, states, controls), eoms, 'jax')


class EquationsOfMotion:
    """+eom(t, state, args) : array"""

    def eom(self, t: float, state: Array, args: Array) -> Array:
        raise NotImplementedError


class CR3BPDynamics(EquationsOfMotion):
    """Today's (and currently only) dynamical model."""

    def __init__(self, problem_def: ProblemDefinition):
        moon_radius_nd = _MOON_RADIUS_KM / problem_def.Sys['Ls']
        self._eom_eval = _cr3bp_eom_lambdify(
            problem_def.spacecraft.U_Acc_min_nd,
            problem_def.spacecraft.ve,
            problem_def.Sys['mu'],
            moon_radius_nd,
        )

    def eom(self, t: float, state: Array, args: Array) -> Array:
        return self._eom_eval(t, state, args).reshape(-1)
