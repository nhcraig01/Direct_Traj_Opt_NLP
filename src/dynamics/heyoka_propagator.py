"""heyoka (Taylor-method) propagators - drop-in replacements for the diffrax ones
in propagator.py, selected with Toggles.integrator == 'heyoka'.

Each subclasses its diffrax counterpart and overrides only propagate(), so the
diffrax internals (_propagate_arc, _arc_dAB, ...) that mc_runner.py and
detailed_trajectory.py call are inherited unchanged: optimization runs on
heyoka, Monte Carlo and data generation keep replaying with diffrax.

heyoka isn't JAX. Each chain of sub-arcs - the forward arcs from X0, the
backward arcs from Xf, the post-insertion coast - is ONE jax.pure_callback that
integrates the whole chain on the host, wrapped in a jax.custom_jvp so JAX can
differentiate through it. The callback returns every sub-arc's endpoint and its
LOCAL sensitivities from heyoka's variational equations; chaining those along
the arcs (dX_i = J_i [dX_{i-1}; dU_i] + f_i dtau_i) is cheap linear algebra done
in JAX. So each sub-arc is integrated once however many tangent directions
jacfwd pushes through - which is the speedup over forward-mode AD through
diffrax. Benchmarks and derivation: heyoka_benchmark/.

heyoka is imported here at load time; trajectory_optimization_problem.py only
imports this module when the heyoka integrator is selected, so environments
without heyoka still work.
"""

import threading
import warnings

import heyoka as hy
import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from .equations_of_motion import EquationsOfMotion, cr3bp_eom_constants, cr3bp_eom_sympy
from .propagator import AB_rev2fwd_vmap, StatePropagator, StateSensitivityPropagator, node_times
from src.problem.problem_definition import ProblemDefinition

_EVENT_COOLDOWN = 1e-9  # nd time; see _Integrator


def heyoka_cr3bp_system(problem_def: ProblemDefinition):
    """(ODE system, variational args, number of states, softening events) for
    heyoka, built from the same sympy EOM as the jax build. States map to heyoka
    variables and controls to runtime parameters par[i], so U changes per arc
    without recompiling.

    Gravity softening: heyoka's select() would fix the branch at the start of
    each Taylor step and nothing would end a step at the switch radius, so arcs
    crossing the softened core came out wrong. Instead each Piecewise becomes
    flag * inner + (1 - flag) * outer, with the flag a runtime parameter after
    the controls, set by a terminal event on the switch condition (see
    _Integrator). The events come back as (heyoka expressions, numpy function of
    the states) - the latter sets the flags at the start of each sub-arc."""
    _, states, controls, eoms, softening = cr3bp_eom_sympy(*cr3bp_eom_constants(problem_def))
    X = hy.make_vars(*[str(s) for s in states])
    pars = [hy.par[i] for i in range(len(controls))]
    base = dict(zip([*states, *controls], [*X, *pars]))

    def to_hy(ex):
        return hy.from_sympy(sp.sympify(ex), base)

    s_dict, events = dict(base), []
    for k, (placeholder, pw) in enumerate(softening.items()):
        (inner, cond), (outer, _) = pw.args  # sympy canonicalises the second condition to True
        assert isinstance(cond, (sp.LessThan, sp.StrictLessThan)), "first branch must be the inside of the core"
        flag = hy.par[len(controls) + k]
        s_dict[placeholder] = flag * to_hy(inner) + (1.0 - flag) * to_hy(outer)
        events.append(cond.lhs ** 2 - cond.rhs ** 2)  # same root as the distance, no sqrt; <= 0 inside

    sys_ = list(zip(X, (hy.from_sympy(e, s_dict) for e in eoms)))
    event_fn = sp.lambdify([list(states)], events, 'numpy')
    return sys_, [*X, *pars], len(X), ([to_hy(g) for g in events], event_fn)


class _Integrator:
    """One heyoka integrator at a given variational order (0 = state only), reused
    for every sub-arc of one chain - never shared between chains: XLA runs
    independent callbacks concurrently and heyoka releases the GIL while it
    integrates, so two chains on one integrator would reset each other mid-arc.
    The lock covers anything else that might call into the same chain.

    Each softening switch is a terminal event whose callback sets that primary's
    flag from the crossing direction and lets the integration carry on. heyoka
    reports the direction as the sign of d(event)/dt in TIME, so backward
    sub-arcs multiply by the sign of tau. The cooldown is explicit: heyoka's
    automatic one scales with the largest entry of the whole state, variational
    part included, which at order 2 outlasted a core crossing and missed the
    exit. A spurious re-trigger is harmless - the callback sets, never toggles."""

    def __init__(self, sys_, args, n, order, tol, events, event_fn):
        self._n, self._nz, self._order = n, len(args), order
        self._n_u, self._event_fn, self._direction = len(args) - n, event_fn, 1.0
        if order:
            sys_ = hy.var_ode_sys(sys_, args, order=order)
        t_events = [hy.t_event(g, callback=self._crossing(k), cooldown=_EVENT_COOLDOWN)
                    for k, g in enumerate(events)]
        # Unrolled (non-compact) code runs 1.4-2.3x faster at orders 0/1 for a
        # one-time compile of a few seconds, cached on disk by heyoka. At order 2
        # it doesn't finish compiling in useful time, so that stays compact.
        X_init = np.zeros(n)
        X_init[-1] = 1.0  # any valid state - every run resets it
        self._ta = hy.taylor_adaptive(sys_, X_init, pars=np.zeros(self._n_u + len(events)), tol=tol,
                                      compact_mode=(order == 2), t_events=t_events)
        # heyoka sets the variational initial conditions (identity for d/dX0,
        # zero for everything else) only at construction: keep them to restore.
        self._var_ic = self._ta.state[n:].copy()
        self._lock = threading.Lock()

        # get_mindex(idx) = [component, n_0, ..., n_{nz-1}]: the derivative order
        # wrt each arg. First-order entries scatter into J[comp, j], second-order
        # into H[comp, j, k] (mixed ones are stored once - H gets both halves).
        J_map, H_map = [], []
        for idx in range(n, len(self._ta.state)):
            mi = self._ta.get_mindex(idx)
            comp, nz_args = mi[0], np.nonzero(mi[1:])[0]
            if sum(mi[1:]) == 1:
                J_map.append((idx, comp, nz_args[0]))
            else:
                j, k = (nz_args[0], nz_args[0]) if len(nz_args) == 1 else nz_args
                H_map.append((idx, comp, j, k))
        self._J_map = np.array(J_map, dtype=int).reshape(-1, 3)
        self._H_map = np.array(H_map, dtype=int).reshape(-1, 4)

    def _crossing(self, k):
        def callback(ta, d_sgn):
            ta.pars[self._n_u + k] = float(d_sgn * self._direction < 0)  # entering -> softened
            return True
        return callback

    def _inside(self, x):
        return (np.asarray(self._event_fn(x)) <= 0).astype(float)

    def chain(self, X_start, U, tau):
        """Integrate sub-arcs back to back on the host. numpy in, numpy out:
        (X_ends,) plus J (k, n, nz) at order >= 1, plus H (k, n, nz, nz) at 2."""
        k, n, nz = len(tau), self._n, self._nz
        X_ends = np.empty((k, n))
        J = np.zeros((k, n, nz)) if self._order >= 1 else None
        H = np.zeros((k, n, nz, nz)) if self._order >= 2 else None
        x = X_start
        for i in range(k):
            with self._lock:
                self._ta.time = 0.0
                self._ta.state[:n] = x
                self._ta.state[n:] = self._var_ic
                self._ta.pars[:] = np.concatenate([U[i], self._inside(x)])
                self._direction = 1.0 if tau[i] >= 0 else -1.0
                self._ta.reset_cooldowns()
                self._ta.propagate_until(float(tau[i]))
                s = self._ta.state
                X_ends[i] = s[:n]
                if np.any(self._inside(X_ends[i]) != self._ta.pars[self._n_u:]):
                    warnings.warn("heyoka: a gravity-softening flag disagrees with where a sub-arc ended - "
                                  "a switch-radius crossing was missed (grazing?), so that sub-arc is wrong")
                if J is not None:
                    J[i, self._J_map[:, 1], self._J_map[:, 2]] = s[self._J_map[:, 0]]
                if H is not None:
                    vals = s[self._H_map[:, 0]]
                    H[i, self._H_map[:, 1], self._H_map[:, 2], self._H_map[:, 3]] = vals
                    H[i, self._H_map[:, 1], self._H_map[:, 3], self._H_map[:, 2]] = vals
            x = X_ends[i]
        return tuple(a for a in (X_ends, J, H) if a is not None)


class HeyokaChain:
    """A chain of sub-arcs, each starting where the previous one ended, as a
    differentiable JAX function of (X_start (n,), U (k, m), tau (k,)):

        order 0:  X_ends (k, n)
        order 1:  X_ends, A (k, n, n), B (k, n, m)    (local, per sub-arc)

    tau is each sub-arc's signed duration - negative integrates backward in
    time; the dynamics are autonomous. The primal runs a heyoka integrator of
    this order, the jvp rule one of order + 1, and chains the local
    sensitivities in JAX. Each chain owns its own integrators (see _Integrator).
    """

    def __init__(self, problem_def: ProblemDefinition, eom: EquationsOfMotion, order: int):
        sys_, args, n, softening = heyoka_cr3bp_system(problem_def)
        tol = problem_def.integration.r_tol
        self._eom, self._order, self._n, self._nz = eom, order, n, len(args)
        self._primal_int = _Integrator(sys_, args, n, order, tol, *softening)
        self._jvp_int = _Integrator(sys_, args, n, order + 1, tol, *softening)

        self.chain = jax.custom_jvp(self._primal)
        self.chain.defjvp(self._jvp)

    def _callback(self, integrator, order, X_start, U, tau):
        k, n, nz = tau.shape[0], self._n, self._nz
        shapes = [(k, n), (k, n, nz), (k, n, nz, nz)][:order + 1]
        # The callback is handed JAX arrays: convert to numpy ONCE, or every
        # index in the host loop pays a JAX dispatch (5x slower integration).
        return jax.pure_callback(
            lambda *a: integrator.chain(*map(np.asarray, a)),
            tuple(jax.ShapeDtypeStruct(s, jnp.float64) for s in shapes),
            X_start, U, tau, vmap_method='sequential',
        )

    def _split(self, J):
        return J[:, :, :self._n], J[:, :, self._n:]

    def _primal(self, X_start, U, tau):
        out = self._callback(self._primal_int, self._order, X_start, U, tau)
        return out[0] if self._order == 0 else (out[0], *self._split(out[1]))

    def _jvp(self, primals, tangents):
        X_start, U, tau = primals
        dX_start, dU, dtau = tangents
        out = self._callback(self._jvp_int, self._order + 1, X_start, U, tau)
        X_ends, J = out[0], out[1]

        # d/dtau of a sub-arc's endpoint is the dynamics there
        eom = lambda x, u: self._eom.eom(0.0, x, u)
        f = jax.vmap(eom)(X_ends, U)

        if self._order == 0:
            def step(dX_prev, per_arc):
                J_i, f_i, dU_i, dtau_i = per_arc
                dX_i = J_i @ jnp.concatenate([dX_prev, dU_i]) + f_i * dtau_i
                return dX_i, dX_i

            _, dX_ends = jax.lax.scan(step, dX_start, (J, f, dU, dtau))
            return X_ends, dX_ends

        # Order 1: J's own tau-derivative is the variational equation at the
        # endpoint - d(dX/dX0)/dtau = f_x A, d(dX/dU)/dtau = f_x B + f_u.
        H = out[2]
        f_x, f_u = jax.vmap(jax.jacfwd(eom, argnums=(0, 1)))(X_ends, U)
        A, B = self._split(J)
        dJ_dtau = jnp.concatenate([f_x @ A, f_x @ B + f_u], axis=2)

        def step(dX_prev, per_arc):
            J_i, H_i, f_i, dJ_dtau_i, dU_i, dtau_i = per_arc
            dz = jnp.concatenate([dX_prev, dU_i])
            dX_i = J_i @ dz + f_i * dtau_i
            dJ_i = jnp.einsum('ijk,k->ij', H_i, dz) + dJ_dtau_i * dtau_i
            return dX_i, (dX_i, dJ_i)

        _, (dX_ends, dJ) = jax.lax.scan(step, dX_start, (J, H, f, dJ_dtau, dU, dtau))
        return (X_ends, A, B), (dX_ends, *self._split(dJ))


def _chain_inputs(problem_def: ProblemDefinition, sol_data: dict, arcs, direction: float):
    """Per-sub-arc (U, tau) for a chain over `arcs`, in the order integrated.
    Sub-arcs split each arc evenly in time, as _propagate_arc's save points do."""
    dims = problem_def.dims
    U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
    dt_sub = sol_data['arc_length_hst'] / dims.N_subarcs
    return (jnp.repeat(U_arc_hst[arcs], dims.N_subarcs, axis=0),
            direction * jnp.repeat(dt_sub[arcs], dims.N_subarcs))


def _arc_states(X_start, X_ends, n_arcs: int, N_subarcs: int):
    """Chain endpoints -> per-arc histories [arc start, sub-arc ends...], in the
    chain's own direction of travel: (n_arcs, N_subarcs + 1, n)."""
    ends = X_ends.reshape(n_arcs, N_subarcs, -1)
    starts = jnp.concatenate([X_start[None], ends[:-1, -1]])
    return jnp.concatenate([starts[:, None], ends], axis=1)


def _per_arc(M, n_arcs: int, N_subarcs: int):
    """Chain-ordered sub-arc rows -> (n_arcs, N_subarcs, ...)."""
    return M.reshape(n_arcs, N_subarcs, *M.shape[1:])


def _compose_subarcs(A_sub, B_sub):
    """An arc's forward-time A, B from its sub-arcs' (in forward order):
    A = A_N ... A_1, B = sum_j (A_N ... A_{j+1}) B_j - the chain rule, no
    extra integration. Exact for one sub-arc (I @ A and 0 + B are exact)."""
    def step(carry, AB_j):
        A_acc, B_acc = carry
        A_j, B_j = AB_j
        return (A_j @ A_acc, A_j @ B_acc + B_j), None

    (A, B), _ = jax.lax.scan(step, (jnp.eye(A_sub.shape[-1]), jnp.zeros_like(B_sub[0])), (A_sub, B_sub))
    return A, B


class HeyokaStatePropagator(StatePropagator):
    """StatePropagator.propagate on heyoka: forward chain from X0, backward chain
    from Xf, meeting at the midpoint. Same outputs, same shapes."""

    def __init__(self, problem_def: ProblemDefinition, eom: EquationsOfMotion):
        super().__init__(problem_def, eom)
        self._fwd = HeyokaChain(problem_def, eom, order=0)
        self._bwd = HeyokaChain(problem_def, eom, order=0)

    def propagate(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        bc = problem_def.boundary_conditions
        indx_f, indx_b = bc.indx_f, bc.indx_b

        X_ends_f = self._fwd.chain(sol_data['X0'], *_chain_inputs(problem_def, sol_data, indx_f, 1.0))
        X_ends_b = self._bwd.chain(sol_data['Xf'], *_chain_inputs(problem_def, sol_data, indx_b, -1.0))

        X_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, dims.state_dim))
        X_hst = X_hst.at[indx_f].set(_arc_states(sol_data['X0'], X_ends_f, len(indx_f), dims.N_subarcs))
        # backward arcs were integrated end-to-start: flip each into forward order
        X_hst = X_hst.at[indx_b].set(
            jnp.flip(_arc_states(sol_data['Xf'], X_ends_b, len(indx_b), dims.N_subarcs), axis=1))

        t_node_bound = node_times(sol_data['arc_length_hst'])
        t_hst = jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, dims.arc_length_opt))(t_node_bound[:-1], t_node_bound[1:])

        sol_data['X_hst'] = X_hst
        sol_data['t_hst'] = t_hst
        sol_data['t_node_bound'] = t_node_bound
        return sol_data


class HeyokaStateSensitivityPropagator(StateSensitivityPropagator):
    """StateSensitivityPropagator.propagate on heyoka - same outputs, same shapes:
    X_hst, A_hst/B_hst per sub-arc, A_arc_hst/B_arc_hst per arc, A_postinsert.

    Sub-arc A/B come straight from heyoka's variational equations; arc-level
    A/B are composed from them (_compose_subarcs). Three chains, each with its
    own integrators: forward from X0, backward from Xf, and the coast."""

    def __init__(self, problem_def: ProblemDefinition, eom: EquationsOfMotion):
        super().__init__(problem_def, eom)
        self._fwd = HeyokaChain(problem_def, eom, order=1)
        self._bwd = HeyokaChain(problem_def, eom, order=1)
        self._coast = HeyokaChain(problem_def, eom, order=1)

    def propagate(self, problem_def: ProblemDefinition, sol_data: dict) -> dict:
        dims = problem_def.dims
        bc = problem_def.boundary_conditions
        indx_f, indx_b = bc.indx_f, bc.indx_b
        n_f, n_b, n_sub = len(indx_f), len(indx_b), dims.N_subarcs

        X_ends_f, A_f, B_f = self._fwd.chain(sol_data['X0'], *_chain_inputs(problem_def, sol_data, indx_f, 1.0))
        X_ends_b, A_b, B_b = self._bwd.chain(sol_data['Xf'], *_chain_inputs(problem_def, sol_data, indx_b, -1.0))

        # Backward sub-arcs map their later state to their earlier one: convert
        # each to the forward-time map, then put each arc's sub-arcs back in
        # forward order.
        A_b, B_b = AB_rev2fwd_vmap(A_b, B_b)
        A_hst = jnp.zeros((dims.N_arcs, n_sub, dims.state_dim, dims.state_dim))
        B_hst = jnp.zeros((dims.N_arcs, n_sub, dims.state_dim, dims.control_dim))
        A_hst = A_hst.at[indx_f].set(_per_arc(A_f, n_f, n_sub)).at[indx_b].set(jnp.flip(_per_arc(A_b, n_b, n_sub), axis=1))
        B_hst = B_hst.at[indx_f].set(_per_arc(B_f, n_f, n_sub)).at[indx_b].set(jnp.flip(_per_arc(B_b, n_b, n_sub), axis=1))
        A_arc_hst, B_arc_hst = jax.vmap(_compose_subarcs)(A_hst, B_hst)

        X_hst = jnp.zeros((dims.N_arcs, n_sub + 1, dims.state_dim))
        X_hst = X_hst.at[indx_f].set(_arc_states(sol_data['X0'], X_ends_f, n_f, n_sub))
        X_hst = X_hst.at[indx_b].set(jnp.flip(_arc_states(sol_data['Xf'], X_ends_b, n_b, n_sub), axis=1))

        t_node_bound = node_times(sol_data['arc_length_hst'])
        t_hst = jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, dims.arc_length_opt))(t_node_bound[:-1], t_node_bound[1:])

        # Post-insertion coast: zero control for the fixed duration tf_T
        _, A_post, _ = self._coast.chain(sol_data['Xf'], jnp.zeros((1, dims.control_dim)), jnp.full((1,), bc.tf_T))

        sol_data['X_hst'] = X_hst
        sol_data['t_hst'] = t_hst
        sol_data['t_node_bound'] = t_node_bound
        sol_data['A_hst'] = A_hst
        sol_data['B_hst'] = B_hst
        sol_data['A_arc_hst'] = A_arc_hst
        sol_data['B_arc_hst'] = B_arc_hst
        sol_data['A_postinsert'] = A_post[0]
        return sol_data
