# Per-evaluation benchmark: production propagate() vs a heyoka drop-in.
#
# Times what SNOPT pays per call in a stochastic solve, shaped exactly like
# production: StateSensitivityPropagator.propagate on a real scenario's hot
# start - 20 forward + 20 backward chained arcs meeting at the midpoint, plus
# the post-insertion coast:
#
#   vals = jit(propagate)          every function evaluation (needs A, B: L1)
#   sens = jit(jacfwd(propagate))  every gradient evaluation, wrt all inputs
#                                  (X0, Xf, U, arc lengths: 174 directions)
#
# Two heyoka drop-ins, both jax.custom_jvp with an L1 pure_callback for the
# primal and an L2 pure_callback in the jvp rule, the tangent contraction done
# in JAX - so each arc integrates once however many directions come in:
#   'heyoka 41cb'  NAIVE worst case, one custom_jvp per arc: 41 host round
#                  trips per vals and per sens.
#   'heyoka 3cb'   BATCHED, one custom_jvp per chain of arcs (forward chain,
#                  backward chain, coast): 3 round trips, with the chaining of
#                  derivatives along each chain done in JAX with a scan.
#
# Covers propagate() only - covariance propagation and constraints downstream
# are identical JAX code on both sides, adding the same absolute time to each.

from cases import TOL, compare, eom_constants, time_it

import threading
import time

import h5py
import heyoka as hy
import jax
import jax.numpy as jnp
import numpy as np

import heyoka_eom
from heyoka_runs import ARGS, COMPACT, N, NZ, derivative_map, unpack
from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.propagator import AB_rev2fwd, StateSensitivityPropagator, node_times
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import yaml_load

SCENARIOS = ('Sandbox', 'L2_S-NRHO_to_L2_N-NRHO')
INPUTS = ('X0', 'Xf', 'U_arc_hst', 'arc_length_hst')
OUTPUTS = ('X_hst', 'A_arc_hst', 'B_arc_hst', 'A_postinsert')
N_NOOP = 41  # callbacks in the overhead measurement - one per arc + the coast


class HeyokaArc:
    """One arc, (X0, U, tau) -> (X_f, A, B), with tau the signed duration
    (negative = backward in time; the dynamics are autonomous). Integrated on
    the host through pure_callback, differentiable through custom_jvp."""

    def __init__(self, eom, X_init):
        self._eom = eom
        # XLA runs independent callbacks concurrently on different threads (the
        # forward and backward chains have no data dependence), and heyoka
        # releases the GIL while it integrates. Without this lock two chains
        # interleave reset/propagate on the same integrator and corrupt each
        # other's arcs.
        self._lock = threading.Lock()
        self._integrators = {}
        for level, order in (('L1', 1), ('L2', 2)):
            ta = hy.taylor_adaptive(hy.var_ode_sys(heyoka_eom.sys_, ARGS, order=order),
                                    np.asarray(X_init), pars=np.zeros(3), tol=TOL,
                                    compact_mode=COMPACT[level])
            J_map, H_map = derivative_map(ta)
            self._integrators[level] = (ta, ta.state[N:].copy(), J_map, H_map)

        self.arc = jax.custom_jvp(self._primal)
        self.arc.defjvp(self._jvp)

    # Host side ---------------------------------------------------------------
    def integrate(self, level, X0, U, tau):
        """Reset, propagate, unpack. Called by the callbacks, and directly to
        time the integration alone."""
        ta, var_ic, J_map, H_map = self._integrators[level]
        with self._lock:
            ta.time = 0.0
            ta.state[:N] = X0
            ta.state[N:] = var_ic
            ta.pars[:] = U
            ta.propagate_until(float(tau))
            out = unpack(ta.state, level, J_map, H_map)
        return (out['X_f'], out['J']) if level == 'L1' else (out['X_f'], out['J'], out['H'])

    # JAX side ----------------------------------------------------------------
    def _callback(self, level, X0, U, tau):
        shapes = [jax.ShapeDtypeStruct((N,), jnp.float64), jax.ShapeDtypeStruct((N, NZ), jnp.float64)]
        if level == 'L2':
            shapes.append(jax.ShapeDtypeStruct((N, NZ, NZ), jnp.float64))
        # The callback is handed JAX arrays, not numpy ones: convert ONCE up
        # front, or every index/slice in the host code goes through JAX's
        # per-op dispatch (measured ~5x slowdown on the integration itself).
        return jax.pure_callback(lambda *a: self.integrate(level, *map(np.asarray, a)),
                                 tuple(shapes), X0, U, tau, vmap_method='sequential')

    def _primal(self, X0, U, tau):
        X_f, J = self._callback('L1', X0, U, tau)
        return X_f, J[:, :N], J[:, N:]

    def _jvp(self, primals, tangents):
        X0, U, tau = primals
        dX0, dU, dtau = tangents
        X_f, J, H = self._callback('L2', X0, U, tau)
        dz = jnp.concatenate([dX0, dU])

        # d/dtau of X_f is the dynamics at the endpoint; of J it's the
        # variational equation there: d(dX_f/dX0)/dtau = f_x A,
        # d(dX_f/dU)/dtau = f_x B + f_u.
        f = self._eom.eom(0.0, X_f, U)
        f_x, f_u = jax.jacfwd(self._eom.eom, argnums=(1, 2))(0.0, X_f, U)
        dJ_dtau = jnp.concatenate([f_x @ J[:, :N], f_x @ J[:, N:] + f_u], axis=1)

        dX_f = J @ dz + f * dtau
        dJ = jnp.einsum('ijk,k->ij', H, dz) + dJ_dtau * dtau
        return (X_f, J[:, :N], J[:, N:]), (dX_f, dJ[:, :N], dJ[:, N:])


class HeyokaChain(HeyokaArc):
    """A chain of arcs, each starting where the previous one ended:
    (X_start, U (k, 3), tau (k,)) -> (X_ends (k, 7), A (k, 7, 7), B (k, 7, 3)).

    ONE host round trip runs the whole chain. The callback returns each arc's
    endpoint and LOCAL J (and H for the jvp); chaining the derivatives along
    the arcs - dX_i = J_i [dX_{i-1}; dU_i] + f_i dtau_i - is cheap linear
    algebra done in JAX with a scan, so batching costs no extra integration."""

    def __init__(self, eom, X_init):
        super().__init__(eom, X_init)
        self.chain = jax.custom_jvp(self._chain_primal)
        self.chain.defjvp(self._chain_jvp)

    def integrate_chain(self, level, X_start, U, tau):
        k = len(tau)
        X_ends, J = np.empty((k, N)), np.empty((k, N, NZ))
        H = np.empty((k, N, NZ, NZ)) if level == 'L2' else None
        x = np.asarray(X_start)
        for i in range(k):
            out = self.integrate(level, x, U[i], tau[i])
            X_ends[i], J[i] = out[0], out[1]
            if H is not None:
                H[i] = out[2]
            x = out[0]
        return (X_ends, J) if H is None else (X_ends, J, H)

    def _chain_callback(self, level, X_start, U, tau):
        k = tau.shape[0]
        shapes = [jax.ShapeDtypeStruct((k, N), jnp.float64), jax.ShapeDtypeStruct((k, N, NZ), jnp.float64)]
        if level == 'L2':
            shapes.append(jax.ShapeDtypeStruct((k, N, NZ, NZ), jnp.float64))
        # numpy up front - see HeyokaArc._callback
        return jax.pure_callback(lambda *a: self.integrate_chain(level, *map(np.asarray, a)),
                                 tuple(shapes), X_start, U, tau, vmap_method='sequential')

    def _chain_primal(self, X_start, U, tau):
        X_ends, J = self._chain_callback('L1', X_start, U, tau)
        return X_ends, J[:, :, :N], J[:, :, N:]

    def _chain_jvp(self, primals, tangents):
        X_start, U, tau = primals
        dX_start, dU, dtau = tangents
        X_ends, J, H = self._chain_callback('L2', X_start, U, tau)

        # Endpoint dynamics and their Jacobians, for every arc at once
        eom = lambda x, u: self._eom.eom(0.0, x, u)
        f = jax.vmap(eom)(X_ends, U)
        f_x, f_u = jax.vmap(jax.jacfwd(eom, argnums=(0, 1)))(X_ends, U)
        dJ_dtau = jnp.concatenate([f_x @ J[:, :, :N], f_x @ J[:, :, N:] + f_u], axis=2)

        def step(dX_prev, per_arc):
            J_i, H_i, f_i, dJ_dtau_i, dU_i, dtau_i = per_arc
            dz = jnp.concatenate([dX_prev, dU_i])
            dX_i = J_i @ dz + f_i * dtau_i
            dJ_i = jnp.einsum('ijk,k->ij', H_i, dz) + dJ_dtau_i * dtau_i
            return dX_i, (dX_i, dJ_i)

        _, (dX_ends, dJ) = jax.lax.scan(step, dX_start, (J, H, f, dJ_dtau, dU, dtau))
        return (X_ends, J[:, :, :N], J[:, :, N:]), (dX_ends, dJ[:, :, :N], dJ[:, :, N:])


def heyoka_propagate_batched(problem_def, hy_chain, sol_data):
    """Same outputs as heyoka_propagate, with 3 callbacks instead of 41: the
    forward chain from X0, the backward chain from Xf, and the coast."""
    dims, bc = problem_def.dims, problem_def.boundary_conditions
    assert dims.N_subarcs == 1, "heyoka_propagate_batched only handles one sub-arc per arc"

    X0, Xf = sol_data['X0'], sol_data['Xf']
    U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
    arc_length_hst = sol_data['arc_length_hst']
    indx_f, indx_b = bc.indx_f, bc.indx_b

    # Forward chain: arcs indx_f in order from X0
    X_f_ends, A_f, B_f = hy_chain.chain(X0, U_arc_hst[indx_f], arc_length_hst[indx_f])
    X_f_starts = jnp.concatenate([X0[None], X_f_ends[:-1]])

    # Backward chain: arcs indx_b (last arc first) from Xf, in reverse time
    X_b_ends, A_b, B_b = hy_chain.chain(Xf, U_arc_hst[indx_b], -arc_length_hst[indx_b])
    X_b_starts = jnp.concatenate([Xf[None], X_b_ends[:-1]])
    A_b_f, B_b_f = jax.vmap(AB_rev2fwd)(A_b, B_b)

    X_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, dims.state_dim))
    X_hst = X_hst.at[indx_f].set(jnp.stack([X_f_starts, X_f_ends], axis=1))
    X_hst = X_hst.at[indx_b].set(jnp.stack([X_b_ends, X_b_starts], axis=1))  # forward order
    A_arc_hst = jnp.zeros((dims.N_arcs, dims.state_dim, dims.state_dim)).at[indx_f].set(A_f).at[indx_b].set(A_b_f)
    B_arc_hst = jnp.zeros((dims.N_arcs, dims.state_dim, dims.control_dim)).at[indx_f].set(B_f).at[indx_b].set(B_b_f)

    t_node_bound = node_times(arc_length_hst)
    _, A_post, _ = hy_chain.chain(Xf, jnp.zeros((1, dims.control_dim)), jnp.full((1,), bc.tf_T))

    return {
        'X_hst': X_hst,
        't_hst': jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, dims.arc_length_opt))(
            t_node_bound[:-1], t_node_bound[1:]),
        't_node_bound': t_node_bound,
        'A_hst': A_arc_hst[:, None], 'B_hst': B_arc_hst[:, None],
        'A_arc_hst': A_arc_hst, 'B_arc_hst': B_arc_hst,
        'A_postinsert': A_post[0],
    }


def heyoka_propagate(problem_def, arc, sol_data):
    """StateSensitivityPropagator.propagate, line for line, with every arc's
    diffeqsolve + jacfwd replaced by one HeyokaArc call. N_subarcs == 1 only."""
    dims, bc = problem_def.dims, problem_def.boundary_conditions
    assert dims.N_subarcs == 1, "heyoka_propagate only handles one sub-arc per arc"

    X0, Xf = sol_data['X0'], sol_data['Xf']
    U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
    arc_length_hst = sol_data['arc_length_hst']

    X_hst = jnp.zeros((dims.N_arcs, dims.N_subarcs + 1, dims.state_dim))
    A_arc_hst = jnp.zeros((dims.N_arcs, dims.state_dim, dims.state_dim))
    B_arc_hst = jnp.zeros((dims.N_arcs, dims.state_dim, dims.control_dim))

    def fwd_body(i, carry):
        X0_f, X_hst, A_arc_hst, B_arc_hst = carry
        X_end, A_i, B_i = arc(X0_f, U_arc_hst[i], arc_length_hst[i])
        X_hst = X_hst.at[i].set(jnp.stack([X0_f, X_end]))
        return X_end, X_hst, A_arc_hst.at[i].set(A_i), B_arc_hst.at[i].set(B_i)

    _, X_hst, A_arc_hst, B_arc_hst = jax.lax.fori_loop(
        0, len(bc.indx_f), fwd_body, (X0, X_hst, A_arc_hst, B_arc_hst))

    def bwd_body(ii, carry):
        X0_b, X_hst, A_arc_hst, B_arc_hst = carry
        i = bc.indx_b[ii]
        X_end, A_i_b, B_i_b = arc(X0_b, U_arc_hst[i], -arc_length_hst[i])
        A_i_f, B_i_f = AB_rev2fwd(A_i_b, B_i_b)
        X_hst = X_hst.at[i].set(jnp.stack([X_end, X0_b]))  # flipped to forward order
        return X_end, X_hst, A_arc_hst.at[i].set(A_i_f), B_arc_hst.at[i].set(B_i_f)

    _, X_hst, A_arc_hst, B_arc_hst = jax.lax.fori_loop(
        0, len(bc.indx_b), bwd_body, (Xf, X_hst, A_arc_hst, B_arc_hst))

    t_node_bound = node_times(arc_length_hst)
    _, A_postinsert, _ = arc(Xf, jnp.zeros(dims.control_dim), bc.tf_T)

    return {
        'X_hst': X_hst,
        't_hst': jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, dims.arc_length_opt))(
            t_node_bound[:-1], t_node_bound[1:]),
        't_node_bound': t_node_bound,
        'A_hst': A_arc_hst[:, None], 'B_hst': B_arc_hst[:, None],
        'A_arc_hst': A_arc_hst, 'B_arc_hst': B_arc_hst,
        'A_postinsert': A_postinsert,
    }


def load_inputs(scenario):
    with h5py.File(f'Results/{scenario}/deterministic/sol.h5', 'r') as f:
        return {'X0': jnp.asarray(f['X0'][:]), 'Xf': jnp.asarray(f['Xf'][:]),
                'U_arc_hst': jnp.asarray(f['control_arc_hst'][:]),  # cartesian: control == U
                'arc_length_hst': jnp.asarray(f['arc_length_hst'][:])}


def compiled_and_time(fn, inputs):
    """(output, compile s, median steady s) for jit(fn)."""
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(inputs).compile()
    compile_s = time.perf_counter() - t0
    out, _, steady = time_it(compiled, inputs)
    return out, compile_s, steady


def worst_rel_diff(a, b):
    """Max normwise relative difference over matching leaves of two pytrees,
    skipping blocks that are exactly zero on the reference side."""
    worst = 0.0
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if np.max(np.abs(y)) > 0:
            worst = max(worst, compare(x, y)[1])
    return worst


if __name__ == "__main__":
    # Callback overhead on its own: N_NOOP do-nothing round trips in a compiled
    # loop, carrying a state-sized array.
    def noop_loop(x):
        def body(i, x):
            return jax.pure_callback(lambda v: v, jax.ShapeDtypeStruct(x.shape, x.dtype), x,
                                     vmap_method='sequential')
        return jax.lax.fori_loop(0, N_NOOP, body, x)

    _, _, noop_s = time_it(jax.jit(noop_loop).lower(jnp.zeros(N)).compile(), jnp.zeros(N))
    print(f"pure_callback overhead: {noop_s / N_NOOP * 1e6:.1f} us per round trip "
          f"({N_NOOP} no-op callbacks in a jitted loop: {noop_s * 1e3:.3f} ms)\n")

    for scenario in SCENARIOS:
        pd = build_problem_def(yaml_load(f'Scenarios/{scenario}.yaml'), Toggles())
        k = eom_constants()
        assert np.isclose(pd.spacecraft.U_Acc_min_nd, k['U_Acc_min_nd']) and np.isclose(
            pd.spacecraft.ve, k['ve']) and np.isclose(pd.Sys['mu'], k['mu']), \
            f"{scenario}'s engine/system constants differ from the ones heyoka_eom was built with"

        eom = CR3BPDynamics(pd)
        prop = StateSensitivityPropagator(pd, eom)
        inputs = load_inputs(scenario)
        hy_chain = HeyokaChain(eom, inputs['X0'])   # also provides the per-arc .arc

        variants = {
            'diffrax': lambda inp: prop.propagate(pd, dict(inp)),
            'heyoka 41cb': lambda inp: heyoka_propagate(pd, hy_chain.arc, inp),
            'heyoka 3cb': lambda inp: heyoka_propagate_batched(pd, hy_chain, inp),
        }

        results = {}
        for name, prop_fn in variants.items():
            F = lambda inp, prop_fn=prop_fn: {k_: v for k_, v in prop_fn(inp).items() if k_ in OUTPUTS}
            vals_out, vals_c, vals_s = compiled_and_time(F, inputs)
            sens_out, sens_c, sens_s = compiled_and_time(jax.jacfwd(F), inputs)
            results[name] = dict(vals=vals_out, sens=sens_out, vals_c=vals_c, vals_s=vals_s,
                                 sens_c=sens_c, sens_s=sens_s)

        # heyoka's integration alone: the same 41 arcs called directly on the
        # host, no JAX - the rest of each heyoka total is callbacks + JAX glue.
        X_hst = np.asarray(results['heyoka 3cb']['vals']['X_hst'])
        U = np.asarray(inputs['U_arc_hst']).reshape(-1, 3)
        dt = np.asarray(inputs['arc_length_hst'])
        bc = pd.boundary_conditions
        arcs = [(X_hst[i, 0], U[i], dt[i]) for i in np.asarray(bc.indx_f)]
        arcs += [(X_hst[i, 1], U[i], -dt[i]) for i in np.asarray(bc.indx_b)]
        arcs += [(np.asarray(inputs['Xf']), np.zeros(3), bc.tf_T)]
        host = {}
        for level in ('L1', 'L2'):
            _, _, host[level] = time_it(lambda: [hy_chain.integrate(level, *a) for a in arcs])

        d = results['diffrax']
        names = list(variants)
        print(f"{scenario}  ({pd.dims.N_arcs} arcs, dt = {dt[0]:.4f} nd, coast {bc.tf_T:.4f} nd)")
        print(f"    {'':22s}" + "".join(f"{n:>13s}" for n in names) + "   speedup 41cb / 3cb")
        for key, label in (('vals', 'vals (function eval)'), ('sens', 'sens (gradient eval)')):
            print(f"    {label:22s}" + "".join(f"{results[n][key + '_s'] * 1e3:10.2f} ms" for n in names)
                  + "".join(f"{d[key + '_s'] / results[n][key + '_s']:9.1f}x" for n in names[1:]))
        for key, label in (('vals', 'vals compile'), ('sens', 'sens compile')):
            print(f"    {label:22s}" + "".join(f"{results[n][key + '_c']:11.2f} s" for n in names))
        for key, level in (('vals', 'L1'), ('sens', 'L2')):
            print(f"    {key} integration alone: {host[level] * 1e3:.2f} ms ({len(arcs)} {level} arcs);"
                  f" callbacks/JAX on top:"
                  + "".join(f"  {n} {(results[n][key + '_s'] - host[level]) * 1e3:.2f} ms" for n in names[1:]))
        for n in names[1:]:
            print(f"    agreement {n} vs diffrax: vals {worst_rel_diff(results[n]['vals'], d['vals']):.1e},"
                  f"  sens {worst_rel_diff(results[n]['sens'], d['sens']):.1e}")
        h41, h3 = results['heyoka 41cb'], results['heyoka 3cb']
        print(f"    agreement 3cb vs 41cb:     vals {worst_rel_diff(h3['vals'], h41['vals']):.1e},"
              f"  sens {worst_rel_diff(h3['sens'], h41['sens']):.1e}")
        print()
