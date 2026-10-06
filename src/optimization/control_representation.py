"""ControlRepresentation hierarchy - 07b_control_representation.puml (tbd).

ControlRepresentation <|-- CartesianControl   (control_representation == 'cartesian')
ControlRepresentation <|-- RegularizedControl (control_representation == 'regularized')

Owns everything that differs between representations so DeterministicTOP/
StochasticTOP can stay branch-free regardless of which one is active - how
the control optimization variable is interpreted, the cartesian conversion
needed before dynamics propagation, the control-sensitivity jacobian
(identity for cartesian, non-trivial for regularized - feeds into both gain
parameterization and error propagation via sol_data['Hq_arc_hst'], never by
mutating B_hst/B_arc_hst in place), and the objective/constraint control-norm
terms (deterministic-only and stochastic).

There is exactly ONE control optimization variable, always named
'control_arc_hst', whatever units the active representation works in - no
per-representation variable names, so nothing downstream has to case-switch
to find it. TOP.evaluate() converts it once, at the top of the run, into
sol_data['U_arc_hst'] (physical cartesian thrust) via to_cartesian(); every
physically-cartesian consumer downstream - dynamics integration,
gates2Gexe's execution-error model, dV accounting, spherical plotting -
reads that derived key and never needs to know which representation
produced it.

ErrorPropagator is deliberately decoupled from this: it just stashes the
per-arc feedback-command dispersion in whichever form is natural for it -
P_control_arc_hst (full covariance) or S_control_arc_hst (sqrt factor, with
S @ S.T == P) - and the bound methods below consume whichever is present.
Nothing in the propagator hierarchy knows about control representation or
quantile levels, and nothing here needs an ErrorPropagator instance.
"""

import jax
import jax.numpy as jnp
from jax import Array

from src.problem.data_structures import OptimizationVariable
from src.problem.problem_definition import ProblemDefinition
from src.utils.math_utils import (
    U_cart_to_U_reg_vmap, U_reg_to_U_cart, U_reg_to_U_cart_vmap,
    chi2_inv_cdf, chi2_inv_cdf_vmap, mat_lmax_vmap,
    patnaik_rho_nu_from_cov, patnaik_rho_nu_from_cov_vmap,
)


def _arc_blocks(problem_def: ProblemDefinition, M: Array) -> Array:
    """Per-arc (m, ...) row blocks of a trajectory-wide control matrix.

    The lifted true-state propagators hand back whole-trajectory objects -
    P_Q (N_arcs*m square, with cross-arc blocks) or S_Q (N_arcs*m rows) -
    whose per-arc content lives in row block i. For P_Q the diagonal block
    (i, i) is that arc's own covariance; for S_Q row block i is its sqrt
    factor. Both are what the per-arc control-norm constraint wants."""
    dims = problem_def.dims
    m, N = dims.control_dim, dims.N_arcs
    rows = M.reshape(N, m, M.shape[-1])
    if M.shape[0] == M.shape[-1] and M.shape[-1] == N * m:
        return jax.vmap(lambda i, blk: jax.lax.dynamic_slice(blk, (0, i * m), (m, m)),
                        in_axes=(0, 0))(jnp.arange(N), rows)
    return rows


class ControlRepresentation:
    """+variables(problem_def, scale) : list[OptimizationVariable]
    +to_cartesian(problem_def, U_repr_flat) : array
    +control_jacobian_hst(problem_def, sol_data) : array
    +control_norm_term(problem_def, sol_data) : array
    +stochastic_objective_term(problem_def, sol_data, error_propagator) : array
    +stochastic_constraint_term(problem_def, sol_data, error_propagator) : array

    `scale` on variables() is the initial-guess std-dev multiplier - 1e-1 for
    DeterministicTOP, 1e0 for StochasticTOP, matching the existing (working)
    values for 'cartesian' today; not representation-dependent, so it's an
    argument here rather than baked into each subclass.
    """

    def variables(self, problem_def: ProblemDefinition, scale: float) -> list[OptimizationVariable]:
        raise NotImplementedError

    def to_cartesian(self, problem_def: ProblemDefinition, U_repr_flat: Array) -> Array:
        raise NotImplementedError

    def to_cartesian_single(self, problem_def: ProblemDefinition, U_repr_vec: Array) -> Array:
        """Single-arc, non-batched sibling of to_cartesian() - for MC-trial
        code, which operates per-arc in an unrolled Python loop rather than
        on the full (N_arcs, control_dim) array."""
        raise NotImplementedError

    def control_jacobian_hst(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        raise NotImplementedError

    def control_jacobian_single(self, problem_def: ProblemDefinition, U_repr_vec: Array) -> Array:
        """Single-arc, non-batched sibling of control_jacobian_hst() - for
        MC-trial code (the per-trial EKF's Hq_arc, evaluated at the fixed
        mean regularized command, identical to what control_jacobian_hst()
        computes for that same arc during optimization/deterministic
        propagation)."""
        raise NotImplementedError

    def control_norm_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        """Deterministic-only nominal control-norm term (length N_arcs) -
        consumed by ControlNormConstraint/objective() when there's no
        ErrorPropagator (plain DeterministicTOP)."""
        raise NotImplementedError

    def stochastic_objective(self, problem_def: ProblemDefinition, sol_data: dict,
                             weights: Array) -> float:
        """Scalar 99%-quantile dV over the whole transfer, weighting each
        arc's contribution by weights[i] (arc lengths for the optimization
        objective, dimensional U_acc/mass*dt*Vs for a reported dV_99).

        Returns a scalar rather than a per-arc array because the weighting
        has to happen inside the quantile for representations that combine
        arcs distributionally - see RegularizedControl."""
        raise NotImplementedError

    def stochastic_constraint_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        """Per-arc term (length N_arcs) for ControlNormConstraint's 99.9%
        quantile control-norm bound. Stays per-arc - thruster saturation is
        inherently an per-arc limit, not something to combine across arcs."""
        raise NotImplementedError


class CartesianControl(ControlRepresentation):
    """Direct thrust-vector optimization ('cartesian'). Nominal control-norm
    and stochastic TCM-dispersion terms combine additively - valid here
    because the stochastic TCM is zero-mean and adds linearly to the nominal
    thrust vector before the (separately handled) norm is taken."""

    def variables(self, problem_def: ProblemDefinition, scale: float) -> list[OptimizationVariable]:
        dims = problem_def.dims
        U_size = dims.control_dim * dims.N_arcs
        return [
            OptimizationVariable(
                name='control_arc_hst', size=U_size,
                lower=-jnp.ones(U_size), upper=jnp.ones(U_size), value=jnp.zeros(U_size),
                guess_fn=lambda key, prior: jnp.sqrt(scale) * jax.random.normal(key, shape=(U_size,)),
            ),
        ]

    def to_cartesian(self, problem_def: ProblemDefinition, U_repr_flat: Array) -> Array:
        return U_repr_flat

    def to_cartesian_single(self, problem_def: ProblemDefinition, U_repr_vec: Array) -> Array:
        return U_repr_vec

    def control_jacobian_hst(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        return jnp.broadcast_to(jnp.eye(dims.control_dim), (dims.N_arcs, dims.control_dim, dims.control_dim))

    def control_jacobian_single(self, problem_def: ProblemDefinition, U_repr_vec: Array) -> Array:
        return jnp.eye(problem_def.dims.control_dim)

    def control_norm_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        eps = 1e-12
        control_arc_hst = sol_data['control_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        return jnp.sqrt(control_arc_hst[:, 0] ** 2 + control_arc_hst[:, 1] ** 2
                        + control_arc_hst[:, 2] ** 2 + eps)

    @staticmethod
    def _control_dispersion(problem_def: ProblemDefinition, sol_data: dict) -> Array:
        """Per-arc sqrt(lambda_max(P_control)) - the max-direction 1-sigma
        feedback-command magnitude. Taken as sigma_max straight off a sqrt
        factor when the propagator supplied one (no squaring of the
        condition number), else from the covariance."""
        if 'S_Q' in sol_data:
            S_blocks = _arc_blocks(problem_def, sol_data['S_Q'])
            return jax.vmap(lambda S: jnp.linalg.norm(S, ord=2))(S_blocks)
        if 'P_Q' in sol_data:
            return jnp.sqrt(mat_lmax_vmap(_arc_blocks(problem_def, sol_data['P_Q'])))
        if 'S_control_arc_hst' in sol_data:
            return jax.vmap(lambda S: jnp.linalg.norm(S, ord=2))(sol_data['S_control_arc_hst'])
        return jnp.sqrt(mat_lmax_vmap(sol_data['P_control_arc_hst']))

    def stochastic_objective(self, problem_def: ProblemDefinition, sol_data: dict,
                             weights: Array) -> float:
        # No distributional object to combine here (the bound is a norm plus
        # a scaled dispersion, not a quadratic form), so arcs stay summed
        # independently - the weighted sum of per-arc bounds, as before.
        unc = problem_def.uncertainty
        per_arc = self.control_norm_term(problem_def, sol_data) + \
            unc.mx_dV_bound * self._control_dispersion(problem_def, sol_data)
        return per_arc @ weights

    def stochastic_constraint_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        unc = problem_def.uncertainty
        return self.control_norm_term(problem_def, sol_data) + \
            unc.mx_tcm_bound * self._control_dispersion(problem_def, sol_data)


class RegularizedControl(ControlRepresentation):
    """Regularized-thrust optimization ('regularized'). The nominal command
    and stochastic TCM dispersion do NOT combine additively here - the
    physical quantity of interest (||U_cart||) is the quadratic form q^T q of
    the total (non-central) regularized command, so mean and dispersion are
    combined via the Patnaik/scaled-chi-squared quantile fit (see
    math_utils.patnaik_rho_nu_from_cov), not added after the fact.

    Both the constraint and the objective are per-arc quantiles today, the
    objective simply weighted and summed. (The objective still returns a
    scalar and takes its weights as an argument so that a joint,
    all-arcs-at-once quantile - which needs the weights folded in before the
    Patnaik fit rather than applied after - can drop in without touching
    callers.)

    Works from either dispersion form the propagators supply - the Patnaik
    fit needs tr(P), tr(P^2) and q^T P q, all of which come from the compact
    (m, m) P, so a sqrt factor is simply squared up first (free at control
    dimension).
    """

    def variables(self, problem_def: ProblemDefinition, scale: float) -> list[OptimizationVariable]:
        dims = problem_def.dims
        U_size = dims.control_dim * dims.N_arcs
        return [
            OptimizationVariable(
                name='control_arc_hst', size=U_size,
                lower=-jnp.ones(U_size), upper=jnp.ones(U_size), value=jnp.zeros(U_size),
                guess_fn=lambda key, prior: U_cart_to_U_reg_vmap(
                    (jnp.sqrt(scale) * jax.random.normal(key, shape=(U_size,))).reshape(dims.N_arcs, dims.control_dim)
                ).flatten(),
            ),
        ]

    def to_cartesian(self, problem_def: ProblemDefinition, U_repr_flat: Array) -> Array:
        dims = problem_def.dims
        U_reg_arc_hst = U_repr_flat.reshape(dims.N_arcs, dims.control_dim)
        return U_reg_to_U_cart_vmap(U_reg_arc_hst).flatten()

    def to_cartesian_single(self, problem_def: ProblemDefinition, U_repr_vec: Array) -> Array:
        return U_reg_to_U_cart(U_repr_vec)

    def control_jacobian_hst(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        U_reg_arc_hst = sol_data['control_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        return jax.vmap(jax.jacfwd(U_reg_to_U_cart))(U_reg_arc_hst)

    def control_jacobian_single(self, problem_def: ProblemDefinition, U_repr_vec: Array) -> Array:
        return jax.jacfwd(U_reg_to_U_cart)(U_repr_vec)

    def control_norm_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        dims = problem_def.dims
        U_reg_arc_hst = sol_data['control_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        return jnp.sum(U_reg_arc_hst ** 2, axis=1)

    @staticmethod
    def _control_cov_arc_hst(problem_def: ProblemDefinition, sol_data: dict) -> Array:
        """Per-arc (m, m) feedback-command covariance, from whichever form
        the propagator supplied. Patnaik needs tr(P), tr(P^2) and q^T P q,
        all of which come from the compact P, so a sqrt factor is squared up
        here - cheap and no conditioning penalty at control dimension."""
        if 'S_Q' in sol_data:
            S = _arc_blocks(problem_def, sol_data['S_Q'])
            return jnp.einsum('aij,akj->aik', S, S)
        if 'P_Q' in sol_data:
            return _arc_blocks(problem_def, sol_data['P_Q'])
        if 'S_control_arc_hst' in sol_data:
            S = sol_data['S_control_arc_hst']
            return jnp.einsum('aij,akj->aik', S, S)
        return sol_data['P_control_arc_hst']

    @staticmethod
    def _control_cov_full(sol_data: dict) -> Array | None:
        """Whole-trajectory (N_arcs*m) control covariance INCLUDING cross-arc
        blocks, or None when the propagator only produced per-arc blocks.

        Presence is the signal that a joint, all-arcs-at-once quantile is
        legitimate: only the lifted true-state propagators carry the
        cross-correlations that a joint fit needs. The estimated-state
        propagators don't, so they fall back to the per-arc sum."""
        if 'S_Q' in sol_data:
            S = sol_data['S_Q']
            return S @ S.T
        return sol_data.get('P_Q')

    def _quantile_term(self, problem_def: ProblemDefinition, sol_data: dict, p: float) -> Array:
        """Per-arc p-quantile of the total command's quadratic form."""
        dims = problem_def.dims
        q_bar_hst = sol_data['control_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        rho_hst, nu_hst = patnaik_rho_nu_from_cov_vmap(
            q_bar_hst, self._control_cov_arc_hst(problem_def, sol_data))
        return rho_hst * chi2_inv_cdf_vmap(nu_hst, p)

    def stochastic_objective(self, problem_def: ProblemDefinition, sol_data: dict,
                             weights: Array) -> float:
        """dV_99 as ONE joint quantile over the whole trajectory when the
        propagator supplied cross-arc correlations, else the (conservative)
        weighted sum of per-arc quantiles.

        dV = sum_i w_i q_i^T q_i, so stack the weight-scaled commands
        p_i = sqrt(w_i) q_i: p^T p is exactly that sum, and
        p ~ N(sqrt(w_i) q_bar_i, D P_Q D) with D = diag(sqrt(w_i)) kron I_m.
        Patnaik applies to the augmented vector unchanged - it never assumed
        a dimension. Summing per-arc quantiles instead would add worst cases
        that cannot co-occur, an over-bound growing like N_arcs rather than
        sqrt(N_arcs)."""
        dims = problem_def.dims
        P_full = self._control_cov_full(sol_data)
        if P_full is None:
            return self._quantile_term(problem_def, sol_data, p=0.99) @ weights

        q_bar_hst = sol_data['control_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        d = jnp.repeat(jnp.sqrt(weights), dims.control_dim)     # diag of D
        q_aug = (jnp.sqrt(weights)[:, None] * q_bar_hst).flatten()
        P_aug = d[:, None] * P_full * d[None, :]

        rho, nu = patnaik_rho_nu_from_cov(q_aug, P_aug)
        return rho * chi2_inv_cdf(nu, 0.99)

    def stochastic_constraint_term(self, problem_def: ProblemDefinition, sol_data: dict) -> Array:
        return self._quantile_term(problem_def, sol_data, p=0.999)
