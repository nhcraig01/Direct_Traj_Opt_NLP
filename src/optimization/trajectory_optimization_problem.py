"""Trajectory optimization problem hierarchy.

TrajectoryOptimizationProblem: abstract base class with concrete propagate(),
constraints(), and to_pyoptsparse() shared by all subclasses.

DeterministicTOP: simplest concrete implementation — no GainParameterization,
no ErrorPropagator.
"""

import jax
import jax.numpy as jnp
from pyoptsparse import Optimization

from .constraints import (
    Constraint,
    BoundaryConstraint, ControlNormConstraint, CovarianceConstraint, DetCollisionAvoidanceConstraint,
    MatchPointConstraint, MeshConstraint,
)
from .control_representation import CartesianControl, ControlRepresentation, RegularizedControl
from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.propagator import Propagator, StatePropagator, StateSensitivityPropagator
from src.problem.data_structures import OptimizationVariable
from src.problem.problem_definition import ProblemDefinition
from src.stochastic.error_propagator import (
    EstimatedStateCovPropagator, EstimatedStateSqrtCovPropagator,
    TrueStateCovPropagator, TrueStateSqrtCovPropagator,
)
from src.stochastic.gain_parameterization import LQRArcGains, LQRFullTrajGains
from src.utils.console import EvalTable
from src.utils.io import process_sparsity

_CONTROL_REPR_CLS = {'cartesian': CartesianControl, 'regularized': RegularizedControl}


def _build_control_repr(problem_def: ProblemDefinition) -> ControlRepresentation:
    control_representation = problem_def.toggles.control_representation.lower()
    cls = _CONTROL_REPR_CLS.get(control_representation)
    if cls is None:
        raise ValueError(f"Unknown control_representation: {control_representation!r}")
    return cls()


def _build_propagator(problem_def: ProblemDefinition, sensitivities: bool) -> Propagator:
    """State-only (DeterministicTOP) or state + sensitivity (StochasticTOP)
    propagator for toggles.integrator."""
    integrator = problem_def.toggles.integrator.lower()
    if integrator == 'diffrax':
        cls = StateSensitivityPropagator if sensitivities else StatePropagator
    elif integrator == 'heyoka':
        # Imported here so heyoka is only needed when it's selected.
        from src.dynamics.heyoka_propagator import HeyokaStatePropagator, HeyokaStateSensitivityPropagator
        cls = HeyokaStateSensitivityPropagator if sensitivities else HeyokaStatePropagator
    else:
        raise ValueError(f"Unknown integrator: {integrator!r}")
    return cls(problem_def, CR3BPDynamics(problem_def))


class TrajectoryOptimizationProblem:
    # Progress table columns (console.EvalTable, fed from evaluate()): the
    # objective terms, then each constraint in _PRINT_CONSTRAINTS that this
    # problem has, then any subclass variable columns.
    _OBJECTIVE_COLUMNS = ('J',)
    _VARIABLE_COLUMNS = ()
    _PRINT_CONSTRAINTS = (
        ('c_X0', 'X0'), ('c_Xf', 'Xf'), ('c_X_mp', 'X_mp'),
        ('c_P_Xf', 'P_Xf'), ('c_Us', 'Us'), ('c_det_col_avoid', 'Col'),
    )

    def __init__(self, problem_def: ProblemDefinition, propagator: Propagator, constraints_list: list[Constraint]):
        self._problem_def = problem_def
        self._propagator = propagator
        self._constraints_list = constraints_list

        # Equality constraints print as max|c|, inequality ones as max(c - upper)
        # so every inequality column reads "<= 0 is satisfied" whatever its
        # bound. Linear constraints aren't in evaluate()'s output, so they're out.
        printed = {c.name: c for c in constraints_list if not c.linearities}
        present = [(printed[n], label) for n, label in self._PRINT_CONSTRAINTS if n in printed]
        self._print_eq = [(c, l) for c, l in present if bool(jnp.all(c.lower == c.upper))]
        self._print_ineq = [(c, l) for c, l in present if not bool(jnp.all(c.lower == c.upper))]
        self.progress = EvalTable(list(self._OBJECTIVE_COLUMNS), [l for _, l in self._print_eq],
                                  [l for _, l in self._print_ineq], list(self._VARIABLE_COLUMNS))

    def variables(self) -> list[OptimizationVariable]:
        raise NotImplementedError

    def propagate(self, sol_data: dict) -> dict:
        return self._propagator.propagate(self._problem_def, sol_data)

    def objective(self, sol_data: dict) -> float:
        raise NotImplementedError

    def constraints(self) -> list[Constraint]:
        return self._constraints_list

    def evaluate(self, inputs: dict) -> dict:
        raise NotImplementedError

    def _constraint_progress(self, output: dict) -> list:
        """The constraint columns of the progress table, in table order."""
        return ([jnp.max(jnp.abs(output[c.name] - c.lower)) for c, _ in self._print_eq]
                + [jnp.max(output[c.name] - c.upper) for c, _ in self._print_ineq])

    def to_pyoptsparse(self, init_guess: dict) -> tuple[Optimization, callable]:
        vals = jax.jit(self.evaluate, backend='cpu')
        optprob = Optimization(type(self).__name__, vals)

        for v in self.variables():
            value = init_guess.get(v.name, v.value)
            lower, upper = (value, value) if v.pinned else (v.lower, v.upper)
            optprob.addVarGroup(v.name, v.size, "c", value=value, lower=lower, upper=upper)

        optprob.addObj('o')

        grad = jax.jit(jax.jacfwd(self.evaluate), backend='cpu')
        eval_point = {v.name: init_guess.get(v.name, v.value) for v in self.variables()}
        grad_sparse = process_sparsity(grad(eval_point))

        for c in self.constraints():
            if c.linearities:
                optprob.addConGroup(
                    c.name, c.size, lower=c.lower, upper=c.upper,
                    linear=True, wrt=list(c.linearities.keys()), jac=c.linearities,
                )
            else:
                optprob.addConGroup(
                    c.name, c.size, lower=c.lower, upper=c.upper,
                    linear=False, wrt=list(grad_sparse[c.name].keys()), jac=grad_sparse[c.name],
                )

        sens = jax.jit(lambda inputs, cvals: grad(inputs))

        return optprob, sens

class DeterministicTOP(TrajectoryOptimizationProblem):
    """control_representation ('cartesian'/'regularized') is dispatched once
    at construction time via self._control_repr - the rest of the pipeline
    (variables/objective/evaluate) stays branch-free regardless of which one
    is active, same pattern StochasticTOP already uses for
    gain_param_type/feedback_control_type.

    The control variable is always 'control_arc_hst' whatever the
    representation; evaluate() converts it once, up front, into the derived
    cartesian 'U_arc_hst' that the propagator and everything physical
    downstream consumes."""

    def __init__(self, problem_def: ProblemDefinition):
        propagator = _build_propagator(problem_def, sensitivities=False)
        self._control_repr = _build_control_repr(problem_def)
        constraints_list = [
            BoundaryConstraint(problem_def, "initial"),
            BoundaryConstraint(problem_def, "final"),
            MatchPointConstraint(problem_def),
            ControlNormConstraint(problem_def, self._control_repr),
        ]
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            constraints_list.append(MeshConstraint(problem_def))
        if problem_def.toggles.det_col_avoid and not problem_def.toggles.stat_col_avoid:
            constraints_list.append(DetCollisionAvoidanceConstraint(problem_def))
        super().__init__(problem_def, propagator, constraints_list)

    def variables(self) -> list[OptimizationVariable]:
        dims = self._problem_def.dims
        bc = self._problem_def.boundary_conditions

        variables_list = [
            *self._control_repr.variables(self._problem_def, scale=3e-3),
            OptimizationVariable(
                name='alpha', size=1,
                lower=jnp.asarray(bc.alpha_min), upper=jnp.asarray(bc.alpha_max), value=jnp.asarray(bc.alpha_min),
                guess_fn=lambda key, prior: jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max),
            ),
            OptimizationVariable(
                name='beta', size=1,
                lower=jnp.asarray(bc.beta_min), upper=jnp.asarray(bc.beta_max), value=jnp.asarray(bc.beta_min),
                guess_fn=lambda key, prior: jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max),
            ),
            OptimizationVariable(
                name='X0', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.X0_init, jnp.array([1.0])]),
                # derived from alpha's already-resolved guess, not its own draw
                guess_fn=lambda key, prior: jnp.concatenate([
                    bc.X0_interp.evaluate(prior['alpha']), jnp.array([1.0]),
                ]),
            ),
            OptimizationVariable(
                name='Xf', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.Xf_init, jnp.array([0.95])]),
                guess_fn=lambda key, prior: jnp.concatenate([
                    bc.Xf_interp.evaluate(prior['beta']), jnp.array([0.95]),
                ]),
            ),
        ]

        # Always declared, whatever the mesh type - 'fixed' just pins it to its
        # initial value (the hot start's mesh if there is one, else the
        # scenario's), so nothing downstream has to branch on mesh type to find
        # the arc lengths. Durations rather than cumulative node times: the
        # dynamics are autonomous, so only durations are physical, and the
        # difference operator relating the two costs cond(D)^2 in the Hessian.
        dt_arc = jnp.diff(bc.t_node_bound)
        variables_list.append(OptimizationVariable(
            name='arc_length_hst', size=dims.N_arcs,
            lower=jnp.full(dims.N_arcs, 1e-5), upper=jnp.full(dims.N_arcs, 1e6),
            value=dt_arc, guess_fn=lambda key, prior: dt_arc,
            pinned=self._problem_def.toggles.adaptive_mesh_type.lower() != 'adaptive_fixedtof',
        ))

        return variables_list

    def objective(self, sol_data: dict) -> float:
        dims = self._problem_def.dims

        control_norms = self._control_repr.control_norm_term(self._problem_def, sol_data)

        arc_length_hst = sol_data['arc_length_hst']
        J = control_norms @ arc_length_hst

        if self._problem_def.toggles.adaptive_mesh_type.lower() != 'fixed':
            J = J + 1e-6 * (arc_length_hst @ arc_length_hst)

        return J

    def evaluate(self, inputs: dict) -> dict:
        inputs['U_arc_hst'] = self._control_repr.to_cartesian(
            self._problem_def, inputs['control_arc_hst'],
        )
        sol_data = self.propagate(dict(inputs))

        output = {'o': self.objective(sol_data)}
        for c in self.constraints():
            if c.linearities:
                continue  # linear constraints evaluated by pyoptsparse via their jac, not vals()
            output[c.name] = c.evaluate(self._problem_def, sol_data)

        jax.debug.callback(self.progress, output['o'], *self._constraint_progress(output))

        return output

class StochasticTOP(TrajectoryOptimizationProblem):
    """Adds ControlRepresentation + GainParameterization + ErrorPropagator to
    DeterministicTOP's pipeline. evaluate() stays branch-free regardless of
    which control representation/gain/feedback type is active -
    StochasticTOP.__init__ does the type dispatch once, at construction
    time, and the rest of the pipeline just calls whichever concrete
    instances got built.

    sol_data['Hq_arc_hst'] (this arc's d(U_cart)/d(U_repr) jacobian - identity
    for cartesian) is stashed as its own key right after propagate(), rather
    than folded into B_hst/B_arc_hst in place, so anything else that reads
    the raw cartesian sensitivity (e.g. detailed_trajectory.py's dV
    linearization) is unaffected; GainParameterization.compute_gains() and
    ErrorPropagator.propagate_cov() each apply it locally, right where they
    already consume B.
    """

    _OBJECTIVE_COLUMNS = ('J', 'J_det', 'J_stat')
    _VARIABLE_COLUMNS = ('xi_max',)

    def __init__(self, problem_def: ProblemDefinition):
        propagator = _build_propagator(problem_def, sensitivities=True)
        self._control_repr = _build_control_repr(problem_def)

        gain_param_type = problem_def.toggles.gain_param_type.lower()
        if gain_param_type == 'arc_lqr':
            self._gain_param = LQRArcGains(problem_def)
        elif gain_param_type == 'fulltraj_lqr':
            self._gain_param = LQRFullTrajGains(problem_def)
        else:
            raise ValueError(f"Unknown gain_param_type: {gain_param_type!r}")

        # Uniform __init__(problem_def) across every ErrorPropagator lets this
        # be a single dict lookup with no per-type argument plumbing - each
        # propagator builds whatever it needs (e.g. estimated_state's
        # MeasurementModel) from problem_def internally.
        feedback_control_type = problem_def.toggles.feedback_control_type.lower()
        error_propagator_cls = {
            'true_state': TrueStateCovPropagator,
            'true_state_sqrt': TrueStateSqrtCovPropagator,
            'estimated_state': EstimatedStateCovPropagator,
            'estimated_state_sqrt': EstimatedStateSqrtCovPropagator,
        }.get(feedback_control_type)
        if error_propagator_cls is None:
            raise NotImplementedError(
                f"feedback_control_type={feedback_control_type!r} isn't implemented - "
                f"expected one of 'true_state', 'true_state_sqrt', 'estimated_state', "
                f"'estimated_state_sqrt'."
            )
        self._error_propagator = error_propagator_cls(problem_def)

        constraints_list = [
            BoundaryConstraint(problem_def, "initial"),
            BoundaryConstraint(problem_def, "final"),
            MatchPointConstraint(problem_def),
            ControlNormConstraint(problem_def, self._control_repr, stochastic=True),
            CovarianceConstraint(self._error_propagator),
        ]
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            constraints_list.append(MeshConstraint(problem_def))
        if problem_def.toggles.det_col_avoid and not problem_def.toggles.stat_col_avoid:
            constraints_list.append(DetCollisionAvoidanceConstraint(problem_def))
        super().__init__(problem_def, propagator, constraints_list)

    def variables(self) -> list[OptimizationVariable]:
        # Same base list as DeterministicTOP.variables() - duplicated rather
        # than shared (see DeterministicTOP), plus the gain parameterization's
        # own variable(s) appended.
        dims = self._problem_def.dims
        bc = self._problem_def.boundary_conditions

        variables_list = [
            *self._control_repr.variables(self._problem_def, scale=1e0),
            OptimizationVariable(
                name='alpha', size=1,
                lower=jnp.asarray(bc.alpha_min),
                upper=jnp.asarray(bc.alpha_max),
                value=jnp.asarray(bc.alpha_min),
                guess_fn=lambda key, prior: jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max),
            ),
            OptimizationVariable(
                name='beta', size=1,
                lower=jnp.asarray(bc.beta_min),
                upper=jnp.asarray(bc.beta_max),
                value=jnp.asarray(bc.beta_min),
                guess_fn=lambda key, prior: jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max),
            ),
            OptimizationVariable(
                name='X0', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.X0_init, jnp.array([1.0])]),
                guess_fn=lambda key, prior: jnp.concatenate([
                    bc.X0_interp.evaluate(prior['alpha']), jnp.array([1.0]),
                ]),
            ),
            OptimizationVariable(
                name='Xf', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.Xf_init, jnp.array([0.95])]),
                guess_fn=lambda key, prior: jnp.concatenate([
                    bc.Xf_interp.evaluate(prior['beta']), jnp.array([0.95]),
                ]),
            ),
        ]

        # Always declared, whatever the mesh type - 'fixed' just pins it to its
        # initial value (the hot start's mesh if there is one, else the
        # scenario's), so nothing downstream has to branch on mesh type to find
        # the arc lengths. Durations rather than cumulative node times: the
        # dynamics are autonomous, so only durations are physical, and the
        # difference operator relating the two costs cond(D)^2 in the Hessian.
        dt_arc = jnp.diff(bc.t_node_bound)
        variables_list.append(OptimizationVariable(
            name='arc_length_hst', size=dims.N_arcs,
            lower=jnp.full(dims.N_arcs, 1e-5), upper=jnp.full(dims.N_arcs, 1e6),
            value=dt_arc, guess_fn=lambda key, prior: dt_arc,
            pinned=self._problem_def.toggles.adaptive_mesh_type.lower() != 'adaptive_fixedtof',
        ))

        variables_list += self._gain_param.variables()

        return variables_list

    def objective(self, sol_data: dict) -> float:
        dims = self._problem_def.dims

        arc_length_hst = sol_data['arc_length_hst']

        J = self._control_repr.stochastic_objective(self._problem_def, sol_data, arc_length_hst)

        if self._problem_def.toggles.adaptive_mesh_type.lower() != 'fixed':
            J = J + 1e-10* (arc_length_hst @ arc_length_hst)

        # Stash a deterministic/stochastic split for the progress print
        # (consistent with the flat-sol_data pattern - keys added where
        # computed). For 'cartesian' this is the exact nominal/TCM split;
        # for 'regularized' there's no such split any more (the mean command
        # is baked into the unified quantile term), so J_det is just the
        # plain nominal norm term and J_stat is whatever's left - informative
        # for the print, not a physically separable quantity there. J itself
        # (returned below) is the real objective either way.
        J_det = self._control_repr.control_norm_term(self._problem_def, sol_data) @ arc_length_hst
        sol_data['J_det'] = J_det
        sol_data['J_stat'] = J - J_det

        return J

    def evaluate(self, inputs: dict) -> dict:
        inputs['U_arc_hst'] = self._control_repr.to_cartesian(
            self._problem_def, inputs['control_arc_hst'],
        )
        sol_data = self.propagate(dict(inputs))
        sol_data['Hq_arc_hst'] = self._control_repr.control_jacobian_hst(self._problem_def, sol_data)
        sol_data = self._gain_param.compute_gains(self._problem_def, sol_data)
        sol_data = self._error_propagator.propagate_cov(self._problem_def, sol_data)

        output = {'o': self.objective(sol_data)}
        for c in self.constraints():
            if c.linearities:
                continue  # linear constraints evaluated by pyoptsparse via their jac, not vals()
            output[c.name] = c.evaluate(self._problem_def, sol_data)

        jax.debug.callback(self.progress, output['o'], sol_data['J_det'], sol_data['J_stat'],
                           *self._constraint_progress(output), jnp.max(sol_data['gain_weights']))

        return output