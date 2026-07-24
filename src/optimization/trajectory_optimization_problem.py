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
    MatchPointConstraint, MeshConstraint, ControlNormRegConstraint
)
from src.dynamics.equations_of_motion import CR3BPDynamics
from src.dynamics.propagator import Propagator, StatePropagator, StateSensitivityPropagator
from src.problem.data_structures import OptimizationVariable
from src.problem.problem_definition import ProblemDefinition
from src.stochastic.error_propagator import (
    EstimatedStateCovPropagator, TrueStateCovPropagator, TrueStateSqrtCovPropagator,
)
from src.stochastic.gain_parameterization import LQRArcGains, LQRFullTrajGains
from src.utils.io import process_sparsity
from src.utils.math_utils import U_reg_to_U_vmap


class TrajectoryOptimizationProblem:
    def __init__(self, problem_def: ProblemDefinition, propagator: Propagator, constraints_list: list[Constraint]):
        self._problem_def = problem_def
        self._propagator = propagator
        self._constraints_list = constraints_list

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

    # Per-iteration progress print (fires on every function evaluation, same
    # as legacy objective_and_constraints). Built dynamically so it only
    # reports the constraints actually present in `output` - linear ones
    # (skipped in evaluate) and inactive ones (e.g. collision avoidance when
    # off) are simply omitted rather than KeyError'd.
    _PRINT_CONSTRAINTS = (
        ('c_X0', 'X0', True), ('c_Xf', 'Xf', True), ('c_X_mp', 'X_mp', True),
        ('c_P_Xf', 'P_Xf', False), ('c_det_col_avoid', 'Col', False),
    )

    def _constraint_print_terms(self, output: dict) -> tuple[str, list]:
        """(format-suffix, args) for whichever standard nonlinear constraints
        exist in `output`. Equality constraints report max|value|, inequality
        constraints report max(value)."""
        fmt, args = "", []
        for name, label, signed in self._PRINT_CONSTRAINTS:
            if name in output:
                fmt += f", {label}: {{:.0e}}"
                args.append(jnp.max(jnp.abs(output[name])) if signed else jnp.max(output[name]))
        return fmt, args

    def to_pyoptsparse(self, init_guess: dict) -> tuple[Optimization, callable]:
        vals = jax.jit(self.evaluate, backend='cpu')
        optprob = Optimization(type(self).__name__, vals)

        for v in self.variables():
            optprob.addVarGroup(v.name, v.size, "c", value=init_guess.get(v.name, v.value), lower=v.lower, upper=v.upper)

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
    def __init__(self, problem_def: ProblemDefinition):
        propagator = StatePropagator(problem_def, CR3BPDynamics(problem_def))
        constraints_list = [
            BoundaryConstraint(problem_def, "initial"),
            BoundaryConstraint(problem_def, "final"),
            MatchPointConstraint(problem_def),
            ControlNormConstraint(problem_def),
        ]
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            constraints_list.append(MeshConstraint(problem_def))
        if problem_def.toggles.det_col_avoid and not problem_def.toggles.stat_col_avoid:
            constraints_list.append(DetCollisionAvoidanceConstraint(problem_def))
        super().__init__(problem_def, propagator, constraints_list)

    def variables(self) -> list[OptimizationVariable]:
        dims = self._problem_def.dims
        bc = self._problem_def.boundary_conditions

        U_size = dims.control_dim * dims.N_arcs

        variables_list = [
            OptimizationVariable(
                name='U_arc_hst', size=U_size,
                lower=-jnp.ones(U_size), upper=jnp.ones(U_size), value=jnp.zeros(U_size),
                guess_fn=lambda key: jnp.sqrt(1e0) * jax.random.normal(key, shape=(U_size,)),
            ),
            OptimizationVariable(
                name='X0', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.X0_init, jnp.array([1.0])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.X0_interp.evaluate(jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max)),
                    jnp.array([1.0]),
                ]),
            ),
            OptimizationVariable(
                name='Xf', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.Xf_init, jnp.array([0.95])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.Xf_interp.evaluate(jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max)),
                    jnp.array([0.95]),
                ]),
            ),
            OptimizationVariable(
                name='alpha', size=1,
                lower=jnp.asarray(bc.alpha_min), upper=jnp.asarray(bc.alpha_max), value=jnp.asarray(bc.alpha_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max),
            ),
            OptimizationVariable(
                name='beta', size=1,
                lower=jnp.asarray(bc.beta_min), upper=jnp.asarray(bc.beta_max), value=jnp.asarray(bc.beta_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max),
            ),
        ]

        if self._problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            variables_list.append(OptimizationVariable(
                name='t_node_bound', size=dims.N_nodes,
                lower=jnp.zeros(dims.N_nodes), upper=jnp.full(dims.N_nodes, bc.t_node_bound[-1]),
                value=bc.t_node_bound, guess_fn=lambda key: bc.t_node_bound,
            ))

        return variables_list

    def objective(self, sol_data: dict) -> float:
        dims = self._problem_def.dims

        eps = 1e-12
        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        control_norms = jnp.sqrt(U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps)

        t_node_bound = sol_data['t_node_bound']
        arc_length_hst = jnp.diff(t_node_bound)
        J = control_norms @ arc_length_hst

        if self._problem_def.toggles.adaptive_mesh_type.lower() != 'fixed':
            dt_even = (t_node_bound[-1] - t_node_bound[0]) / dims.N_arcs
            rel_dt_err = arc_length_hst / dt_even - 1.0
            J = J + 1e-6 * (rel_dt_err @ rel_dt_err)

        return J

    def evaluate(self, inputs: dict) -> dict:
        sol_data = self.propagate(dict(inputs))

        output = {'o': self.objective(sol_data)}
        for c in self.constraints():
            if c.linearities:
                continue  # linear constraints evaluated by pyoptsparse via their jac, not vals()
            output[c.name] = c.evaluate(self._problem_def, sol_data)

        fmt, args = self._constraint_print_terms(output)
        jax.debug.print("J: {:.3e}" + fmt, output['o'], *args)

        return output

class DeterministicTOPReg(TrajectoryOptimizationProblem):
    def __init__(self, problem_def: ProblemDefinition):
        propagator = StatePropagator(problem_def, CR3BPDynamics(problem_def))
        constraints_list = [
            BoundaryConstraint(problem_def, "initial"),
            BoundaryConstraint(problem_def, "final"),
            MatchPointConstraint(problem_def),
            ControlNormRegConstraint(problem_def),
        ]
        if problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            constraints_list.append(MeshConstraint(problem_def))
        if problem_def.toggles.det_col_avoid and not problem_def.toggles.stat_col_avoid:
            constraints_list.append(DetCollisionAvoidanceConstraint(problem_def))
        super().__init__(problem_def, propagator, constraints_list)

    def variables(self) -> list[OptimizationVariable]:
        dims = self._problem_def.dims
        bc = self._problem_def.boundary_conditions

        U_size = dims.control_dim * dims.N_arcs

        variables_list = [
            OptimizationVariable(
                name='U_reg_arc_hst', size=U_size,
                lower=-jnp.ones(U_size), upper=jnp.ones(U_size), value=jnp.zeros(U_size),
                guess_fn=lambda key: jnp.sqrt(1e0) * jax.random.normal(key, shape=(U_size,)),
            ),
            OptimizationVariable(
                name='X0', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.X0_init, jnp.array([1.0])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.X0_interp.evaluate(jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max)),
                    jnp.array([1.0]),
                ]),
            ),
            OptimizationVariable(
                name='Xf', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.Xf_init, jnp.array([0.95])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.Xf_interp.evaluate(jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max)),
                    jnp.array([0.95]),
                ]),
            ),
            OptimizationVariable(
                name='alpha', size=1,
                lower=jnp.asarray(bc.alpha_min), upper=jnp.asarray(bc.alpha_max), value=jnp.asarray(bc.alpha_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max),
            ),
            OptimizationVariable(
                name='beta', size=1,
                lower=jnp.asarray(bc.beta_min), upper=jnp.asarray(bc.beta_max), value=jnp.asarray(bc.beta_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max),
            ),
        ]

        if self._problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            variables_list.append(OptimizationVariable(
                name='t_node_bound', size=dims.N_nodes,
                lower=jnp.zeros(dims.N_nodes), upper=jnp.full(dims.N_nodes, bc.t_node_bound[-1]),
                value=bc.t_node_bound, guess_fn=lambda key: bc.t_node_bound,
            ))

        return variables_list

    def objective(self, sol_data: dict) -> float:
        dims = self._problem_def.dims

        U_reg_arc_hst = sol_data['U_reg_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        control_reg_dot = jnp.sum(U_reg_arc_hst ** 2, axis=1)

        t_node_bound = sol_data['t_node_bound']
        arc_length_hst = jnp.diff(t_node_bound)
        J = control_reg_dot @ arc_length_hst

        if self._problem_def.toggles.adaptive_mesh_type.lower() != 'fixed':
            dt_even = (t_node_bound[-1] - t_node_bound[0]) / dims.N_arcs
            rel_dt_err = arc_length_hst / dt_even - 1.0
            J = J + 1e-6 * (rel_dt_err @ rel_dt_err)

        return J

    def evaluate(self, inputs: dict) -> dict:
        dims = self._problem_def.dims
        inputs['U_arc_hst'] = U_reg_to_U_vmap(inputs['U_reg_arc_hst'].reshape(dims.N_arcs, dims.control_dim)).flatten()
        sol_data = self.propagate(dict(inputs))

        output = {'o': self.objective(sol_data)}
        for c in self.constraints():
            if c.linearities:
                continue  # linear constraints evaluated by pyoptsparse via their jac, not vals()
            output[c.name] = c.evaluate(self._problem_def, sol_data)

        fmt, args = self._constraint_print_terms(output)
        jax.debug.print("J: {:.3e}" + fmt, output['o'], *args)

        return output


class StochasticTOP(TrajectoryOptimizationProblem):
    """Adds GainParameterization + ErrorPropagator to DeterministicTOP's
    pipeline. evaluate() stays branch-free regardless of which gain/feedback
    type is active - StochasticTOP.__init__ does the type dispatch once, at
    construction time, and the rest of the pipeline just calls whichever
    concrete instances got built.
    """

    def __init__(self, problem_def: ProblemDefinition):
        propagator = StateSensitivityPropagator(problem_def, CR3BPDynamics(problem_def))

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
        }.get(feedback_control_type)
        if error_propagator_cls is None:
            raise NotImplementedError(
                f"feedback_control_type={feedback_control_type!r} isn't implemented - "
                f"expected one of 'true_state', 'true_state_sqrt', 'estimated_state'."
            )
        self._error_propagator = error_propagator_cls(problem_def)

        constraints_list = [
            BoundaryConstraint(problem_def, "initial"),
            BoundaryConstraint(problem_def, "final"),
            MatchPointConstraint(problem_def),
            ControlNormConstraint(problem_def, self._error_propagator),
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

        U_size = dims.control_dim * dims.N_arcs

        variables_list = [
            OptimizationVariable(
                name='U_arc_hst', size=U_size,
                lower=-jnp.ones(U_size), upper=jnp.ones(U_size), value=jnp.zeros(U_size),
                guess_fn=lambda key: jnp.sqrt(1e0) * jax.random.normal(key, shape=(U_size,)),
            ),
            OptimizationVariable(
                name='X0', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.X0_init, jnp.array([1.0])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.X0_interp.evaluate(jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max)),
                    jnp.array([1.0]),
                ]),
            ),
            OptimizationVariable(
                name='Xf', size=dims.state_dim,
                lower=jnp.array([-10., -10., -10., -10., -10., -10., 1e-1]),
                upper=jnp.array([10., 10., 10., 10., 10., 10., 1.]),
                value=jnp.concatenate([bc.Xf_init, jnp.array([0.95])]),
                guess_fn=lambda key: jnp.concatenate([
                    bc.Xf_interp.evaluate(jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max)),
                    jnp.array([0.95]),
                ]),
            ),
            OptimizationVariable(
                name='alpha', size=1,
                lower=jnp.asarray(bc.alpha_min), 
                upper=jnp.asarray(bc.alpha_max), 
                value=jnp.asarray(bc.alpha_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.alpha_min, maxval=bc.alpha_max),
            ),
            OptimizationVariable(
                name='beta', size=1,
                lower=jnp.asarray(bc.beta_min), 
                upper=jnp.asarray(bc.beta_max), 
                value=jnp.asarray(bc.beta_min),
                guess_fn=lambda key: jax.random.uniform(key, minval=bc.beta_min, maxval=bc.beta_max),
            ),
        ]

        if self._problem_def.toggles.adaptive_mesh_type.lower() == 'adaptive_fixedtof':
            variables_list.append(OptimizationVariable(
                name='t_node_bound', size=dims.N_nodes,
                lower=jnp.zeros(dims.N_nodes), upper=jnp.full(dims.N_nodes, bc.t_node_bound[-1]),
                value=bc.t_node_bound, guess_fn=lambda key: bc.t_node_bound,
            ))

        variables_list += self._gain_param.variables()

        return variables_list

    def objective(self, sol_data: dict) -> float:
        dims = self._problem_def.dims

        eps = 1e-12
        U_arc_hst = sol_data['U_arc_hst'].reshape(dims.N_arcs, dims.control_dim)
        control_norms = jnp.sqrt(U_arc_hst[:, 0] ** 2 + U_arc_hst[:, 1] ** 2 + U_arc_hst[:, 2] ** 2 + eps)

        t_node_bound = sol_data['t_node_bound']
        arc_length_hst = jnp.diff(t_node_bound)
        J = control_norms @ arc_length_hst

        if self._problem_def.toggles.adaptive_mesh_type.lower() != 'fixed':
            dt_even = (t_node_bound[-1] - t_node_bound[0]) / dims.N_arcs
            rel_dt_err = arc_length_hst / dt_even - 1.0
            J = J + 1e-6 * (rel_dt_err @ rel_dt_err)

        control_max_eig = self._error_propagator.stochastic_control_term(self._problem_def, sol_data)
        J_stat = control_max_eig @ arc_length_hst

        # Stash the deterministic/stochastic split for the progress print
        # (consistent with the flat-sol_data pattern - keys added where
        # computed). J itself is returned as the total objective.
        sol_data['J_det'] = J
        sol_data['J_stat'] = J_stat

        return J + J_stat

    def evaluate(self, inputs: dict) -> dict:
        sol_data = self.propagate(dict(inputs))
        sol_data = self._gain_param.compute_gains(self._problem_def, sol_data)
        sol_data = self._error_propagator.propagate_cov(self._problem_def, sol_data)

        output = {'o': self.objective(sol_data)}
        for c in self.constraints():
            if c.linearities:
                continue  # linear constraints evaluated by pyoptsparse via their jac, not vals()
            output[c.name] = c.evaluate(self._problem_def, sol_data)

        fmt, args = self._constraint_print_terms(output)
        jax.debug.print(
            "J_d: {:.3e}, J_s: {:.3e}" + fmt + ", max_xi: {:.1e}",
            sol_data['J_det'], sol_data['J_stat'], *args, jnp.max(sol_data['gain_weights']),
        )

        return output
