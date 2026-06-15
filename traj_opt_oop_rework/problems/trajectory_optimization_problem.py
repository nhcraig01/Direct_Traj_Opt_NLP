"""TrajectoryOptimizationProblem - abstract base class, 05_top_overview.puml.

propagate(), constraints(), and to_pyoptsparse() are CONCRETE here -
identical for DeterministicTOP and StochasticTOP. variables(), objective(),
and evaluate() are abstract - overridden by each subclass.
"""

import jax
from pyoptsparse import Optimization

from Lib.utilities import process_sparsity
from traj_opt_oop_rework.constraints.constraint import Constraint
from traj_opt_oop_rework.data_structures import ErrorDynamics, ErrorState, OptimizationVariable, TrajectoryState
from traj_opt_oop_rework.problem_definition import ProblemDefinition
from traj_opt_oop_rework.propagators.propagator import Propagator


class TrajectoryOptimizationProblem:
    def __init__(self, problem_def: ProblemDefinition, propagator: Propagator, constraints_list: list[Constraint]):
        self._problem_def = problem_def
        self._propagator = propagator
        self._constraints_list = constraints_list

    def variables(self) -> list[OptimizationVariable]:
        raise NotImplementedError

    def propagate(self, inputs: dict) -> tuple[TrajectoryState, ErrorDynamics | None]:
        return self._propagator.propagate(inputs, self._problem_def)

    def objective(self, inputs: dict, traj_state: TrajectoryState, error_state: ErrorState | None) -> float:
        raise NotImplementedError

    def constraints(self) -> list[Constraint]:
        return self._constraints_list

    def evaluate(self, inputs: dict) -> dict:
        raise NotImplementedError

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
