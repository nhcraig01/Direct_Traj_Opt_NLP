"""Initial-guess generation - standalone module, independent of any
TrajectoryOptimizationProblem subclass.

Each OptimizationVariable carries its own random-guess policy as `guess_fn`
(PRNGKey -> Array), defined alongside the variable itself in variables() -
adding a new variable never requires touching this module. Variables with
`guess_fn is None` simply keep variables()'s default `value`. The resulting
override dicts are consumed by to_pyoptsparse(init_guess) via
init_guess.get(v.name, v.value), so hot-starting a stochastic problem from a
deterministic solution (or vice versa) is just a matter of which override
dict gets passed in - variables() never changes.
"""

import h5py
import jax
import jax.numpy as jnp
from jax import Array

from src.problem.data_structures import OptimizationVariable


def random_init_guess(variables: list[OptimizationVariable], key: Array) -> dict[str, Array]:
    """Apply each variable's `guess_fn` (if any) to its own split of `key`."""
    keys = jax.random.split(key, len(variables))

    init_guess = {}
    for v, k in zip(variables, keys):
        if v.guess_fn is not None:
            init_guess[v.name] = v.guess_fn(k)

    return init_guess


def hot_start_init_guess(path: str, variables: list[OptimizationVariable]) -> dict[str, Array]:
    """Load an override dict from a previously saved solution (HDF5, see
    Lib.utilities.save_OptimizerSol).

    Only entries whose name matches a variable in `variables` are kept, so a
    deterministic solution can hot-start a stochastic problem - extra
    variables like gain_weights simply fall back to variables()'s default
    value.
    """
    names = {v.name for v in variables}
    init_guess = {}
    with h5py.File(path, "r") as f:
        for name in f.keys():
            if name in names:
                init_guess[name] = jnp.asarray(f[name][:])
    return init_guess
