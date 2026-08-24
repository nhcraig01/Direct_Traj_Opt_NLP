from typing import Callable, NamedTuple

from jax import Array


class OptimizationVariable(NamedTuple):
    """Build-time description of one addVarGroup() entry."""
    name: str
    size: int
    lower: Array
    upper: Array
    value: Array  # default/fallback initial guess
    guess_fn: Callable[[Array, dict], Array] | None = None  # (PRNGKey, prior guesses) -> random initial guess; None = use `value`
