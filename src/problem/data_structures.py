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
    # Hold the variable at whatever initial value it ends up with (lower = upper
    # = value), rather than at `lower`/`upper`. Lets a variable be declared
    # unconditionally and simply frozen in the configurations that don't
    # optimize it - the value may come from a hot start, so the freeze point
    # isn't known until to_pyoptsparse() merges the initial guess, which is why
    # this is a flag rather than bounds set here.
    pinned: bool = False
