from typing import Callable, NamedTuple

from jax import Array


class OptimizationVariable(NamedTuple):
    """Build-time description of one addVarGroup() entry."""
    name: str
    size: int
    lower: Array
    upper: Array
    value: Array  # default/fallback initial guess
    guess_fn: Callable[[Array], Array] | None = None  # PRNGKey -> random initial guess; None = no random policy, use `value`


class TrajectoryState(NamedTuple):
    """Where the spacecraft is, produced by Propagator.propagate()."""
    X_hst: Array        # (N_arcs, N_subarcs+1, 7)
    t_hst: Array
    t_node_bound: Array  # (N_arcs+1,)


class ErrorDynamics(NamedTuple):
    """Linearized perturbation model along the trajectory (stochastic only).

    Produced by SensitivityPropagator.propagate() with A_hst/B_hst/
    A_arc_hst/B_arc_hst populated and H_hst/P_v_hst/K_arc_hst/G_exe_arc_hst
    zero-filled, then progressively filled in via _replace() during
    StochasticTOP.evaluate():
      - controller.augment_error_dynamics() -> H_hst, P_v_hst
        (EstimatedStateFeedback only; no-op for TrueStateFeedback)
      - gain_param.compute_gains() -> K_arc_hst
      - gates2Gexe_vmap() -> G_exe_arc_hst

    G_stoch (process-noise covariance) is not stored here - it is a
    constant from ProblemDefinition, read directly by propagate_cov().
    """
    A_hst: Array          # (N_arcs, N_subarcs, 7, 7)
    B_hst: Array          # (N_arcs, N_subarcs, 7, 3)
    A_arc_hst: Array      # (N_arcs, 7, 7)
    B_arc_hst: Array      # (N_arcs, 7, 3)
    H_hst: Array          # (N_arcs, N_subarcs+1, meas_dim, 7), zero for true_state
    P_v_hst: Array        # (N_arcs, N_subarcs+1, meas_dim, meas_dim), zero for true_state
    K_arc_hst: Array      # (N_arcs, 3, 7), zero until gain_param.compute_gains()
    G_exe_arc_hst: Array  # (N_arcs, 3, 3), zero until gates2Gexe_vmap()


class ErrorState(NamedTuple):
    """Standardized covariance output, identical for every FeedbackControlMethod."""
    P_hst: Array        # (N_arcs, N_subarcs+1, 7, 7), "true" deviation covariance
    P_U_arc_hst: Array  # (N_arcs, 3, 3)
