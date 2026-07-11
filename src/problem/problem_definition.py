from dataclasses import dataclass, field, replace

import numpy as np
import jax.numpy as jnp
import diffrax as dfx
from scipy import stats
from astropy.coordinates import spherical_to_cartesian as sph_to_cart
from jax import Array

from src.utils.math_utils import calc_t_elapsed_nd, sig2cov
from src.utils.io import yaml_load, load_family


@dataclass(frozen=True)
class Dimensions:
    """Sizes used throughout TOP/Propagator/Constraint/etc. for shaping arrays."""
    state_dim: int = 7      # [r (3), v (3), m]
    control_dim: int = 3    # thrust unit-direction components
    meas_dim: int = 0       # 0 for true_state, else len(measurements)-dependent
    N_arcs: int = 0
    N_subarcs: int = 0
    N_nodes: int = 0        # N_arcs + 1
    N_trials: int = 0       # Monte Carlo trials (used later)
    N_save: int = 0         # detailed save points per arc
    arc_length_opt: int = 0
    arc_length_det: int = 0
    transfer_length_det: int = 0
    post_insert_length: int = 0
    length: int = 0


@dataclass(frozen=True)
class Integration:
    a_tol: float = 0.0
    r_tol: float = 0.0


@dataclass(frozen=True)
class Spacecraft:
    m0: float = 0.0           # initial mass [kg]
    Isp: float = 0.0
    T_max: float = 0.0        # [N]
    U_Acc_min_nd: float = 0.0  # min nd acceleration at max mass
    ve: float = 0.0


@dataclass(frozen=True)
class BoundaryConditions:
    """Initial/final orbit data and phasing configuration.

    Orbit family data (Orb0/Orbf X_hst/t_hst, X0_interp/Xf_interp) is still
    loaded via Lib.utilities.load_family() for now - porting that to an
    in-repo format is a separate future effort.

    alpha/beta are always optimization variables indexing X0_interp/Xf_interp
    (see FreeBoundaryConstraint in 05_constraints.puml); a "fixed" phasing is
    just alpha_min == alpha_max (and/or beta_min == beta_max).
    """
    X0_init: Array = None  # == X0_interp(0), convenience default
    Xf_init: Array = None  # == Xf_interp(0)

    Orb0_X_hst: Array = None
    Orb0_t_hst: Array = None
    Orbf_X_hst: Array = None
    Orbf_t_hst: Array = None

    X0_interp: object = None  # diffrax CubicInterpolation
    Xf_interp: object = None
    alpha_min: float = None
    alpha_max: float = None
    beta_min: float = None
    beta_max: float = None

    t_node_bound: Array = None  # (N_nodes,)
    tf_T: float = 0.0            # post-insertion safety time, nd
    indx_f: Array = None         # forward arc indices
    indx_b: Array = None         # backward arc indices


@dataclass(frozen=True)
class Uncertainty:
    """Covariance / dispersion / gating terms for the stochastic problem."""
    Phat_0: Array = None         # initial dispersion covariance
    Ptild_0: Array = None        # initial estimation error covariance
    P_XT_targ: Array = None      # target true-state deviation covariance
    P_XT_targ_inv: Array = None
    S_XT_targ: Array = None
    S_XT_targ_inv: Array = None
    G_stoch: Array = None        # process-noise covariance
    gates: Array = None          # [fixed_mag, prop_mag, fixed_point, prop_point]
    mx_tcm_bound: float = 0.0
    mx_dV_bound: float = 0.0


@dataclass(frozen=True)
class Measurement:
    """Raw measurement-model parameters (MeasurementModel itself is built
    separately at build_problem() time - see 02_feedback_control_method.puml)."""
    r_obs: Array = None
    pos_sig: float = 0.0
    range_sig: float = 0.0
    rate_sig: float = 0.0
    angles_sig: float = 0.0


@dataclass(frozen=True)
class CollisionAvoidance:
    """Numeric collision-avoidance parameters. det_col_avoid/stat_col_avoid
    live on ProblemDefinition.toggles, not here - see Toggles."""
    r_obj: Array = None
    d_safe: float = 0.0
    mx_col_bound: float = 0.0
    alpha_UT: float = 0.0
    beta_UT: float = 0.0
    kappa_UT: float = 0.0


@dataclass(frozen=True)
class Toggles:
    """Problem-type switches that drive which classes build_problem() constructs
    and which branches TOP.variables()/objective()/evaluate() take."""
    problem_type: str = "deterministic"       # "deterministic" | "stochastic_gauss_zoh"
    feedback_control_type: str = "true_state"  # "true_state" | "estimated_state"
    measurements: tuple = ()                   # e.g. ("range", "range-rate", "angles")
    gain_param_type: str = "fulltraj_lqr"      # "arc_lqr" | "fulltraj_lqr"
    adaptive_mesh_type: str = "fixed"          # "fixed" | "adaptive_fixedtof"
    alpha_rng: tuple = (0.0, 1.0)              # (min, max) bounds on the alpha phasing variable; (a, a) pins it
    beta_rng: tuple = (0.0, 1.0)               # (min, max) bounds on the beta phasing variable; (b, b) pins it
    det_col_avoid: bool = False
    stat_col_avoid: bool = False


@dataclass(frozen=True)
class ProblemDefinition:
    """Pure config/constants. Built ONCE by build_problem_def(), stored as a
    private attribute on Propagator/GainParameterization/FeedbackControlMethod/
    Constraint subclasses at construction time. Never passed through
    evaluate(inputs) - inputs (the optimization variables dict) is the only
    thing traced through jit/jacfwd.

    Sys is the dict loaded from the external system-parameter file
    (e.g. EMsys.yaml) - kept as-is for now.
    """
    name: str = ""
    Sys: dict = field(default_factory=dict)
    dims: Dimensions = field(default_factory=Dimensions)
    integration: Integration = field(default_factory=Integration)
    spacecraft: Spacecraft = field(default_factory=Spacecraft)
    boundary_conditions: BoundaryConditions = field(default_factory=BoundaryConditions)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    measurement: Measurement = field(default_factory=Measurement)
    collision_avoidance: CollisionAvoidance = field(default_factory=CollisionAvoidance)
    toggles: Toggles = field(default_factory=Toggles)


# Standard gravitational acceleration [km/s^2], used to convert Isp -> exhaust velocity
_G0 = 9.81 / 1000


def build_problem_def(config: dict, toggles: Toggles) -> ProblemDefinition:
    """Build a ProblemDefinition from a loaded config.yaml dict and a seed Toggles.

    `toggles` carries the problem-type switches that are NOT part of config.yaml
    (problem_type, feedback_control_type, measurements, gain_param_type,
    adaptive_mesh_type - today's Problem_Type/Feedback_Control_Type/Measurements/
    Gain_Parametrization_Type/Adaptive_Mesh_Type in dir_traj_opt.py). The
    config-derived switches (det_col_avoid, stat_col_avoid) are folded in
    here to produce the final Toggles stored on the returned
    ProblemDefinition.

    This is a straightforward restructuring of Lib.utilities.process_config() -
    same math/sources, reorganized into ProblemDefinition's groups.
    """
    Sys = yaml_load(config['dynamics']['sys_param_file'])

    # Trajectory sizing
    N_arcs = config['traj_parameters']['control_arcs']
    N_subarcs = config['traj_parameters']['sub_arcs']
    N_nodes = N_arcs + 1
    N_trials = config['traj_parameters']['MC_trials']
    N_save = config['traj_parameters']['save_pts_detailed']
    arc_length_opt = N_subarcs + 1
    arc_length_det = N_subarcs * (N_save - 1) + 1
    transfer_length_det = N_arcs * (arc_length_det - 1) + 1

    indx_f = jnp.array(np.arange(0, N_arcs // 2))
    indx_b = jnp.array(np.flip(np.arange(N_arcs // 2, N_arcs)))

    # Boundary conditions / orbit families
    t0 = config['boundary_conditions']['t0']
    tf = config['boundary_conditions']['tf']
    tf_T = config['boundary_conditions']['tf_T'] * (24 * 3600 / Sys['Ts'])
    t_node_bound = calc_t_elapsed_nd(t0, tf, N_nodes, Sys['Ts'])
    dt_detail = (t_node_bound[1] - t_node_bound[0]) / (arc_length_det - 1)
    post_insert_length = int(np.ceil(tf_T / dt_detail + 1))
    length = transfer_length_det + post_insert_length - 1

    Family0 = load_family(config['boundary_conditions']['initial_orbit']['family_path'])
    Familyf = load_family(config['boundary_conditions']['final_orbit']['family_path'])

    init_orbit_cfg = config['boundary_conditions']['initial_orbit']
    final_orbit_cfg = config['boundary_conditions']['final_orbit']

    Orb0_X_hst = Family0['X_hst'][init_orbit_cfg['Orb_ID'], :, :]
    Orb0_X_hst = jnp.array(np.roll(Orb0_X_hst, shift=-init_orbit_cfg['Start_Idx'], axis=0))
    Orb0_t_hst = jnp.array(Family0['t_hst'][init_orbit_cfg['Orb_ID'], :])

    Orbf_X_hst = Familyf['X_hst'][final_orbit_cfg['Orb_ID'], :, :]
    Orbf_X_hst = jnp.array(np.roll(Orbf_X_hst, shift=-final_orbit_cfg['Start_Idx'], axis=0))
    Orbf_t_hst = jnp.array(Familyf['t_hst'][final_orbit_cfg['Orb_ID'], :])

    # alpha/beta are always optimization variables that index into the
    # orbit-family interpolants below; their bounds come from
    # toggles.alpha_rng/beta_rng - "fixed" phasing is just the degenerate
    # case where alpha_rng = (a, a) (and/or beta_rng = (b, b)) pins the value.
    alpha_min, alpha_max = toggles.alpha_rng
    beta_min, beta_max = toggles.beta_rng

    Orb0_coefs = dfx.backward_hermite_coefficients(Orb0_t_hst / jnp.max(Orb0_t_hst), Orb0_X_hst)
    Orbf_coefs = dfx.backward_hermite_coefficients(Orbf_t_hst / jnp.max(Orbf_t_hst), Orbf_X_hst)
    X0_interp = dfx.CubicInterpolation(Orb0_t_hst / jnp.max(Orb0_t_hst), Orb0_coefs)
    Xf_interp = dfx.CubicInterpolation(Orbf_t_hst / jnp.max(Orbf_t_hst), Orbf_coefs)

    X0_init = Orb0_X_hst[0, :]  # == X0_interp(0), kept as a convenience default
    Xf_init = Orbf_X_hst[0, :]  # == Xf_interp(0)

    boundary_conditions = BoundaryConditions(
        X0_init=X0_init, Xf_init=Xf_init,
        Orb0_X_hst=Orb0_X_hst, Orb0_t_hst=Orb0_t_hst,
        Orbf_X_hst=Orbf_X_hst, Orbf_t_hst=Orbf_t_hst,
        X0_interp=X0_interp, Xf_interp=Xf_interp,
        alpha_min=alpha_min, alpha_max=alpha_max, beta_min=beta_min, beta_max=beta_max,
        t_node_bound=t_node_bound, tf_T=tf_T, indx_f=indx_f, indx_b=indx_b,
    )

    # Spacecraft / propulsion
    m0 = config['engine']['m0']
    Isp = config['engine']['Isp']
    T_max = config['engine']['T_max']
    U_Acc_min_nd = (T_max / 1000) / (Sys['As'] * m0)
    ve = Isp * _G0 / Sys['Vs']

    if U_Acc_min_nd * (t_node_bound[-1] - t_node_bound[0]) / (1 - 1e-2) > ve:
        print("Warning: S/C has insufficient mass to continuously thrust over the transfer.")

    spacecraft = Spacecraft(m0=m0, Isp=Isp, T_max=T_max, U_Acc_min_nd=U_Acc_min_nd, ve=ve)

    # Uncertainty / covariance terms
    disp_cfg = config['uncertainty']['covariance']['initial_dispersion']
    Phat_0 = sig2cov(disp_cfg['pos_sig'], disp_cfg['vel_sig'], disp_cfg['mass_sig'], Sys, m0)

    err_cfg = config['uncertainty']['covariance']['initial_error']
    Ptild_0 = sig2cov(err_cfg['pos_sig'], err_cfg['vel_sig'], err_cfg['mass_sig'], Sys, m0)

    targ_cfg = config['uncertainty']['covariance']['post_insert_target_total']
    P_XT_targ = sig2cov(targ_cfg['pos_sig'], targ_cfg['vel_sig'], targ_cfg['mass_sig'], Sys, m0)

    a_err = config['uncertainty']['acc_sig']
    U_dyn_err = (a_err / Sys['As']) / U_Acc_min_nd
    G_stoch = np.diag(np.array([U_dyn_err, U_dyn_err, U_dyn_err]))

    gates_cfg = config['uncertainty']['gates']
    gates = np.array([gates_cfg['fixed_mag'], gates_cfg['prop_mag'], gates_cfg['fixed_point'], gates_cfg['prop_point']])

    mx_tcm_bound = np.sqrt(stats.chi2.ppf(config['uncertainty']['tcm_stat_bound'], 3))
    mx_dV_bound = np.sqrt(stats.chi2.ppf(config['uncertainty']['dV_bound'], 3))

    uncertainty = Uncertainty(
        Phat_0=Phat_0, Ptild_0=Ptild_0, P_XT_targ=P_XT_targ,
        P_XT_targ_inv=np.linalg.inv(P_XT_targ),
        S_XT_targ=np.linalg.cholesky(P_XT_targ),
        S_XT_targ_inv=np.linalg.inv(np.linalg.cholesky(P_XT_targ)),
        G_stoch=G_stoch, gates=gates,
        mx_tcm_bound=mx_tcm_bound, mx_dV_bound=mx_dV_bound,
    )

    # Measurement model parameters
    meas_cfg = config['uncertainty']['measurement']
    r_obs_x, r_obs_y, r_obs_z = sph_to_cart(*meas_cfg['observer_alt_lat_lon'])
    r_obs_body = np.array([r_obs_x, r_obs_y, r_obs_z])
    r_obs = (Sys['dim'][meas_cfg['observer_body']] + r_obs_body) / Sys['Ls']
    pos_sig = meas_cfg['pos_sig'] / Sys['Ls']
    range_sig = meas_cfg['range_sig'] / Sys['Ls']
    rate_sig = meas_cfg['range_rate_sig'] / (Sys['Ls'] / Sys['Ts'])
    angles_sig = meas_cfg['angles_sig'] / (60 * 60) * (jnp.pi / 180)

    measurement = Measurement(r_obs=r_obs, pos_sig=pos_sig, range_sig=range_sig, rate_sig=rate_sig, angles_sig=angles_sig)

    # Measurement dimension (n_meas) by summing the active measurement sizes -
    # matches what MeasurementModel builds, without the SymPy construction.
    _MEAS_DIMS = {'position': 3, 'pos': 3, 'range': 1, 'rng': 1,
                  'range-rate': 1, 'rng-rate': 1, 'rate': 1, 'angles': 2, 'ang': 2}
    meas_dim = 0
    if toggles.feedback_control_type.lower() == 'estimated_state':
        meas_dim = sum(_MEAS_DIMS[m.lower()] for m in toggles.measurements)

    # Collision avoidance
    col_avoid_cfg = config['constraints']['col_avoid']
    r_obj = np.asarray(Sys[col_avoid_cfg['parameters']['obj_body']])
    d_safe = jnp.array(col_avoid_cfg['parameters']['safe_d']) / Sys['Ls']
    det_col_avoid = col_avoid_cfg['det']['bool']
    stat_col_avoid = col_avoid_cfg['stat']['bool']
    mx_col_bound = np.sqrt(stats.norm.ppf(col_avoid_cfg['stat']['bound']))
    alpha_UT = col_avoid_cfg['stat']['UT']['alpha']
    beta_UT = col_avoid_cfg['stat']['UT']['beta']
    kappa_UT = col_avoid_cfg['stat']['UT']['kappa']

    collision_avoidance = CollisionAvoidance(
        r_obj=r_obj, d_safe=d_safe, mx_col_bound=mx_col_bound,
        alpha_UT=alpha_UT, beta_UT=beta_UT, kappa_UT=kappa_UT,
    )

    dims = Dimensions(
        N_arcs=N_arcs, N_subarcs=N_subarcs, N_nodes=N_nodes, N_trials=N_trials, N_save=N_save,
        meas_dim=meas_dim, arc_length_opt=arc_length_opt, arc_length_det=arc_length_det,
        transfer_length_det=transfer_length_det, post_insert_length=post_insert_length, length=length,
    )

    final_toggles = replace(toggles, det_col_avoid=det_col_avoid, stat_col_avoid=stat_col_avoid)

    return ProblemDefinition(
        name=config['name'],
        Sys=Sys,
        dims=dims,
        integration=Integration(a_tol=config['integration']['a_tol'], r_tol=config['integration']['r_tol']),
        spacecraft=spacecraft,
        boundary_conditions=boundary_conditions,
        uncertainty=uncertainty,
        measurement=measurement,
        collision_avoidance=collision_avoidance,
        toggles=final_toggles,
    )
