"""Quick-look Matplotlib plotting library for saved trajectory solutions.

Heavy-lifting counterpart to plot_traj.py (the config-at-top entry script) - the
split mirrors dir_traj_opt.py + src/. Reads the data.h5 that dir_traj_opt.py
writes (legacy save_sol schema) plus the sidecar Sys.mat, and produces up to two
figures:

  1. Trajectory + control - 3D transfer (coast/thrust colored, Moon + L1/L2,
     departure/arrival orbits) alongside the control profile (|U|, azimuth,
     elevation). Always produced.
  2. True-state deviation history - per-state |error| vs time with the
     time-varying 3-sigma predicted envelope and a flat target-covariance line.
     Stochastic runs only; both true_state and estimated_state feedback are
     shown on the true-state deviation, so one function covers both.

This is intentionally a "quick look" companion so results are viewable without
MATLAB; the polished distribution-matrix / animation figures stay on the MATLAB
side. Standard deps only (numpy, h5py, scipy, matplotlib); labels use Matplotlib
mathtext, so no LaTeX installation is required.
"""

import os

import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.utils.io import load_results

# Mean equatorial radii [km]. Sys.mat's 'dim' struct carries primary positions/
# masses but not body radii, so the Moon radius is a constant here.
MOON_R_KM = 1737.4

# Per-state error labels (mathtext) + axis-unit suffix for the 6 kinematic states
# (mass is not shown in the deviation history).
_ERR_LABELS = [r'$\delta x$', r'$\delta y$', r'$\delta z$',
               r'$\delta \dot{x}$', r'$\delta \dot{y}$', r'$\delta \dot{z}$']
_ERR_UNITS = [' [km]', ' [km]', ' [km]', ' [km/s]', ' [km/s]', ' [km/s]']


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

class TrajData:
    """Attribute view over the grouped data.h5 (via io.load_results). Stochastic-
    only fields (/det/P, /mc/*, ...) are None for a deterministic case."""

    def __init__(self, results):
        meta, det, mc, orb = results['meta'], results['det'], results['mc'], results['orbits']
        gm = lambda d, k: d[k] if k in d else None

        self.name = str(meta.get('name', ''))
        self.feedback_type = str(meta.get('feedback_type', ''))
        self.stochastic = bool(int(meta.get('stochastic', 0)))

        # Detailed trajectory / control / orbits (always present)
        self.det_X = det['X']            # (L, 7)
        self.det_U = det['U']            # (L, 3) cartesian control
        self.det_U_sph = det['U_sph']    # (L, 3): |U|, azimuth[deg], elevation[deg]
        self.det_t = det['t']            # (L,)
        self.det_X_node = det['X_node']  # (N_arcs+1, 7) arc-boundary states
        self.det_U_arc = det['U_arc']    # (N_arcs, 3) per-arc commanded control
        self.dV_mean = float(det['dV_mean'])
        self.orb0_X = orb['orb0_X']
        self.orbf_X = orb['orbf_X']

        # Stochastic-only fields
        self.det_P = gm(det, 'P')            # (L, 7, 7)
        self.det_P_targ = gm(det, 'P_targ')  # (post-insert, 7, 7)
        self.U_bound = gm(det, 'U_bound')    # (L,)
        self.dV_bound = float(det['dV_bound']) if 'dV_bound' in det else None
        self.mc_X = gm(mc, 'X')              # (N, L, 7)
        self.mc_U_sph = gm(mc, 'U_sph')      # (N, L, 3)


class SysParams:
    """Attribute view over Sys.mat (scales + geometry needed for the plots)."""

    def __init__(self, sys_path):
        S = loadmat(sys_path, squeeze_me=True, struct_as_record=False)
        self.Ls = float(S['Ls'])                 # length scale [km]
        self.Ts = float(S['Ts'])                 # time scale [s]
        self.Vs = float(S['Vs'])                 # velocity scale [km/s]
        self.LagrPts = np.asarray(S['LagrPts'])  # (5, 3) nondimensional
        self.moon_pos = np.asarray(S['dim'].r2)  # secondary position [km]


def load_case(case_dir):
    """Load (TrajData, SysParams) from a case directory holding data.h5 + Sys.mat."""
    return (TrajData(load_results(case_dir)),
            SysParams(os.path.join(case_dir, 'Sys.mat')))


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

def _setup_style(use_tex=False):
    """Give axes a LaTeX-like look. By default uses Matplotlib's built-in
    mathtext with the Computer Modern font set (no LaTeX install needed, so it
    stays portable for the public release). use_tex=True switches to a real
    LaTeX backend for users who have a TeX distribution installed."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.usetex'] = bool(use_tex)
    if not use_tex:
        plt.rcParams['mathtext.fontset'] = 'cm'


def theme_colors(theme):
    if theme == 'dark':
        return dict(bg='black', axis='white', text='white', grid=(0.35, 0.35, 0.35),
                    orb0=(0.9, 0.9, 0.9), orbf=(0.2, 0.9, 0.2),
                    det=(1.0, 1.0, 1.0), mc=(0.9, 0.9, 0.1),
                    thrust=(0.95, 0.25, 0.15), coast=(0.35, 0.55, 0.95),
                    bound=(0.3, 0.75, 0.3), targ=(0.55, 0.85, 0.55), moon=(0.6, 0.6, 0.62))
    return dict(bg='white', axis='black', text='black', grid=(0.8, 0.8, 0.8),
                orb0=(0.1, 0.1, 0.1), orbf=(0.15, 0.7, 0.15),
                det=(0.0, 0.0, 0.0), mc=(0.2, 0.3, 0.9),
                thrust=(0.9, 0.1, 0.1), coast=(0.1, 0.1, 0.9),
                bound=(0.2, 0.6, 0.2), targ=(0.3, 0.7, 0.3), moon=(0.5, 0.5, 0.52))


def _style_axes(ax, c):
    ax.set_facecolor(c['bg'])
    ax.tick_params(colors=c['axis'], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(c['axis'])
    ax.grid(True, color=c['grid'], linewidth=0.5, alpha=0.7)


def _legend(ax, c, **kw):
    leg = ax.legend(facecolor=c['bg'], edgecolor=c['axis'], **kw)
    for txt in leg.get_texts():
        txt.set_color(c['text'])
    return leg


def _mc_alpha(n_trials):
    # Fainter as the trial count grows (matches the MATLAB heuristic).
    return float(np.clip(0.3 / (np.log10(max(n_trials, 1)) + 0.3), 0.02, 0.5))


# --------------------------------------------------------------------------- #
# Figure 1: trajectory + control
# --------------------------------------------------------------------------- #

def plot_trajectory_control(data, sys, c):
    """3D transfer (left) + control profile (right) in one figure."""
    fig = plt.figure(figsize=(15, 8), facecolor=c['bg'])
    gs = GridSpec(3, 2, width_ratios=[1.5, 1], wspace=0.18, hspace=0.12, figure=fig)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    _draw_trajectory(ax3d, data, sys, c)

    ax_mag = fig.add_subplot(gs[0, 1])
    ax_az = fig.add_subplot(gs[1, 1], sharex=ax_mag)
    ax_el = fig.add_subplot(gs[2, 1], sharex=ax_mag)
    _draw_control(ax_mag, ax_az, ax_el, data, sys, c)

    kind = 'Stochastic' if data.stochastic else 'Deterministic'
    dv = f'$\\Delta V_\\mathrm{{mean}}$ = {data.dV_mean:.3f}'
    if data.stochastic and data.dV_bound is not None:
        dv += f', $\\Delta V_{{99}}$ = {data.dV_bound:.3f}'
    dv += ' [km/s]'
    fig.suptitle(f'{data.name}, {kind}\n{dv}', color=c['text'], fontsize=15)
    return fig


def _draw_trajectory(ax, data, sys, c):
    ax.set_facecolor(c['bg'])
    Ls = sys.Ls
    Xt = data.det_X[:, :3] * Ls

    # Transfer colored by thrust vs coast (|U| threshold mirrors DetTraj2ClrDat).
    seg = np.stack([Xt[:-1], Xt[1:]], axis=1)
    thrust = np.linalg.norm(data.det_U[:-1, :], axis=1) > 1e-2
    seg_colors = np.where(thrust[:, None], c['thrust'], c['coast'])
    ax.add_collection3d(Line3DCollection(seg, colors=seg_colors, linewidths=1.6))

    O0 = data.orb0_X[:, :3] * Ls
    Of = data.orbf_X[:, :3] * Ls
    ax.plot(O0[:, 0], O0[:, 1], O0[:, 2], color=c['orb0'], lw=1.1, alpha=0.7, label=r'$\mathrm{Orb}_0$')
    ax.plot(Of[:, 0], Of[:, 1], Of[:, 2], color=c['orbf'], lw=1.1, alpha=0.7, label=r'$\mathrm{Orb}_f$')

    _plot_sphere(ax, sys.moon_pos, MOON_R_KM, c['moon'])

    for idx in (0, 1):  # L1, L2
        P = sys.LagrPts[idx] * Ls
        ax.scatter(P[0], P[1], P[2], marker='^', s=28, color=c['text'])
        ax.text(P[0], P[1], P[2], f'  $L_{idx + 1}$', color=c['text'], fontsize=11)

    # Equal aspect; limits from trajectory + orbits (not the far-off Earth).
    allpos = np.vstack([Xt, O0, Of])
    center = 0.5 * (allpos.max(0) + allpos.min(0))
    half = 1.1 * float(np.abs(allpos - center).max())

    # Burn feathers: shaft-only quiver pointing opposite the thrust (feather
    # length proportional to |U|), one per arc where the control is on.
    if data.det_X_node is not None and data.det_U_arc is not None:
        Xn = data.det_X_node[:-1, :3] * Ls        # arc-start positions [km]
        Ua = data.det_U_arc                       # (N_arcs, 3)
        on = np.linalg.norm(Ua, axis=1) > 1e-2
        if on.any():
            feat = -Ua[on] * (0.12 * half)
            ax.quiver(Xn[on, 0], Xn[on, 1], Xn[on, 2],
                      feat[:, 0], feat[:, 1], feat[:, 2],
                      color=c['thrust'], linewidth=1.1,
                      arrow_length_ratio=0.0, normalize=False)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel(r'$\hat{X}$ [km]', color=c['text'], fontsize=11)
    ax.set_ylabel(r'$\hat{Y}$ [km]', color=c['text'], fontsize=11)
    ax.set_zlabel(r'$\hat{Z}$ [km]', color=c['text'], fontsize=11)
    ax.tick_params(colors=c['axis'], labelsize=8)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(c['bg'])
        pane.pane.set_edgecolor(c['grid'])
    _legend(ax, c, loc='upper right', fontsize=9)


def _plot_sphere(ax, center, radius, color, n=24):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.9, linewidth=0, shade=True)


def _draw_control(ax_mag, ax_az, ax_el, data, sys, c):
    t = data.det_t * sys.Ts / 86400.0  # days
    n_mc = data.mc_U_sph.shape[0] if (data.stochastic and data.mc_U_sph is not None) else 0
    a_mc = _mc_alpha(n_mc)

    for row, (ax, ylab, ylim, yticks) in enumerate([
        (ax_mag, r'$\|U\|$ [ND]', (-0.05, 1.05), np.linspace(0, 1, 6)),
        (ax_az, r'$U_\theta$ [deg]', (-185, 185), np.linspace(-180, 180, 5)),
        (ax_el, r'$U_\phi$ [deg]', (-95, 95), np.linspace(-90, 90, 5)),
    ]):
        _style_axes(ax, c)
        if n_mc:
            ax.plot(t, data.mc_U_sph[:, :, row].T, color=c['mc'], lw=0.4, alpha=a_mc)
        if row == 0 and n_mc and data.U_bound is not None:
            ax.plot(t, data.U_bound, '--', color=c['det'], lw=1.1, label='99.9%')
        ax.plot(t, data.det_U_sph[:, row], '-', color=c['det'], lw=1.3, label='Det')
        ax.set_ylabel(ylab, color=c['text'], fontsize=13)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xlim(t[0], t[-1])

    _legend(ax_mag, c, loc='best', fontsize=9)
    plt.setp(ax_mag.get_xticklabels(), visible=False)
    plt.setp(ax_az.get_xticklabels(), visible=False)
    ax_el.set_xlabel('Time [days]', color=c['text'], fontsize=12)


# --------------------------------------------------------------------------- #
# Figure 2: true-state deviation history (stochastic only)
# --------------------------------------------------------------------------- #

def plot_deviation_history(data, sys, c, sigma=3):
    """Per-state |MC deviation| vs time with the 3-sigma predicted envelope and a
    flat target-covariance line. Plotted on the true-state deviation for both
    feedback types."""
    t = data.det_t * sys.Ts / 86400.0
    nd = np.array([sys.Ls, sys.Ls, sys.Ls, sys.Vs, sys.Vs, sys.Vs])

    fig, axes = plt.subplots(6, 1, figsize=(9, 11), sharex=True, facecolor=c['bg'])
    n_mc = data.mc_X.shape[0]
    a_mc = _mc_alpha(n_mc)

    for i in range(6):
        ax = axes[i]
        _style_axes(ax, c)

        # |deviation| of each MC trial about the deterministic reference.
        err = np.abs(data.mc_X[:, :, i] - data.det_X[None, :, i]) * nd[i]  # (N, L)
        ax.plot(t, err.T, color=c['mc'], lw=0.4, alpha=a_mc)

        # Time-varying sigma-sigma predicted envelope from the deterministic cov
        # (black / theme-foreground line per request).
        bound = np.sqrt(np.clip(data.det_P[:, i, i], 0, None)) * sigma * nd[i]
        ax.plot(t, bound, '--', color=c['det'], lw=1.4,
                label=(f'${sigma}\\sigma$ pred.' if i == 0 else None))

        # Flat target-covariance line (final post-insertion target).
        if data.det_P_targ is not None:
            targ = np.sqrt(max(data.det_P_targ[-1, i, i], 0.0)) * sigma * nd[i]
            ax.axhline(targ, ls='-.', color=c['targ'], lw=1.4,
                       label=('target' if i == 0 else None))

        ax.set_ylabel(_ERR_LABELS[i] + _ERR_UNITS[i], color=c['text'], fontsize=12)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Time [days]', color=c['text'], fontsize=12)
    _legend(axes[0], c, loc='upper left', fontsize=9)
    fig.suptitle(f'{data.name}\nTrue-State Deviation History ({n_mc} MC trials)',
                 color=c['text'], fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def plot_case(case_dir, theme='light', save=True, show=False, use_tex=False):
    """Make the quick-look figures for a case directory (data.h5 + Sys.mat).

    Deterministic -> figure 1 only. Stochastic -> figures 1 and 2. Returns the
    dict of figures; saves them as PNGs next to data.h5 when save=True.
    use_tex=True renders labels with a real LaTeX backend (requires a TeX
    install); the default is portable Computer-Modern mathtext.
    """
    _setup_style(use_tex)
    data, sys = load_case(case_dir)
    c = theme_colors(theme)

    figs = {'Traj_Ctrl': plot_trajectory_control(data, sys, c)}
    if data.stochastic:
        figs['Deviation_Hist'] = plot_deviation_history(data, sys, c)

    if save:
        for name, fig in figs.items():
            out = os.path.join(case_dir, f'{name}.png')
            fig.savefig(out, dpi=200, facecolor=c['bg'], bbox_inches='tight')
            print(f'saved {out}')
    if show:
        plt.show()
    return figs
