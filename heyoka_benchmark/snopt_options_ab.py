# SNOPT option comparison on the heyoka integrator.
#
# Rebuilds the stochastic case exactly as dir_traj_opt.py does (same toggles,
# seed and hot start), then solves it once per SNOPT option variant - only the
# option under test changes. Every run's print/summary/history goes to
# heyoka_benchmark/_out/snopt_options/, so Results/ and the root SNOPT_*.out
# files are never touched.

from cases import OUT_DIR

import os
import time

import jax
import numpy as np
from pyoptsparse import History
from pyoptsparse.pySNOPT.pySNOPT import SNOPT

from src.optimization.init_guess import hot_start_init_guess, random_init_guess
from src.optimization.trajectory_optimization_problem import StochasticTOP
from src.problem.problem_definition import Toggles, build_problem_def
from src.utils.io import parse_snopt_stats, yaml_load

SCENARIO = 'Sandbox'
TOGGLES = Toggles(problem_type='stochastic_gauss_zoh', control_representation='cartesian',
                  feedback_control_type='true_state_sqrt', measurements=('range', 'range-rate', 'angles'),
                  gain_param_type='fulltraj_lqr', adaptive_mesh_type='fixed', integrator='heyoka',
                  alpha_rng=(0.0, 0.0), beta_rng=(0.0, 0.0))
HOT_START = f'Results/{SCENARIO}/deterministic/sol.h5'
SEED = 1
TIME_LIMIT = 600  # seconds per run

# dir_traj_opt.py's stochastic options, verbatim
BASE_OPTIONS = {'Major optimality tolerance': 1e-4,
                'Major feasibility tolerance': 1e-6,
                'Minor feasibility tolerance': 1e-6,
                'Major iterations limit': 20000,
                'Partial price': 1,
                'Linesearch tolerance': .99,
                'Function precision': 1e-10,
                'Verify level': -1,
                'Nonderivative linesearch': 1,
                'Major step limit': 1e-2,
                'Elastic weight': 1.e4,
                'Hessian': 'limited memory'}

VARIANTS = {
    'baseline': {},
    'full_memory': {'Hessian': 'full memory'},
    'scale_1': {'Scale option': 1},
    'scale_2': {'Scale option': 2},
    'full_memory+scale_2': {'Hessian': 'full memory', 'Scale option': 2},
}

if __name__ == "__main__":
    out = os.path.join(OUT_DIR, 'snopt_options')
    os.makedirs(out, exist_ok=True)

    top = StochasticTOP(build_problem_def(yaml_load(f'Scenarios/{SCENARIO}.yaml'), TOGGLES))
    init_guess = random_init_guess(top.variables(), jax.random.PRNGKey(SEED))
    init_guess = {**init_guess, **hot_start_init_guess(HOT_START, top.variables())}

    rows = []
    for name, extra in VARIANTS.items():
        opts = {**BASE_OPTIONS, **extra,
                'Print file': os.path.join(out, f'{name}_print.out'),
                'Summary file': os.path.join(out, f'{name}_summary.out')}
        hist_path = os.path.join(out, f'{name}.hst')
        if os.path.exists(hist_path):
            os.remove(hist_path)

        optprob, sens = top.to_pyoptsparse(init_guess)
        print(f"=== {name}: {extra or 'dir_traj_opt.py options'}", flush=True)
        t0 = time.perf_counter()
        sol = SNOPT(options=opts)(optprob, sens=sens, timeLimit=TIME_LIMIT,
                                   storeHistory=hist_path, storeSens=False)
        wall = time.perf_counter() - t0

        stats = parse_snopt_stats(opts['Summary file'])
        h = History(hist_path).getValues(names=['nMajor', 'feasibility', 'optimality', 'o'], major=True)
        feas, opt, obj = (np.asarray(h[k]).ravel() for k in ('feasibility', 'optimality', 'o'))
        ok = feas <= BASE_OPTIONS['Major feasibility tolerance']
        last = slice(-min(100, len(opt)), None)
        rows.append(dict(
            name=name, wall=wall, inform=sol.optInform['value'], text=sol.optInform['text'],
            majors=len(opt), f_star=float(sol.fStar),
            final_feas=feas[-1], final_opt=opt[-1],
            best_feasible_obj=float(obj[ok].min()) if ok.any() else np.nan,
            min_opt_feasible=float(opt[ok].min()) if ok.any() else np.nan,
            median_opt_last100=float(np.median(opt[last])),
            frac_feasible=float(ok.mean()),
            fun_time=stats.get('time_constraint_functions', np.nan),
        ))
        r = rows[-1]
        print(f"    {r['inform']}: {r['text']} | {r['majors']} majors, {wall:.1f} s", flush=True)

    print("\nsummary")
    print(f"  {'variant':22s} {'exit':>5s} {'majors':>7s} {'wall s':>7s} {'final feas':>10s} {'final opt':>10s}"
          f" {'best feas. obj':>14s} {'min opt (feas.)':>15s} {'med opt last100':>15s} {'% feas':>6s}")
    for r in rows:
        print(f"  {r['name']:22s} {r['inform']:5d} {r['majors']:7d} {r['wall']:7.1f} {r['final_feas']:10.1e}"
              f" {r['final_opt']:10.1e} {r['best_feasible_obj']:14.8f} {r['min_opt_feasible']:15.1e}"
              f" {r['median_opt_last100']:15.1e} {100 * r['frac_feasible']:5.0f}%")
