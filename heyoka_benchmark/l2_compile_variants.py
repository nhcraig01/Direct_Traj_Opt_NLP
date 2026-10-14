# L2 (second-order variational) integrator under different compile settings.
#
# Non-compact mode at full LLVM optimization (opt_level=3) did not finish
# compiling the 5288-operation L2 system in 17+ minutes, so this tries cheaper
# compiles: lower opt_level (how hard LLVM optimizes the generated code) and
# parjit (compile in parallel threads). Each configuration runs in its own
# subprocess with a hard time cap, so a runaway compile can't hang the run.
#
# Usage: python heyoka_benchmark/l2_compile_variants.py

import json
import os
import subprocess
import sys

CAP_S = 600  # per-configuration wall-clock cap (compile + all cases)

# (label, compact_mode, opt_level, parjit) - the compact opt_level=3 row is the
# current heyoka_runs.py setting, rerun here as the in-session baseline.
#
# Result (2026-10-08): compact opt3 is the best usable L2 setting. parjit had no
# effect; non-compact opt0 compiles in 1.8 s but runs 4x SLOWER than compact;
# non-compact opt1/opt2 were still compiling after 6+ min and opt3 after 17+ min,
# so they're left out here - add them back only with a generous CAP_S.
CONFIGS = [
    ('compact     opt3',        True,  3, False),
    ('compact     opt3 parjit', True,  3, True),
    ('non-compact opt0 parjit', False, 0, True),
]


def worker(compact, opt_level, parjit):
    """Build one L2 integrator, time every case, score against the reference.
    Prints a single JSON line for the parent to collect."""
    import resource
    import time

    from cases import build_cases, compare, load_run, time_it
    from heyoka_runs import N, build, derivative_map, unpack

    cases = build_cases()
    ref = load_run('reference')

    t0 = time.perf_counter()
    ta = build('L2', cases[0], compact_mode=compact, opt_level=opt_level, parjit=parjit)
    compile_s = time.perf_counter() - t0
    J_map, H_map = derivative_map(ta)
    var_ic = ta.state[N:].copy()

    def run(c):
        ta.time = 0.0
        ta.state[:N] = c['X0']
        ta.state[N:] = var_ic
        ta.pars[:] = c['U']
        return ta.propagate_until(c['dt'])

    per_case = {}
    for c in cases:
        res, _, steady = time_it(run, c)
        out = unpack(ta.state, 'L2', J_map, H_map)
        err = max(compare(out[f], ref[(c['key'], 'L2')][f])[1] for f in ('X_f', 'J', 'H'))
        per_case[c['key']] = {'steady': steady, 'steps': int(res[3]), 'err': err}

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
    print(json.dumps({'compile': compile_s, 'peak_mb': peak_mb, 'cases': per_case}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker(sys.argv[2] == 'True', int(sys.argv[3]), sys.argv[4] == 'True')
        sys.exit()

    here = os.path.abspath(__file__)
    results = {}
    for label, compact, opt_level, parjit in CONFIGS:
        print(f"{label}: running (cap {CAP_S} s)...", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, here, 'worker', str(compact), str(opt_level), str(parjit)],
                capture_output=True, text=True, timeout=CAP_S,
            )
        except subprocess.TimeoutExpired:
            print(f"    exceeded {CAP_S} s - skipped", flush=True)
            continue
        if proc.returncode != 0:
            print(f"    failed:\n{proc.stderr[-2000:]}", flush=True)
            continue
        results[label] = json.loads(proc.stdout.strip().splitlines()[-1])
        r = results[label]
        print(f"    compile {r['compile']:7.1f} s   peak memory {r['peak_mb']:6.0f} MB", flush=True)

    if not results:
        sys.exit("no configuration completed")

    base = results.get('compact     opt3')
    keys = list(next(iter(results.values()))['cases'])
    print(f"\nL2 steady time per case (ms); speedup vs compact opt3 in brackets\n")
    print(f"    {'case':12s}" + "".join(f"{l:>26s}" for l in results))
    for k in keys:
        row = f"    {k:12s}"
        for label, r in results.items():
            s = r['cases'][k]['steady']
            sp = f" [{base['cases'][k]['steady'] / s:4.2f}x]" if base else ""
            row += f"{s * 1e3:18.3f}{sp:>8s}"
        print(row)
    print(f"\n    {'compile (s)':12s}" + "".join(f"{r['compile']:26.1f}" for r in results.values()))
    print(f"    {'max err vs ref':12s}" + "".join(
        f"{max(c['err'] for c in r['cases'].values()):26.1e}" for r in results.values()))
