# Side-by-side report: diffrax vs heyoka at the benchmark tolerance.
#
# Reads _out/diffrax.pkl, _out/heyoka.pkl and _out/reference.pkl (run
# diffrax_runs.py, heyoka_runs.py and reference.py first). Errors are normwise
# relative against the multiprecision reference - see cases.compare().

from cases import TOL, build_cases, compare, load_run

import numpy as np

FIELDS = {'L0': ('X_f',), 'L1': ('X_f', 'J'), 'L2': ('X_f', 'J', 'H')}

if __name__ == "__main__":
    cases = build_cases()
    dfx, hy, ref = load_run('diffrax'), load_run('heyoka'), load_run('reference')

    print(f"diffrax (Dopri8) vs heyoka (Taylor), both at tol {TOL:g}")
    print("errors: normwise relative vs 113-bit reference, shown as diffrax / heyoka\n")

    tally = {}   # field -> list of (diffrax err, heyoka err)
    for level, fields in FIELDS.items():
        print(f"{level}")
        print(f"    {'case':12s} {'diffrax':>9s} {'heyoka':>9s} {'speedup':>8s} {'steps':>10s}"
              + "".join(f"{f + ' err':>22s}" for f in fields))
        for c in cases:
            d, h, r = (run[(c['key'], level)] for run in (dfx, hy, ref))
            row = (f"    {c['key']:12s} {d['steady'] * 1e3:6.3f} ms {h['steady'] * 1e3:6.3f} ms"
                   f" {d['steady'] / h['steady']:7.1f}x {d['steps']:5d}/{h['steps']:<4d}")
            for f in fields:
                de, he = compare(d[f], r[f])[1], compare(h[f], r[f])[1]
                tally.setdefault(f, []).append((de, he))
                row += f"   {de:8.1e} / {he:8.1e}"
            print(row)
        print()

    print("accuracy summary, all cases and levels")
    for f, pairs in tally.items():
        de, he = np.array(pairs).T
        print(f"    {f:4s} heyoka more accurate in {np.sum(he < de):2d}/{len(pairs)} comparisons;"
              f"  median error  diffrax {np.median(de):.1e}  heyoka {np.median(he):.1e}"
              f"  (median ratio dfx/hy {np.median(de / he):.1f}x)")
