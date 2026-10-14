"""Terminal output for a run: case banner, setup steps, the per-evaluation
progress table, SNOPT's summary and the saved-files block. Plain ASCII so it
reads the same in a terminal or a log file."""

import re
import time
from contextlib import contextmanager

WIDTH = 74


def _rule():
    print("=" * WIDTH)


def banner(scenario: str, toggles, extra: list[tuple[str, str]]):
    """Scenario, problem type (with feedback type and, for estimated state, the
    measurements) and whatever `extra` (key, value) lines the caller adds."""
    if toggles.problem_type.lower() == 'deterministic':
        problem = 'deterministic'
    else:
        problem = f"stochastic | {toggles.feedback_control_type} feedback"
        if toggles.feedback_control_type.lower().startswith('estimated'):
            problem += f" ({', '.join(toggles.measurements)})"
    _rule()
    for key, value in [('scenario', scenario), ('problem', problem), *extra]:
        print(f"  {key:<10} {value}")
    _rule()


@contextmanager
def step(label: str, timed: bool = False, indent: int = 2):
    """`  label ........ done (1.2 s)` around a block of work."""
    print(f"{' ' * indent}{label} ".ljust(46, '.') + ' ', end='', flush=True)
    t0 = time.perf_counter()
    yield
    print('done' + (f' ({time.perf_counter() - t0:.1f} s)' if timed else ''))


class EvalTable:
    """One row per objective/constraint evaluation, fed by jax.debug.callback
    from TOP.evaluate(). Silent until start() - so evaluations made while the
    SNOPT problem is being built don't print - then a header, the first N_SHOW
    rows, '...', and one row rewritten in place. Widths are fixed, so the live
    row never leaves stray characters behind; a group widens to fit its title."""

    N_SHOW = 10
    _KINDS = {'objective': ('{: .3e}', 10), 'constraint': ('{: .0e}', 6), 'variable': ('{: .1e}', 9)}

    def __init__(self, objective: list[str], equality: list[str], inequality: list[str], variables: list[str]):
        self._groups = []
        for title, labels, kind in (('objective', objective, 'objective'),
                                    ('equality |c|', equality, 'constraint'),
                                    ('inequality <= 0', inequality, 'constraint'),
                                    ('variables', variables, 'variable')):
            if not labels:
                continue
            fmt, w = self._KINDS[kind]
            cols = [[label, fmt, w] for label in labels]
            span = len(cols) * (w + 1) - 1
            if len(title) > span:
                cols[0][2] += len(title) - span
                span = len(title)
            self._groups.append((title, cols, span))
        self._n, self._on, self._live_width = 0, False, 0

    def start(self):
        self._n, self._on, self._live_width = 0, True, 0
        groups = [('', [['eval', None, 6]], 6), *self._groups]
        print()
        print(' | '.join(title.center(span) for title, _, span in groups).rstrip())
        print(' | '.join(' '.join(label.rjust(w) for label, _, w in cols) for _, cols, _ in groups))
        print('-+-'.join('-' * span for *_, span in groups))

    def __call__(self, *values):
        if not self._on:
            return
        self._n += 1
        vals = iter(values)
        row = f"{self._n:>6} | " + ' | '.join(
            ' '.join(fmt.format(float(next(vals))).rjust(w) for _, fmt, w in cols) for _, cols, _ in self._groups)
        if self._n <= self.N_SHOW:
            print(row)
            return
        if self._n == self.N_SHOW + 1:
            print('   ... |')
        self._live_width = max(self._live_width, len(row))
        print('\r' + row.ljust(self._live_width), end='', flush=True)

    def finish(self):
        if self._on and self._n > self.N_SHOW:
            print()
        self._on = False


def _final_major(summary_path: str):
    """(feasibility, optimality) from SNOPT's last major-iteration line, as SNOPT
    prints them - parentheses mean within tolerance - or (None, None)."""
    pattern = re.compile(r'^\s*\d+\s+\d+\s+-?[\d.]+E[-+]\d+\s+\d+\s*(\(?[\d.]+E[-+]\d+\)?)\s*(\(?[\d.]+E[-+]\d+\)?)')
    last = (None, None)
    with open(summary_path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                last = m.groups()
    return last


def snopt_summary(stats: dict, summary_path: str):
    """SNOPT's own end-of-solve numbers, from its summary file (parse_snopt_stats)."""
    pad = ' ' * 12
    print(f"\n  SNOPT     EXIT {stats['exit_code']} / INFO {stats['info_code']} - {stats['info_text']}")
    print(f"{pad}J* = {stats['f_star']:.6e}")
    feas, opt = _final_major(summary_path)
    if feas is not None:
        print(f"{pad}final feasibility {feas}, optimality {opt}")
    print(f"{pad}{stats['major_iterations']} majors | {stats['minor_iterations']} minors | "
          f"{stats['user_func_calls']} function calls")
    total = stats['solve_time_s']
    funcs = stats['constraint_func_time_s'] + stats['objective_func_time_s']
    print(f"{pad}{total:.1f} s = {funcs:.1f} s in functions ({funcs / stats['user_func_calls'] * 1e3:.1f} ms/call)"
          f" + {total - funcs:.1f} s in SNOPT")


def saved(case_dir: str, files: list[str]):
    print()
    _rule()
    print(f"  saved -> {case_dir}")
    print(f"           {', '.join(files)}")
    _rule()
