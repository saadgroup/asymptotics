"""
asymptotics.methods.stepwise
=========================
Step-by-step perturbation expansion for ODE problems.

Usage
-----
>>> sol = eq.begin_expansion(order=2)
>>> sol.show()                         # see all equations, nothing solved yet
>>>
>>> sol[0].solve()                     # try SymPy
>>> sol[0].set_solution(expr)          # or provide manually
>>>
>>> sol[1].solve()                     # SymPy handles linear orders
>>>
>>> sol.solve_all()                    # try all remaining
>>>
>>> sol.expansion                      # available once all solved
>>> sol.show()
>>> sol.to_latex()
>>> sol.eval(eps=0.1, at=t_vals)
>>> sol.compare_numeric(eps=0.1)
"""

from __future__ import annotations
from sympy import (
    Function, Symbol, symbols, series, expand, diff,
    dsolve, solve, Eq, Add, Integer, sympify,
    exp as _exp, cos as _cos
)
from asymptotics.methods.regular_ode import _bc_value_at_order


# ---------------------------------------------------------------------------
# Order entry for step-by-step expansion
# ---------------------------------------------------------------------------

class StepwiseOrderEntry:
    """
    One order in a step-by-step perturbation expansion.

    Attributes
    ----------
    order               : int
    ode                 : Eq  — the ODE at this order (always available)
    ode_substituted     : Eq  — ODE with lower-order solutions substituted
                               (available once all lower orders are solved)
    general_solution    : Expr — with free constants (after solve)
    particular_solution : Expr — constants fixed by conditions (after solve)
    secular             : bool
    is_solved           : bool
    symbol              : Function — u_k(t)
    """

    def __init__(self, order, ode_symbolic, ode_coeffs, symbol, hierarchy):
        self.order               = order
        self._ode_symbolic       = ode_symbolic   # Eq with uk, lower as Functions
        self._ode_coeffs         = ode_coeffs     # raw coefficients (pre-substitution)
        self.symbol              = symbol
        self._hierarchy          = hierarchy       # back-reference

        self.general_solution    = None
        self.particular_solution = None
        self.secular             = False
        self.is_solved           = False

    @property
    def ode(self):
        """
        The ODE at this order.
        - If lower orders not yet solved: shows symbolic form with u_k symbols
        - If lower orders solved: shows both symbolic AND substituted forms
        """
        h = self._hierarchy
        k = self.order

        # Build substituted version if all lower orders are solved
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            return _OdePair(
                symbolic    = self._ode_symbolic,
                substituted = self._build_substituted_ode(h),
                order       = k,
                eps         = h.small_param,
            )
        else:
            return _OdePair(
                symbolic    = self._ode_symbolic,
                substituted = None,
                order       = k,
                eps         = h.small_param,
            )

    def _build_substituted_ode(self, h):
        """Build the ODE with lower-order solutions substituted."""
        ode_expr = self._ode_coeffs
        for j in range(self.order):
            if h.entries[j].is_solved:
                func = h._u_funcs[j]
                sol  = h.entries[j].particular_solution
                ode_expr = ode_expr.subs(func, sol)
        # Evaluate any Derivative objects that became concrete after substitution
        # (e.g. Derivative(4*eta - 4*eta**2, eta, 4) → 0).
        # SymPy's .subs() replaces the function body but leaves the Derivative
        # wrapper unevaluated; .doit() resolves it.
        ode_expr = ode_expr.doit()
        ode_expr = expand(ode_expr)
        ode_expr = expand(expand(ode_expr.rewrite(_exp)).rewrite(_cos))
        return Eq(ode_expr, 0)

    def solve(self):
        """
        Try to solve this order using SymPy's dsolve.

        If successful, applies conditions and stores the particular solution.
        If SymPy fails, prints a clear message with the equation and instructions.

        Returns
        -------
        True if solved, False if SymPy could not solve it.
        """
        h   = self._hierarchy
        k   = self.order
        t   = h.independent
        uk  = self.symbol

        # Check lower orders are solved
        unsolved_below = [j for j in range(k) if not h.entries[j].is_solved]
        if unsolved_below:
            print(
                f"\n  ⚠️  Cannot solve order {k} — lower orders not yet solved: "
                f"{unsolved_below}\n"
                f"  Solve or set solutions for orders {unsolved_below} first.\n"
            )
            return False

        # Build the substituted ODE
        ode_eq = self._build_substituted_ode(h).rewrite(_cos) if k > 0 \
                 else Eq(expand(self._ode_coeffs), 0)

        # Try dsolve
        try:
            gen_sol = dsolve(ode_eq, uk)
            if isinstance(gen_sol, list):
                gen_sol = gen_sol[0]
            gen_expr = gen_sol.rhs
        except Exception as e:
            self._print_solve_failure(ode_eq, e)
            return False

        # Apply conditions
        part_expr = h._apply_conditions(gen_expr, k, t)
        if part_expr is None:
            self._print_solve_failure(ode_eq, "Could not apply conditions")
            return False

        self._store_solution(gen_expr, part_expr, t)
        h._known_solutions[uk] = part_expr
        print(f"  ✓  Order {k} solved: {dep_name(uk)} = {part_expr}")
        # Auto-finalize if all orders are now solved
        if h.n_pending == 0:
            h._finalize()
        return True

    def set_solution(self, expr):
        """
        Provide the solution for this order manually.

        Parameters
        ----------
        expr : SymPy expression or str
            The particular solution u_k(t) (with free constants already fixed).

        Examples
        --------
        >>> sol[0].set_solution(sympify("4*eta*(1 - eta)"))
        >>> sol[0].set_solution("4*eta - 4*eta**2")
        """
        h  = self._hierarchy
        t  = h.independent
        uk = self.symbol

        if isinstance(expr, str):
            expr = sympify(expr)

        # Validate: substitute into conditions to check
        # (just a soft check — user is responsible)
        self._store_solution(expr, expr, t)
        h._known_solutions[uk] = expr
        print(f"  ✓  Order {self.order} set manually: {dep_name(uk)} = {expr}")
        # Auto-finalize if all orders are now solved
        if h.n_pending == 0:
            h._finalize()

    def _store_solution(self, gen_expr, part_expr, t):
        """Store the solution and detect secular terms."""
        from asymptotics.methods.regular_ode import _has_secular_terms
        self.general_solution    = gen_expr
        self.particular_solution = expand(part_expr)
        self.secular             = _has_secular_terms(part_expr, t)
        self.is_solved           = True

    def _print_solve_failure(self, ode_eq, error):
        k = self.order
        eps_str = str(self._hierarchy.small_param)
        sup = ['⁰','¹','²','³','⁴','⁵']
        order_str = f"ε{sup[k]}" if k < len(sup) else f"ε^{k}"
        print(
            f"\n  ✗  Could not solve O({order_str}) equation automatically:\n"
            f"     {ode_eq}\n\n"
            f"  Provide the solution manually:\n"
            f"     sol[{k}].set_solution(your_expr)\n"
            f"  or solve in Mathematica/Maple and paste the result.\n"
        )

    def __repr__(self):
        status = "solved" if self.is_solved else "not solved"
        return f"StepwiseOrderEntry(order={self.order}, {status})"


def dep_name(uk):
    """Extract u_k name from Function."""
    return str(uk.func)


# ---------------------------------------------------------------------------
# ODE pair: symbolic + substituted
# ---------------------------------------------------------------------------

class _OdePair:
    """Holds symbolic and (optionally) substituted forms of an order-k ODE."""

    def __init__(self, symbolic, substituted, order, eps):
        self.symbolic    = symbolic
        self.substituted = substituted
        self.order       = order
        self._eps        = eps

    def __repr__(self):
        from sympy import latex
        eps_sym = self._eps
        sup = ['⁰','¹','²','³','⁴','⁵']
        k   = self.order
        order_str = f"ε{sup[k]}" if k < len(sup) else f"ε^{k}"

        s = f"O({order_str}) — symbolic:\n  {self.symbolic}"
        if self.substituted is not None:
            s += f"\n\nO({order_str}) — substituted:\n  {self.substituted}"
        return s

    def _repr_latex_(self):
        """Rich Jupyter display."""
        try:
            from IPython.display import display, Math, HTML
            from sympy import latex
            eps_sym = self._eps
            k       = self.order
            sup_u   = ['⁰','¹','²','³','⁴','⁵']
            order_str = f"ε{sup_u[k]}" if k < len(sup_u) else f"ε^{{{k}}}"

            def _lx(expr):
                return latex(expr).replace(str(eps_sym), r'\varepsilon')

            eps_label = r'\varepsilon^{' + str(k) + '}' if k > 1 \
                else (r'\varepsilon' if k == 1 else r'\varepsilon^{0}')

            html  = f"<div style='margin:6px 0;font-weight:500;border-left:3px solid #7F77DD;padding-left:8px;'>"
            html += f"O({order_str})</div>"
            display(HTML(html))

            display(Math(
                r'\textbf{Symbolic:} \quad '
                + _lx(self.symbolic.lhs) + ' = 0'
            ))

            if self.substituted is not None:
                display(Math(
                    r'\textbf{Substituted:} \quad '
                    + _lx(self.substituted.lhs) + ' = 0'
                ))
        except Exception:
            pass
        return ''


# ---------------------------------------------------------------------------
# Stepwise hierarchy
# ---------------------------------------------------------------------------

class StepwiseHierarchy:
    """
    A perturbation hierarchy built step by step.

    Created by ODE.begin_expansion(order=N). Equations are set up
    immediately; solutions are obtained one order at a time via
    sol[k].solve() or sol[k].set_solution(expr).

    Once all orders are solved, the full standard API is available:
    sol.expansion, sol.show(), sol.to_latex(), sol.eval(), sol.compare_numeric()
    """

    def __init__(self):
        self.entries         = []     # list of StepwiseOrderEntry
        self.small_param     = None
        self.independent     = None
        self._problem        = None
        self._problem_type   = None
        self._method         = "Regular perturbation (step-by-step)"
        self._u_funcs        = []
        self._known_solutions = {}   # u_k func -> particular solution expr
        self._n_orders       = 0
        self.expansion       = None  # set after finalize

    def __getitem__(self, k: int) -> StepwiseOrderEntry:
        if k < 0 or k >= len(self.entries):
            raise IndexError(
                f"\n\n  Order {k} out of range. Available: 0 to {len(self.entries)-1}\n"
            )
        return self.entries[k]

    def __len__(self):
        return len(self.entries)

    @property
    def n_solved(self):
        return sum(1 for e in self.entries if e.is_solved)

    @property
    def n_pending(self):
        return sum(1 for e in self.entries if not e.is_solved)

    def solve_all(self):
        """
        Try to solve all unsolved orders using SymPy.
        Stops and reports when SymPy cannot solve an order.
        """
        for e in self.entries:
            if not e.is_solved:
                success = e.solve()
                if not success:
                    print(
                        f"\n  Stopped at order {e.order}. "
                        f"Provide solution manually with:\n"
                        f"    sol[{e.order}].set_solution(your_expr)\n"
                        f"  then call sol.solve_all() again.\n"
                    )
                    return
        self._finalize()

    def _check_all_solved(self, method_name):
        if self.n_pending > 0:
            pending = [e.order for e in self.entries if not e.is_solved]
            solved  = [e.order for e in self.entries if e.is_solved]
            lines   = []
            for k in pending:
                lines.append(
                    f"    sol[{k}].solve()   or   sol[{k}].set_solution(expr)"
                )
            raise RuntimeError(
                f"\n\n  Cannot call '{method_name}' — not all orders are solved.\n"
                f"  Solved:  {solved}\n"
                f"  Pending: {pending}\n\n"
                f"  Solve remaining orders:\n" +
                "\n".join(lines) +
                f"\n  Or: sol.solve_all()\n"
            )

    def _finalize(self):
        """Assemble expansion once all orders are solved."""
        eps = self.small_param
        self.expansion = Add(*[
            self.entries[k].particular_solution * eps**k
            for k in range(len(self.entries))
        ])
        # Also set expansion_t for Lindstedt compatibility
        self.expansion_t = self.expansion
        print(f"\n  ✓  All orders solved. expansion is now available.\n")

    def _apply_conditions(self, gen_expr, k, t):
        """Apply conditions to fix integration constants."""
        from sympy import diff as _diff, Eq as _Eq, solve as _solve
        from sympy import nan as _nan, zoo as _zoo
        from sympy import limit as _lim
        from asymptotics.core.conditions import LimitCondition
        from asymptotics.methods.regular_ode import _apply_limit_condition, _has_secular_terms

        conds = self._problem.conditions
        deriv_syms = self._problem._deriv_syms

        free_consts = sorted(
            [s for s in gen_expr.free_symbols
             if str(s).startswith('C') and str(s)[1:].isdigit()],
            key=lambda s: int(str(s)[1:])
        )

        eps = self.small_param
        cond_equations = []
        for cond in conds:
            if isinstance(cond, LimitCondition):
                eq = _apply_limit_condition(
                    cond, gen_expr, t,
                    self._problem._dependent_name,
                    deriv_syms, k, eps=eps
                )
                if eq is not None and eq is not True:
                    cond_equations.append(eq)
            else:
                pt  = cond.point
                val = _bc_value_at_order(cond.value, eps, k)
                if cond.deriv_order == 0:
                    expr_at_pt = gen_expr.subs(t, pt)
                else:
                    expr_at_pt = _diff(gen_expr, t, cond.deriv_order).subs(t, pt)

                from sympy import nan as _nan2, zoo as _zoo2
                if expr_at_pt in (_nan2, _zoo2) or expr_at_pt.has(_nan2, _zoo2):
                    expr_at_pt = _lim(
                        gen_expr if cond.deriv_order == 0
                        else _diff(gen_expr, t, cond.deriv_order),
                        t, pt, '+'
                    )
                cond_equations.append(_Eq(expr_at_pt, val))

        if not free_consts:
            return gen_expr

        try:
            const_sol = _solve(cond_equations, free_consts)
            if isinstance(const_sol, dict):
                return gen_expr.subs(const_sol)
            elif isinstance(const_sol, list) and const_sol:
                if isinstance(const_sol[0], dict):
                    return gen_expr.subs(const_sol[0])
            return gen_expr
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Standard API — available once all orders solved
    # ------------------------------------------------------------------

    def show(self, mode: str = "auto") -> None:
        """
        Display the hierarchy. Shows equations for all orders.
        Shows solutions for solved orders.
        """
        _show_stepwise(self, mode=mode)

    def to_latex(self, environment='align', show_orders=False, filename=None):
        self._check_all_solved('to_latex')
        from asymptotics.latex_export import to_latex as _to_latex
        return _to_latex(self._as_ode_hierarchy(),
                         environment=environment,
                         show_orders=show_orders,
                         filename=filename)

    def eval(self, eps, at=None, params=None):
        self._check_all_solved('eval')
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

    def compare_numeric(self, eps, params=None, **kwargs):
        self._check_all_solved('compare_numeric')
        from asymptotics.numerics import compare_numeric
        return compare_numeric(self, eps, params=params, **kwargs)

    def _as_ode_hierarchy(self):
        """Convert to ODEHierarchy for use with existing display/export."""
        from asymptotics.methods.regular_ode import ODEHierarchy, ODEOrderEntry
        h = ODEHierarchy()
        h.small_param   = self.small_param
        h.independent   = self.independent
        h._method       = self._method
        h._problem_type = self._problem_type
        h._problem      = self._problem
        h.expansion     = self.expansion
        h.expansion_t   = self.expansion
        for e in self.entries:
            entry = ODEOrderEntry(
                order               = e.order,
                ode                 = e._ode_symbolic,
                general_solution    = e.general_solution,
                particular_solution = e.particular_solution,
                constants           = {},
                symbol              = e.symbol,
                secular             = e.secular,
            )
            h.entries.append(entry)
        return h


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _show_stepwise(h, mode="auto"):
    try:
        from IPython.display import display, Math, HTML
        _jupyter = True
    except ImportError:
        _jupyter = False

    if mode == "text" or (mode == "auto" and not _jupyter):
        _show_text(h)
        return

    from IPython.display import display, Math, HTML
    from sympy import latex

    eps = h.small_param
    sup = ['⁰','¹','²','³','⁴','⁵']

    def _lx(expr):
        return latex(expr).replace(str(eps), r'\varepsilon')

    # Title
    n_solved = h.n_solved
    n_total  = len(h.entries)
    status   = f"{n_solved}/{n_total} orders solved"

    display(HTML(
        f"<div style='margin-bottom:6px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:{'#d4edda' if n_solved==n_total else '#fff3cd'};"
        f"padding:2px 8px;border-radius:4px;font-size:0.85em;'>{status}</span>"
        f"</div>"
    ))

    for e in h.entries:
        k = e.order
        order_str = 'ε' + (sup[k] if k < len(sup) else str(k))
        solved_badge = (
            "<span style='color:#28a745;font-size:0.8em;'>✓ solved</span>"
            if e.is_solved else
            "<span style='color:#dc3545;font-size:0.8em;'>✗ pending</span>"
        )

        display(HTML(
            f"<div style='margin-top:10px;font-weight:500;"
            f"border-left:3px solid #7F77DD;padding-left:8px;'>"
            f"Order {order_str} &nbsp; {solved_badge}</div>"
        ))

        # Show symbolic ODE
        display(Math(
            r'\textbf{ODE:} \quad '
            + _lx(e._ode_symbolic.lhs) + r' = 0'
        ))

        # Show substituted ODE if lower orders are solved
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            sub_ode = e._build_substituted_ode(h)
            display(Math(
                r'\textbf{Substituted:} \quad '
                + _lx(sub_ode.lhs) + r' = 0'
            ))

        # Show solution if solved
        if e.is_solved:
            display(Math(
                r'\textbf{Solution:} \quad '
                + _lx(e.symbol) + r' = '
                + _lx(e.particular_solution)
            ))

    # Show expansion if all solved
    if h.n_solved == n_total and h.expansion is not None:
        display(HTML(
            "<div style='margin-top:10px;font-weight:600;'>Expansion:</div>"
        ))
        remainder = r'\mathcal{O}(\varepsilon^{' + str(n_total) + r'})'
        display(Math(
            r'\boxed{u = ' + _lx(h.expansion) + r' + ' + remainder + r'}'
        ))


def _show_text(h):
    sup = ['⁰','¹','²','³','⁴','⁵']
    width = 64
    print("=" * width)
    print(f"  {h._method}  ({h.n_solved}/{len(h.entries)} solved)")
    print("=" * width)

    for e in h.entries:
        k = e.order
        order_str = 'ε' + (sup[k] if k < len(sup) else str(k))
        status = '✓' if e.is_solved else '✗'
        print(f"\n  {status} O({order_str})")
        print(f"    ODE: {e._ode_symbolic}")
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            sub = e._build_substituted_ode(h)
            print(f"    Substituted: {sub}")
        if e.is_solved:
            print(f"    Solution: {e.symbol} = {e.particular_solution}")

    if h.expansion is not None:
        print(f"\n  Expansion: {h.expansion}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Setup — called by ODE.begin_expansion()
# ---------------------------------------------------------------------------

def begin_expansion_ode(problem, order: int) -> StepwiseHierarchy:
    """
    Set up the perturbation hierarchy without solving anything.
    Extracts order-by-order equations symbolically.
    """
    from asymptotics.core.exceptions import NoSmallParameterError

    eps        = problem.small_param
    t          = problem._indep_sym
    N          = order
    dep        = problem._dependent_name
    deriv_syms = problem._deriv_syms
    f          = problem.equation
    ptype      = problem.problem_type

    if eps not in f.free_symbols:
        raise NoSmallParameterError(eps, f)

    # Build u_k(t) functions
    u_funcs = [Function(f"{dep}_{k}")(t) for k in range(N + 1)]

    # Build ansatz
    u_ans = sum(eps**k * u_funcs[k] for k in range(N + 1))

    # Substitute ansatz
    dep_sym = problem.dependent
    f_sub   = f.subs(dep_sym, u_ans)
    for k_deriv, dsym in deriv_syms.items():
        f_sub = f_sub.subs(dsym, diff(u_ans, t, k_deriv))

    # Series expand
    f_series = series(f_sub, eps, 0, N + 1)

    # Extract coefficients
    coeffs = {k: f_series.coeff(eps, k) for k in range(N + 1)}

    # Build symbolic ODEs (with u_k functions as unknowns)
    # These are shown as-is — no substitution yet
    symbolic_odes = {}
    for k in range(N + 1):
        uk = u_funcs[k]
        # The symbolic ODE has u_k as unknown, lower orders as symbols
        # Just use the raw coefficient — it already has u_j(t) functions
        ode_expr = expand(coeffs[k])
        symbolic_odes[k] = Eq(ode_expr, 0)

    # Build hierarchy
    h = StepwiseHierarchy()
    h.small_param     = eps
    h.independent     = t
    h._problem        = problem
    h._problem_type   = ptype
    h._method         = f"Regular perturbation — ODE ({'IVP' if ptype == 'ivp' else 'BVP'})"
    h._u_funcs        = u_funcs
    h._n_orders       = N
    h._problem        = problem

    for k in range(N + 1):
        entry = StepwiseOrderEntry(
            order        = k,
            ode_symbolic = symbolic_odes[k],
            ode_coeffs   = coeffs[k],
            symbol       = u_funcs[k],
            hierarchy    = h,
        )
        h.entries.append(entry)

    return h
