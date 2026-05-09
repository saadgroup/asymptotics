"""
asymptotics.methods.regular_ode
============================
Regular perturbation expansion for ODEs F(u, u', u'', t, ε) = 0.

Algorithm
---------
1. Build ansatz:  u(t,ε) = u0(t) + ε·u1(t) + ε²·u2(t) + ...
2. Substitute into F and series-expand in ε
3. At each order k, extract the ODE for uk(t)
4. Solve with dsolve (symbolic)
5. Apply boundary/initial conditions to fix integration constants
6. Detect secular terms — warn but do not remove
   (secular term removal is Lindstedt / multiple scales territory)
"""

from __future__ import annotations
from typing import List, Dict, Optional

from sympy import (
    Symbol, Function, symbols, Eq, series, dsolve, solve,
    expand, simplify, diff, latex, Add, Derivative,
    symbols as _symbols, sympify
)

from asymptotics.core.hierarchy  import OrderHierarchy, OrderEntry
from asymptotics.core.exceptions import (
    NoSmallParameterError,
    NoLeadingOrderSolutionError,
    NoHigherOrderSolutionError,
)


class ODEHierarchy:
    """
    Perturbation hierarchy for a single ODE.

    Attributes
    ----------
    entries : list of ODEOrderEntry
    composite : Expr  — the assembled expansion
    small_param : Symbol
    independent : Symbol
    """

    def __init__(self):
        self.entries      = []
        self.composite    = None
        self.small_param  = None
        self.independent  = None
        self._method      = ""
        self._problem_repr = ""
        self._problem_type = ""   # "ivp" or "bvp"

    def __getitem__(self, order: int):
        return self.entries[order]

    def __len__(self):
        return len(self.entries)


    def compare_numeric(self, eps, params=None, **kwargs):
        """
        Compare this expansion against a numerical solution.

        Parameters
        ----------
        eps : float
            Value of the small parameter ε to use.
        problem : ODE, optional
            The original problem object. Defaults to the equation
            used to create this hierarchy — usually not needed.
        **kwargs
            t_range   : [a, b]  — domain for plotting (ODE only)
            n_points  : int     — number of plot points (default 300)

        Returns
        -------
        dict with keys:
            't' or 'x'    : ndarray — evaluation points
            'u_pert'      : ndarray — perturbation composite
            'u_numerical' : ndarray — numerical solution
            'fig'         : matplotlib Figure
            (boundary layer also returns 'u_outer', 'u_inner', 'u_composite')
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, **kwargs)


    def to_latex(self, environment='align', show_orders=False, filename=None):
        """
        Export this expansion as LaTeX source.

        Parameters
        ----------
        environment : str
            LaTeX math environment: 'align' (default), 'equation', or 'gather'.
        show_orders : bool
            If True, include each order u_k separately. Default False.
        filename : str, optional
            If given, write to this file. Otherwise print to console.

        Returns
        -------
        str — the LaTeX source string

        Examples
        --------
        >>> print(sol.to_latex())
        >>> sol.to_latex(filename="result.tex")
        >>> sol.to_latex(environment='equation', show_orders=True)
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)


    def eval(self, eps, at=None, params=None):
        """
        Evaluate the perturbation composite at given eps and independent variable values.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter.
        at : array-like, optional
            Values of the independent variable (for ODEs).
            Not needed for algebraic equations.

        Returns
        -------
        For ODEs:
            ndarray if eps is scalar, dict {eps: ndarray} if eps is a list
        For algebraic:
            float if eps is scalar, ndarray if eps is a list

        Examples
        --------
        >>> # ODE
        >>> t_vals = np.linspace(0, 20, 300)
        >>> u = sol.eval(eps=0.1, at=t_vals)           # ndarray
        >>> u = sol.eval(eps=[0.1, 0.2], at=t_vals)    # dict {0.1: array, 0.2: array}
        >>>
        >>> # Algebraic
        >>> x = sol.eval(eps=0.1)                       # float
        >>> x = sol.eval(eps=[0.1, 0.2, 0.3])           # ndarray
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

    def show(self, orders=None, mode: str = "auto") -> None:
        from asymptotics.display.ode_display import show_ode
        show_ode(self, orders=orders, mode=mode)


class ODEOrderEntry:
    """All information about a single order of the ODE hierarchy."""

    def __init__(self, order, ode, general_solution,
                 particular_solution, constants, symbol, secular=False):
        self.order               = order
        self.ode                 = ode                  # Eq — the ODE at this order
        self.general_solution    = general_solution     # Expr — with free constants
        self.particular_solution = particular_solution  # Expr — constants fixed
        self.constants           = constants            # dict: {C1: val, ...}
        self.symbol              = symbol               # the Function u_k(t)
        self.secular             = secular              # True if secular terms detected
        # solution is an alias for particular_solution for display compatibility
        self.solution            = particular_solution


def expand_regular_ode(problem, order: int = 2) -> ODEHierarchy:
    """
    Apply regular perturbation theory to an ODE.

    Parameters
    ----------
    problem : ODE
    order : int

    Returns
    -------
    ODEHierarchy
    """
    eps    = problem.small_param
    t      = problem._indep_sym
    N      = order
    dep    = problem._dependent_name
    conds  = problem.conditions
    ptype  = problem.problem_type
    deriv_syms = problem._deriv_syms  # {1: du, 2: d2u, ...}
    ode_order  = problem.ode_order
    f          = problem.equation     # SymPy expr in terms of u, du, d2u, eps, t

    # ------------------------------------------------------------------
    # Check small parameter appears
    # ------------------------------------------------------------------
    if eps not in f.free_symbols:
        raise NoSmallParameterError(eps, f)

    # ------------------------------------------------------------------
    # Step 1: build u_k(t) as SymPy Functions
    # ------------------------------------------------------------------
    u_funcs = [Function(f"{dep}_{k}")(t) for k in range(N + 1)]

    # ------------------------------------------------------------------
    # Step 2: build ansatz
    #   u_ans = u_0(t) + eps*u_1(t) + eps^2*u_2(t) + ...
    # ------------------------------------------------------------------
    u_ans = sum(eps**k * u_funcs[k] for k in range(N + 1))

    # ------------------------------------------------------------------
    # Step 3: substitute ansatz into F
    # Replace derivative symbols with actual derivatives of u_ans
    # ------------------------------------------------------------------
    f_sub = f.subs(problem.dependent, u_ans)  # substitute u -> u_ans (symbol)

    # Also substitute derivative symbols
    for k_deriv, dsym in deriv_syms.items():
        f_sub = f_sub.subs(dsym, diff(u_ans, t, k_deriv))

    # Series expand in eps
    f_series = series(f_sub, eps, 0, N + 1)

    # ------------------------------------------------------------------
    # Step 4: collect coefficients at each order
    # ------------------------------------------------------------------
    coeffs = {}
    for k in range(N + 1):
        coeffs[k] = f_series.coeff(eps, k)

    # ------------------------------------------------------------------
    # Step 5: solve order by order
    # ------------------------------------------------------------------
    h = ODEHierarchy()
    h.small_param   = eps
    h.independent   = t
    h._method       = f"Regular perturbation — ODE ({'IVP' if ptype == 'ivp' else 'BVP'})"
    h._problem_repr = f"F({dep}, {dep}', t, {eps}) = 0"
    h._problem_type = ptype

    known_solutions = {}   # u_k(t) -> particular solution expr

    for k in range(N + 1):
        uk = u_funcs[k]

        # Substitute known particular solutions into order-k equation
        ode_expr = coeffs[k]
        for func, sol_expr in known_solutions.items():
            ode_expr = ode_expr.subs(func, sol_expr)

        ode_expr = expand(ode_expr)

        # Pre-expand trig powers to linear cos/sin — critical for dsolve speed
        # e.g. t*cos^2(t)*sin(t) -> linear combination of sin(nt), cos(nt)
        from sympy import exp as _exp, cos as _cos
        ode_expr = expand(expand(ode_expr.rewrite(_exp)).rewrite(_cos))
        ode_eq   = Eq(ode_expr, 0)

        # Solve ODE symbolically
        try:
            gen_sol = dsolve(ode_eq, uk)
        except Exception as e:
            if k == 0:
                raise NoLeadingOrderSolutionError(ode_eq, eps, uk) from e
            else:
                raise NoHigherOrderSolutionError(k, uk, ode_eq, known_solutions) from e

        # gen_sol is Eq(uk, expr_with_constants)
        gen_expr = gen_sol.rhs

        # ------------------------------------------------------------------
        # Step 6: apply conditions to fix integration constants
        # ------------------------------------------------------------------
        # Get free constants (C1, C2, ...)
        free_consts = sorted(
            [s for s in gen_expr.free_symbols
             if str(s).startswith('C') and str(s)[1:].isdigit()],
            key=lambda s: int(str(s)[1:])
        )

        # Build order-k conditions
        # At order 0: full condition value
        # At order k>0: homogeneous (value = 0) since correction terms are zero
        #   UNLESS the condition itself has eps dependence (not supported yet)
        cond_equations = []
        for cond in conds:
            pt  = cond.point
            val = cond.value if k == 0 else sympify(0)
            # Evaluate the k-th order solution derivative at the point
            if cond.deriv_order == 0:
                expr_at_pt = gen_expr.subs(t, pt)
            else:
                expr_at_pt = diff(gen_expr, t, cond.deriv_order).subs(t, pt)
            cond_equations.append(Eq(expr_at_pt, val))

        # Solve for integration constants
        try:
            const_sol = solve(cond_equations, free_consts)
        except Exception:
            const_sol = {}

        if not const_sol and free_consts:
            raise NoHigherOrderSolutionError(
                k, free_consts, cond_equations, known_solutions
            )

        # Substitute constants
        if isinstance(const_sol, dict):
            part_expr = gen_expr.subs(const_sol)
        elif isinstance(const_sol, list) and const_sol:
            part_expr = gen_expr.subs(zip(free_consts, const_sol[0] if isinstance(const_sol[0], (list, tuple)) else [const_sol[0]]))
        else:
            part_expr = gen_expr

        part_expr = expand(part_expr)

        # Detect secular terms (terms growing in t)
        secular = _has_secular_terms(part_expr, t)

        known_solutions[uk] = part_expr

        entry = ODEOrderEntry(
            order            = k,
            ode              = ode_eq,
            general_solution = gen_expr,
            particular_solution = part_expr,
            constants        = const_sol if isinstance(const_sol, dict) else {},
            symbol           = uk,
            secular          = secular,
        )
        h.entries.append(entry)

    # ------------------------------------------------------------------
    # Step 7: composite expansion
    # ------------------------------------------------------------------
    h.composite = Add(*[known_solutions[u_funcs[k]] * eps**k for k in range(N + 1)])

    h._problem = problem
    return h


def _has_secular_terms(expr, t) -> bool:
    """
    Detect secular (unbounded) terms — polynomial growth in t
    multiplied by oscillatory terms like sin, cos.
    e.g.  t*sin(t),  t*cos(t),  t**2*exp(t)
    """
    from sympy import sin, cos, Mul, Pow
    expr_expanded = expand(expr)

    def _check(e):
        if e.is_Mul:
            has_t_power = any(
                (a == t or (a.is_Pow and a.base == t))
                for a in e.args
            )
            has_trig = any(
                isinstance(a, (sin, cos)) for a in e.args
            )
            if has_t_power and has_trig:
                return True
        return False

    from sympy import preorder_traversal
    return any(_check(sub) for sub in preorder_traversal(expr_expanded))
