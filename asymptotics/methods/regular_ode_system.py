"""
asymptotics.methods.regular_ode_system
====================================
Regular perturbation expansion for coupled ODE systems.

Algorithm
---------
At each order k, the equations for u_k^{(i)}(t) are DECOUPLED:
each depends only on lower-order solutions u_j^{(i)} for j < k.

This is because coupling terms like eps*v always involve
the other variable at a previous order after substitution.

So at each order we solve N independent ODEs, one per variable.
"""

from __future__ import annotations
from sympy import (
    Symbol, Function, symbols, series, expand,
    diff, dsolve, solve, Add, Integer, Eq, simplify,
    cos, exp
)
from asymptotics.core.exceptions import NoSmallParameterError


# ---------------------------------------------------------------------------
# Per-variable hierarchy entry
# ---------------------------------------------------------------------------

class ODESystemOrderEntry:
    """One order of the expansion for a single variable."""

    def __init__(self, order, ode, general_solution, particular_solution, symbol):
        self.order               = order
        self.ode                 = ode
        self.general_solution    = general_solution
        self.particular_solution = particular_solution
        self.symbol              = symbol
        self.solution            = particular_solution   # alias


class ODESystemVarHierarchy:
    """
    Perturbation hierarchy for ONE variable in a coupled system.
    Mimics ODEHierarchy so the same display/access patterns work.
    """

    def __init__(self, name):
        self.name      = name
        self.entries   = []
        self.expansion = None

    def __getitem__(self, order: int):
        return self.entries[order]

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Top-level hierarchy container
# ---------------------------------------------------------------------------

class ODESystemHierarchy:
    """
    Result of a regular perturbation expansion for a coupled ODE system.

    Access per-variable results via:
        sol["u"].expansion
        sol["u"][k].particular_solution
        sol["v"].expansion

    Attributes
    ----------
    variables    : list of str
    hierarchies  : dict — {var_name: ODESystemVarHierarchy}
    small_param  : Symbol
    independent  : Symbol
    _method      : str
    """

    def __init__(self):
        self.variables   = []
        self.hierarchies = {}
        self.small_param = None
        self.independent = None
        self._method     = "Regular perturbation — ODE system"

    def __getitem__(self, var: str) -> ODESystemVarHierarchy:
        if var not in self.hierarchies:
            raise KeyError(
                f"\n\n  Variable '{var}' not in system.\n"
                f"  Available: {self.variables}\n"
            )
        return self.hierarchies[var]


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
        Evaluate the perturbation expansion at given eps and independent variable values.

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

    def show(self, mode: str = "auto") -> None:
        from asymptotics.display.ode_system_display import show_ode_system
        show_ode_system(self, mode=mode)

    def compare_numeric(self, eps, params=None, **kwargs):
        """
        Compare this expansion against a numerical solution.

        Parameters
        ----------
        eps : float
        problem : ODESystem — the original problem object (optional, inferred automatically)
        plot_range : [a, b]
        n_points : int

        Returns
        -------
        dict with 't', 'u_pert', 'u_numerical', 'fig'
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, problem=problem, **kwargs)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def expand_regular_ode_system(problem, order: int = 2) -> ODESystemHierarchy:
    """
    Apply regular perturbation to a coupled ODE system.

    Parameters
    ----------
    problem : ODESystem
    order : int

    Returns
    -------
    ODESystemHierarchy
    """
    eps  = problem.small_param
    t    = problem.independent
    deps = problem.dependent_names
    N    = order
    C1, C2 = symbols('C1 C2')

    # Check eps appears in at least one equation
    has_eps = any(eps in problem.equations[dep].free_symbols for dep in deps)
    if not has_eps:
        raise NoSmallParameterError(eps, list(problem.equations.values())[0])

    # ------------------------------------------------------------------
    # Build u_k^{(i)}(t) Function objects for each variable and order
    # u_funcs[dep][k] = Function('dep_k')(t)
    # ------------------------------------------------------------------
    u_funcs = {
        dep: [Function(f"{dep}_{k}")(t) for k in range(N + 1)]
        for dep in deps
    }

    # ------------------------------------------------------------------
    # Build ansatz for each variable:
    # dep_ans[dep] = dep_0(t) + eps*dep_1(t) + eps^2*dep_2(t) + ...
    # ------------------------------------------------------------------
    u_ans = {
        dep: sum(eps**k * u_funcs[dep][k] for k in range(N + 1))
        for dep in deps
    }

    # ------------------------------------------------------------------
    # Substitute ansatz into each equation and expand in eps
    # ------------------------------------------------------------------
    # For each equation for dep i, substitute ALL dependent variables
    # and their derivatives with the ansatz
    coeffs = {dep: {} for dep in deps}

    for dep in deps:
        f_orig = problem.equations[dep]
        dep_syms   = problem._dep_syms
        deriv_syms = problem._deriv_syms

        f_sub = f_orig

        # Substitute each variable's ansatz
        for d in deps:
            f_sub = f_sub.subs(dep_syms[d], u_ans[d])
            # Substitute derivative symbols
            for k_deriv, dsym in deriv_syms[d].items():
                f_sub = f_sub.subs(dsym, diff(u_ans[d], t, k_deriv))

        # Series expand in eps
        f_series = series(f_sub, eps, 0, N + 1)
        for k in range(N + 1):
            coeffs[dep][k] = expand(f_series.coeff(eps, k))

    # ------------------------------------------------------------------
    # Solve order by order — variables are decoupled at each order
    # ------------------------------------------------------------------
    h = ODESystemHierarchy()
    h.variables   = deps
    h.small_param = eps
    h.independent = t
    h._method     = "Regular perturbation — ODE system"

    # Initialize per-variable hierarchies
    for dep in deps:
        h.hierarchies[dep] = ODESystemVarHierarchy(dep)

    # Track known solutions: {Function_k(t): expr}
    known = {}   # all variables all orders

    for k in range(N + 1):
        for dep in deps:
            uk = u_funcs[dep][k]

            # Substitute known lower-order solutions
            ode_expr = coeffs[dep][k]
            for func, sol_expr in known.items():
                ode_expr = ode_expr.subs(func, sol_expr)
            ode_expr = expand(ode_expr)

            # Rewrite to exp form so dsolve can use variation of parameters /
            # undetermined coefficients cleanly.  Do NOT rewrite back to cos/sin
            # here — that converts exp(-t) → cosh(t)-sinh(t) which breaks
            # undetermined coefficients for terms like t*exp(-2t).
            from sympy import exp as _exp
            ode_expr = expand(ode_expr.rewrite(_exp))
            ode_eq   = Eq(ode_expr, 0)

            # Get the ODE order for this variable
            ode_order = problem.ode_orders[dep]

            # Solve
            try:
                gen_sol  = dsolve(ode_eq, uk)
                if isinstance(gen_sol, list):
                    gen_sol = gen_sol[0]
                gen_expr = gen_sol.rhs
            except Exception as e:
                raise RuntimeError(
                    f"\n\n  Could not solve order-{k} ODE for '{dep}':\n"
                    f"  {ode_eq}\n"
                    f"  Error: {e}\n"
                ) from e

            # Apply conditions
            conds_dep = problem.conditions.get(dep, [])

            # k=0: use original ICs; k>0: homogeneous ICs (value=0)
            free_consts = sorted(
                [s for s in gen_expr.free_symbols
                 if str(s).startswith('C') and str(s)[1:].isdigit()],
                key=lambda s: int(str(s)[1:])
            )

            ic_eqs = []
            for cond in conds_dep:
                val = cond.value if k == 0 else Integer(0)
                if cond.deriv_order == 0:
                    ic_eqs.append(Eq(gen_expr.subs(t, cond.point), val))
                else:
                    ic_eqs.append(Eq(
                        diff(gen_expr, t, cond.deriv_order).subs(t, cond.point),
                        val
                    ))

            try:
                const_sol = solve(ic_eqs, free_consts)
            except Exception:
                const_sol = {}

            if isinstance(const_sol, dict):
                part_expr = expand(gen_expr.subs(const_sol))
            else:
                part_expr = gen_expr

            known[uk] = part_expr

            entry = ODESystemOrderEntry(
                order               = k,
                ode                 = ode_eq,
                general_solution    = gen_expr,
                particular_solution = part_expr,
                symbol              = uk,
            )
            h.hierarchies[dep].entries.append(entry)

    # ------------------------------------------------------------------
    # Assemble expansions
    # ------------------------------------------------------------------
    for dep in deps:
        h.hierarchies[dep].expansion = Add(*[
            known[u_funcs[dep][k]] * eps**k for k in range(N + 1)
        ])

    h._problem = problem
    return h
