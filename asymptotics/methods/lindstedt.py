"""
asymptotics.methods.lindstedt
==========================
Lindstedt–Poincaré method for nonlinear oscillators.

For equations of the form:
    u'' + omega_0^2 * u + eps * f(u, u', t) = 0

Algorithm
---------
1. Detect omega_0 from the unperturbed O(1) equation
2. Introduce strained time: tau = omega(eps) * t
   omega = omega_0 + eps*omega_1 + eps^2*omega_2 + ...
3. At each order k:
   a. Build the O(eps^k) ODE using incremental convolution — fast, no series()
   b. Extract secular coefficients from the FORCING only
   c. Solve for omega_k to eliminate secular terms
   d. Solve the de-secularized ODE with homogeneous ICs

Speed: O(N^2) convolutions, no large polynomial expansion.
"""

from __future__ import annotations

from sympy import (
    Symbol, Function, symbols, Eq, dsolve, solve,
    expand, simplify, diff, Add, cos, sin, exp,
    sqrt, Rational, Integer, preorder_traversal
)


# ---------------------------------------------------------------------------
# Hierarchy containers
# ---------------------------------------------------------------------------

class LindstedtHierarchy:
    """
    Result of a Lindstedt–Poincaré expansion.

    Attributes
    ----------
    entries         : list of LindstedtOrderEntry
    omega_expansion : Expr   — omega = omega_0 + eps*omega_1 + ...
    omega_values    : dict   — {omega_k_sym: value}
    expansion       : Expr   — u(tau, eps) assembled
    expansion_t     : Expr   — u(t, eps) with tau = omega*t
    small_param     : Symbol
    independent     : Symbol — t
    tau             : Symbol — strained time
    omega_0         : Expr   — unperturbed frequency
    """

    def __init__(self):
        self.entries         = []
        self.omega_expansion = None
        self.omega_values    = {}
        self.expansion       = None
        self.expansion_t     = None
        self.small_param     = None
        self.independent     = None
        self.tau             = None
        self.omega_0         = None
        self._method         = "Lindstedt–Poincaré"
        self._problem_repr   = ""

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
            plot_range : [a, b]  — domain for plotting (ODE only)
            n_points  : int     — number of plot points (default 300)

        Returns
        -------
        dict with keys:
            't' or 'x'    : ndarray — evaluation points
            'u_pert'      : ndarray — perturbation expansion
            'u_numerical' : ndarray — numerical solution
            'fig'         : matplotlib Figure
            (boundary layer also returns 'u_outer', 'u_inner', 'u_expansion')
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, problem=problem, **kwargs)


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

    def show(self, orders=None, mode: str = "auto") -> None:
        from asymptotics.display.lindstedt_display import show_lindstedt
        show_lindstedt(self, orders=orders, mode=mode)


class LindstedtOrderEntry:
    """One order of the Lindstedt hierarchy."""

    def __init__(self, order, ode, secularity_condition,
                 omega_k_sym, omega_k_val,
                 general_solution, particular_solution, symbol):
        self.order                = order
        self.ode                  = ode
        self.secularity_condition = secularity_condition
        self.omega_k_sym          = omega_k_sym
        self.omega_k_val          = omega_k_val
        self.general_solution     = general_solution
        self.particular_solution  = particular_solution
        self.symbol               = symbol
        self.solution             = particular_solution   # alias

    @property
    def secular(self):
        """True if secular (resonant) terms were detected and eliminated at
        this order.  For Lindstedt the secularity condition is always enforced,
        so this is True whenever ``secularity_condition`` is non-trivial."""
        from sympy import S
        sc = self.secularity_condition
        if sc is None:
            return False
        try:
            return bool(sc != S.Zero)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _secular_coefficients(forcing, tau, omega0):
    """
    Extract coefficients of cos(omega0*tau) and sin(omega0*tau) in forcing.
    Uses rewrite(exp) to handle trig powers like cos^3(tau).
    """
    forcing_reduced = expand(expand(forcing.rewrite(exp)).rewrite(cos))
    coeff_cos = forcing_reduced.coeff(cos(omega0 * tau))
    coeff_sin = forcing_reduced.coeff(sin(omega0 * tau))
    return coeff_cos, coeff_sin


def _conv(a, b, k):
    """k-th coefficient of product of two series (as lists of coefficients)."""
    return expand(sum(
        a[i] * b[k - i]
        for i in range(k + 1)
        if i < len(a) and k - i < len(b)
    ))


def _detect_omega0(f_orig, d2u_sym, u_sym, eps_sym):
    """
    Extract omega_0 from the unperturbed equation f(u, u'', eps=0) = 0.
    Equation must have the form: d2u_coeff * u'' + u_coeff * u = 0
    => omega_0 = sqrt(u_coeff / d2u_coeff)
    """
    f0 = f_orig.subs(eps_sym, 0)
    d2u_coeff = f0.coeff(d2u_sym)
    u_coeff   = f0.coeff(u_sym)

    if d2u_coeff == 0:
        raise ValueError(
            "\n\n  Could not find u'' term in unperturbed equation.\n"
            "  Lindstedt–Poincaré requires u'' + ω₀²·u + ε·f = 0.\n"
        )

    omega0_sq = simplify(u_coeff / d2u_coeff)

    if not (omega0_sq.is_real and omega0_sq > 0):
        raise ValueError(
            f"\n\n  Unperturbed frequency squared is non-positive: omega_0^2 = {omega0_sq}\n"
            "  Lindstedt-Poincare requires an oscillatory problem (omega_0^2 > 0).\n"
            "  Suggestions: use expand_regular() for non-oscillatory problems,\n"
            "  or expand_multiple_scales() for damped oscillators.\n"
        )

    return sqrt(omega0_sq)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def expand_lindstedt(problem, order: int = 2) -> LindstedtHierarchy:
    """
    Apply the Lindstedt–Poincaré method to a nonlinear oscillator ODE.

    Parameters
    ----------
    problem : ODE
    order : int

    Returns
    -------
    LindstedtHierarchy
    """
    from asymptotics.core.exceptions import NoSmallParameterError

    eps        = problem.small_param
    t          = problem._indep_sym
    dep        = problem._dependent_name
    f_orig     = problem.equation
    deriv_syms = problem._deriv_syms
    conds      = problem.conditions
    N          = order

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if eps not in f_orig.free_symbols:
        raise NoSmallParameterError(eps, f_orig)

    if problem.ode_order != 2:
        raise ValueError(
            f"\n\n  Lindstedt–Poincaré requires a 2nd-order ODE.\n"
            f"  Got order {problem.ode_order}.\n"
        )

    if problem.problem_type == "bvp":
        raise ValueError(
            "\n\n  Lindstedt–Poincaré does not apply to BVPs.\n"
            "  The method requires all conditions at t=0 (IVP).\n\n"
            "  For BVPs with nonlinear oscillations, consider\n"
            "  expand_regular() instead, or reformulate as an IVP.\n"
        )

    du_sym  = deriv_syms.get(1)
    d2u_sym = deriv_syms.get(2)
    u_sym   = Symbol(dep)

    # ------------------------------------------------------------------
    # Step 1: detect omega_0 from unperturbed equation
    # ------------------------------------------------------------------
    omega0 = _detect_omega0(f_orig, d2u_sym, u_sym, eps)

    # ------------------------------------------------------------------
    # Step 2: set up strained time and symbols
    # ------------------------------------------------------------------
    tau      = Symbol('tau')
    C1, C2   = symbols('C1 C2')

    omega_syms = [Symbol(f'omega_{k}') for k in range(1, N + 1)]
    u_funcs    = [Function(f'{dep}_{k}')(tau) for k in range(N + 1)]

    # omega series as list: [omega_0, omega_1, ..., omega_N]
    omega_list = [omega0] + list(omega_syms)

    # Precompute omega^2 coefficients: (omega^2)_k = sum_{i=0}^k omega_i * omega_{k-i}
    omega_sq = [
        expand(sum(omega_list[i] * omega_list[k - i] for i in range(k + 1)))
        for k in range(N + 1)
    ]

    # Series of u_k functions and their derivatives
    u_cf   = list(u_funcs)
    d2u_cf = [diff(u, tau, 2) for u in u_cf]
    du_cf  = [diff(u, tau)    for u in u_cf]

    # ------------------------------------------------------------------
    # Step 3: build O(eps^k) coefficients incrementally
    #
    # The equation in tau-space is:
    #   omega^2 * u_tau_tau + omega_0^2 * u / (omega_0^2 / omega^2) ... wait,
    # more carefully:
    #
    # Original: d2u_coeff * u_tt + u_coeff * u + eps * g(u) = 0
    # where d2u_coeff and u_coeff are extracted from f_orig at eps=0.
    # In tau: u_tt = omega^2 * u_tau_tau
    # So:     d2u_coeff * omega^2 * u_tau_tau + u_coeff * u + eps * g(u) = 0
    #
    # The O(k) coefficient is:
    #   d2u_coeff * (omega^2 * u_tau_tau)_k + u_coeff * u_k + (eps*g)_k
    # ------------------------------------------------------------------

    # Extract linear coefficients from f_orig at eps=0
    f0         = f_orig.subs(eps, 0)
    d2u_coeff  = f0.coeff(d2u_sym)
    u_lin_coeff = f0.coeff(u_sym)
    du_coeff   = f0.coeff(du_sym) if du_sym else Integer(0)

    # Nonlinear residual (everything not in linear skeleton)
    f_linear = d2u_coeff * d2u_sym + u_lin_coeff * u_sym
    if du_sym:
        f_linear += du_coeff * du_sym
    f_nonlin = f_orig - f_linear   # e.g. eps*u^3 for Duffing

    # Precompute nonlinear contributions:
    # f_nonlin with u -> sum u_k*eps^k, then extract eps^k coefficient
    # Use series() only on f_nonlin which is simple (e.g. eps*u^3)
    u_subs  = Add(*[u_cf[k] * eps**k for k in range(N + 1)])
    f_nl    = f_nonlin.subs(u_sym, u_subs)
    if du_sym:
        du_subs = Add(*[du_cf[k] * eps**k for k in range(N + 1)])
        f_nl = f_nl.subs(du_sym, du_subs)
    # Note: d2u_sym not substituted in f_nonlin since it's in the linear part

    from sympy import series as _series
    f_nl_series = _series(f_nl, eps, 0, N + 1)
    nl_cf = [expand(f_nl_series.coeff(eps, k)) for k in range(N + 1)]

    # Build O(k) coefficients
    coeffs = {}
    for k in range(N + 1):
        # d2u_coeff * (omega^2 * u_tau_tau)_k
        d2_term = expand(d2u_coeff * sum(
            omega_sq[j] * d2u_cf[k - j]
            for j in range(k + 1)
        ))
        # u_coeff * u_k
        u_term  = expand(u_lin_coeff * u_cf[k])
        # du term if present
        du_term = expand(du_coeff * sum(
            omega_list[j] * du_cf[k - j]
            for j in range(k + 1)
        )) if du_sym else Integer(0)
        # nonlinear contribution
        nl_term = nl_cf[k]

        coeffs[k] = expand(d2_term + u_term + du_term + nl_term)

    # ------------------------------------------------------------------
    # Step 4: solve O(1) with ICs
    # ------------------------------------------------------------------
    u0   = u_funcs[0]
    ode0 = Eq(coeffs[0], 0)

    try:
        sol0 = dsolve(ode0, u0)
    except Exception as e:
        raise ValueError(
            f"\n\n  Could not solve O(1) equation: {ode0}\n  Error: {e}\n"
        ) from e

    # Apply ICs at tau=0
    ic_eqs_0 = []
    for cond in conds:
        if simplify(cond.point) != 0:
            continue
        expr = sol0.rhs
        if cond.deriv_order == 1:
            expr = diff(expr, tau) * omega0
        elif cond.deriv_order >= 2:
            expr = diff(expr, tau, cond.deriv_order) * omega0**cond.deriv_order
        ic_eqs_0.append(Eq(expr.subs(tau, 0), cond.value))

    const_sol_0  = solve(ic_eqs_0, [C1, C2])
    u0_particular = simplify(sol0.rhs.subs(const_sol_0))

    # ------------------------------------------------------------------
    # Build hierarchy
    # ------------------------------------------------------------------
    h = LindstedtHierarchy()
    h.small_param   = eps
    h.independent   = t
    h.tau           = tau
    h.omega_0       = omega0
    h._method       = "Lindstedt–Poincaré"
    h._problem_repr = f"{dep}'' + {omega0}²·{dep} + {eps}·f = 0"

    h.entries.append(LindstedtOrderEntry(
        order                = 0,
        ode                  = ode0,
        secularity_condition = None,
        omega_k_sym          = None,
        omega_k_val          = None,
        general_solution     = sol0.rhs,
        particular_solution  = u0_particular,
        symbol               = u0,
    ))
    h.omega_values[Symbol('omega_0')] = omega0

    # ------------------------------------------------------------------
    # Step 5: higher orders
    # ------------------------------------------------------------------
    known_u     = {u0: u0_particular}
    known_omega = {}

    for k in range(1, N + 1):
        uk     = u_funcs[k]
        ok_sym = omega_syms[k - 1]

        # Substitute known solutions and omega values, evaluate derivatives
        ode_k_expr = coeffs[k]
        for func, sol_expr in known_u.items():
            ode_k_expr = ode_k_expr.subs(func, sol_expr)
        for osym, oval in known_omega.items():
            ode_k_expr = ode_k_expr.subs(osym, oval)
        ode_k_expr = expand(ode_k_expr.doit())

        # Extract the forcing (everything except uk'' and omega_0^2*uk terms)
        uk_d2   = diff(uk, tau, 2)
        forcing = -(ode_k_expr - d2u_coeff * uk_d2 - u_lin_coeff * uk)
        forcing = expand(forcing)

        # Find secular coefficients in forcing (function of ok_sym)
        coeff_cos, coeff_sin = _secular_coefficients(forcing, tau, omega0)

        # Solve secularity condition for ok_sym
        omega_k_val    = None
        secularity_eq  = None

        for sec_coeff in [coeff_cos, coeff_sin]:
            if sec_coeff != 0 and ok_sym in sec_coeff.free_symbols:
                secularity_eq = Eq(sec_coeff, 0)
                sols = solve(sec_coeff, ok_sym)
                if sols:
                    omega_k_val = simplify(sols[0])
                    break

        if omega_k_val is None:
            omega_k_val   = Integer(0)
            secularity_eq = Eq(Integer(0), 0)

        known_omega[ok_sym]    = omega_k_val
        h.omega_values[ok_sym] = omega_k_val

        # Substitute omega_k and rebuild forcing
        forcing = expand(forcing.subs(ok_sym, omega_k_val))

        # Pre-expand trig powers to linear cos/sin — critical for dsolve speed
        # e.g. cos^3(tau) -> 3/4*cos(tau) + 1/4*cos(3*tau)
        forcing = expand(expand(forcing.rewrite(exp)).rewrite(cos))

        # Build and solve the ODE for uk
        ode_k_eq = Eq(d2u_coeff * uk_d2 + u_lin_coeff * uk, forcing)

        try:
            gen_sol_k = dsolve(ode_k_eq, uk)
        except Exception as e:
            raise ValueError(
                f"\n\n  Could not solve order-{k} ODE: {ode_k_eq}\n  Error: {e}\n"
            ) from e

        gen_expr_k = gen_sol_k.rhs

        # Apply homogeneous ICs: uk(0) = 0, uk'(0) = 0
        ic_eqs_k = [
            Eq(gen_expr_k.subs(tau, 0), 0),
            Eq(diff(gen_expr_k, tau).subs(tau, 0), 0),
        ]
        const_sol_k  = solve(ic_eqs_k, [C1, C2])
        part_expr_k  = simplify(gen_expr_k.subs(const_sol_k))

        known_u[uk] = part_expr_k

        h.entries.append(LindstedtOrderEntry(
            order                = k,
            ode                  = ode_k_eq,
            secularity_condition = secularity_eq,
            omega_k_sym          = ok_sym,
            omega_k_val          = omega_k_val,
            general_solution     = gen_expr_k,
            particular_solution  = part_expr_k,
            symbol               = uk,
        ))

    # ------------------------------------------------------------------
    # Step 6: assemble results
    # ------------------------------------------------------------------
    h.expansion = Add(*[known_u[u_funcs[k]] * eps**k for k in range(N + 1)])

    omega_expr = omega0 + sum(
        eps**k * h.omega_values.get(omega_syms[k - 1], Integer(0))
        for k in range(1, N + 1)
    )
    h.omega_expansion = omega_expr
    h.expansion_t     = h.expansion.subs(tau, omega_expr * t)

    h._problem = problem
    return h
