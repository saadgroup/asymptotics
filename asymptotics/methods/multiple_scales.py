"""
asymptotics.methods.multiple_scales
================================
Method of multiple scales for weakly nonlinear oscillators.

For equations of the form:
    u'' + omega_0^2 * u + eps * f(u, u', t) = 0

Introduces two timescales:
    T_0 = t           (fast — oscillation)
    T_1 = eps * t     (slow — amplitude/phase modulation)

And writes u = u_0(T_0, T_1) + eps * u_1(T_0, T_1) + ...

Algorithm at each order k:
    1. Build the O(eps^k) PDE for u_k(T_0, T_1)
    2. The O(1) solution is u_0 = A(T_1)*cos(omega_0*T_0) + B(T_1)*sin(omega_0*T_0)
    3. Extract secular terms (resonant forcing at omega_0)
    4. Set secular coefficients to zero — this gives ODEs for A(T_1), B(T_1)
       (the solvability / amplitude equations)
    5. Solve solvability ODEs with ICs from the original problem
    6. Substitute back and solve the de-secularized PDE for u_k

The composite solution is then:
    u(t, eps) = u_0(t, eps*t) + eps*u_1(t, eps*t) + ...
"""

from __future__ import annotations

from sympy import (
    Symbol, Function, symbols, Eq, dsolve, solve,
    expand, simplify, diff, Add, cos, sin, exp,
    sqrt, Integer, Rational, preorder_traversal,
    collect, trigsimp
)


# ---------------------------------------------------------------------------
# Hierarchy containers
# ---------------------------------------------------------------------------

class MultScalesHierarchy:
    """
    Result of a multiple-scales expansion.

    Attributes
    ----------
    entries             : list of MultScalesOrderEntry
    composite           : Expr  — u(T_0, T_1, eps) assembled
    composite_t         : Expr  — u(t, eps) with T_0=t, T_1=eps*t
    amplitude_A         : Expr  — A(T_1) solved
    amplitude_B         : Expr  — B(T_1) solved
    solvability_odes    : list  — [Eq(dA/dT1, ...), Eq(dB/dT1, ...)]
    small_param         : Symbol
    independent         : Symbol — t
    T0, T1              : Symbol — fast and slow times
    omega_0             : Expr
    """

    def __init__(self):
        self.entries          = []
        self.composite        = None
        self.composite_t      = None
        self.amplitude_A      = None
        self.amplitude_B      = None
        self.solvability_odes = []
        self.small_param      = None
        self.independent      = None
        self.T0               = None
        self.T1               = None
        self.omega_0          = None
        self._method          = "Multiple Scales"
        self._problem_repr    = ""

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
        from asymptotics.display.multiple_scales_display import show_multiple_scales
        show_multiple_scales(self, orders=orders, mode=mode)


class MultScalesOrderEntry:
    """One order of the multiple-scales hierarchy."""

    def __init__(self, order, pde, secular_cos, secular_sin,
                 solvability_A, solvability_B,
                 particular_solution, symbol):
        self.order              = order
        self.pde                = pde                # PDE for u_k
        self.secular_cos        = secular_cos        # coeff of cos(omega0*T0)
        self.secular_sin        = secular_sin        # coeff of sin(omega0*T0)
        self.solvability_A      = solvability_A      # Eq(dA/dT1, ...)
        self.solvability_B      = solvability_B      # Eq(dB/dT1, ...)
        self.particular_solution = particular_solution
        self.symbol             = symbol
        self.solution           = particular_solution  # alias


# ---------------------------------------------------------------------------
# Helper: secular coefficient extraction
# ---------------------------------------------------------------------------

def _secular_coefficients(forcing, T0, omega0):
    """Extract coefficients of cos(omega0*T0) and sin(omega0*T0)."""
    forcing_reduced = expand(expand(forcing.rewrite(exp)).rewrite(cos))
    coeff_cos = forcing_reduced.coeff(cos(omega0 * T0))
    coeff_sin = forcing_reduced.coeff(sin(omega0 * T0))
    return coeff_cos, coeff_sin


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def expand_multiple_scales(problem, order: int = 1) -> MultScalesHierarchy:
    """
    Apply the method of multiple scales to a nonlinear oscillator ODE.

    Parameters
    ----------
    problem : ODE
        2nd-order oscillator: u'' + omega_0^2*u + eps*f(u,u',t) = 0
    order : int
        Number of epsilon corrections (default 1). Each order adds one
        slow-time scale T_k = eps^k * t.

    Returns
    -------
    MultScalesHierarchy
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
            f"\n\n  Multiple scales requires a 2nd-order ODE.\n"
            f"  Got order {problem.ode_order}.\n"
        )

    du_sym  = deriv_syms.get(1)
    d2u_sym = deriv_syms.get(2)
    u_sym   = Symbol(dep)

    # ------------------------------------------------------------------
    # Step 1: detect omega_0 from unperturbed equation
    # ------------------------------------------------------------------
    f0         = f_orig.subs(eps, 0)
    d2u_coeff  = f0.coeff(d2u_sym)
    u_coeff    = f0.coeff(u_sym)
    du_coeff   = f0.coeff(du_sym) if du_sym else Integer(0)

    if d2u_coeff == 0:
        raise ValueError(
            "\n\n  Could not find u'' term in unperturbed equation.\n"
            "  Multiple scales requires u'' + ω₀²·u + ε·f = 0.\n"
        )

    omega0_sq = simplify(u_coeff / d2u_coeff)
    if not (omega0_sq.is_real and omega0_sq > 0):
        raise ValueError(
            f"\n\n  Unperturbed frequency squared is non-positive: ω₀² = {omega0_sq}\n"
            "  Multiple scales requires an oscillatory unperturbed solution.\n"
        )
    omega0 = sqrt(omega0_sq)

    # ------------------------------------------------------------------
    # Step 2: set up timescales and symbols
    # ------------------------------------------------------------------
    T0 = Symbol('T_0')   # fast time = t
    T1 = Symbol('T_1')   # slow time = eps*t

    # A(T1), B(T1) — amplitude functions at leading order
    A = Function('A')(T1)
    B = Function('B')(T1)

    # u_k functions depend on both T0 and T1
    u_funcs = [Function(f'{dep}_{k}')(T0, T1) for k in range(N + 1)]

    # ------------------------------------------------------------------
    # Step 3: O(1) solution
    # u_0 = A(T1)*cos(omega0*T0) + B(T1)*sin(omega0*T0)
    # ------------------------------------------------------------------
    u0_expr = A * cos(omega0 * T0) + B * sin(omega0 * T0)

    # Apply ICs to determine A(0) and B(0)
    # u(0) = u_0(0,0) = A(0) => A(0) = u(0)
    # u'(0) = D0*u_0(0,0)*omega0 = omega0*B(0) => B(0) = u'(0)/omega0
    A0_val = Integer(0)
    B0_val = Integer(0)
    for cond in conds:
        if simplify(cond.point) != 0:
            continue
        if cond.deriv_order == 0:
            A0_val = cond.value
        elif cond.deriv_order == 1:
            B0_val = simplify(cond.value / omega0)

    h = MultScalesHierarchy()
    h.small_param    = eps
    h.independent    = t
    h.T0             = T0
    h.T1             = T1
    h.omega_0        = omega0
    h._method        = "Multiple Scales"
    h._problem_repr  = f"{dep}'' + {omega0}²·{dep} + {eps}·f = 0"

    h.entries.append(MultScalesOrderEntry(
        order              = 0,
        pde                = Eq(diff(u_funcs[0], T0, 2) + omega0**2 * u_funcs[0], 0),
        secular_cos        = None,
        secular_sin        = None,
        solvability_A      = None,
        solvability_B      = None,
        particular_solution = u0_expr,
        symbol             = u_funcs[0],
    ))

    # ------------------------------------------------------------------
    # Step 4: higher orders
    # ------------------------------------------------------------------
    # Nonlinear residual from f_orig
    f_linear = d2u_coeff * d2u_sym + u_coeff * u_sym
    if du_sym and du_sym in f_orig.free_symbols:
        f_linear += du_coeff * du_sym
    f_nonlin = f_orig - f_linear   # e.g. eps*u^3

    # Current best u_0 expression (will be updated with solved A, B)
    u0_current  = u0_expr
    A_sol_expr  = A   # will be replaced after solving solvability
    B_sol_expr  = B

    solvability_odes = []

    for k in range(1, N + 1):
        uk = u_funcs[k]

        # Build O(eps^k) forcing:
        # From the chain rule: d/dt = D0 + eps*D1
        # d2/dt2 = D0^2 + 2*eps*D0*D1 + eps^2*D1^2
        # At O(eps): D0^2*u1 + omega0^2*u1 = -2*D0*D1*u0 - (du_coeff*D0*u0) - f_nonlin_k
        # where f_nonlin_k is the eps^1 part of f_nonlin with u->u0

        # Compute cross-derivative term: 2*D0*D1*u0
        D0_D1_u0 = diff(u0_current, T0, T1)
        cross_term = 2 * D0_D1_u0

        # Damping term: du_coeff * D0*u0
        D0_u0 = diff(u0_current, T0)
        damp_term = du_coeff * D0_u0 if du_sym else Integer(0)

        # Nonlinear term at O(eps^(k-1)):
        # substitute u->u0_current in f_nonlin, extract eps^(k-1) coeff
        # (since f_nonlin already has one explicit eps factor)
        nl_expr = f_nonlin.subs(eps, Integer(1)).subs(u_sym, u0_current)
        if du_sym:
            nl_expr = nl_expr.subs(du_sym, D0_u0)
        if d2u_sym:
            nl_expr = nl_expr.subs(d2u_sym, diff(u0_current, T0, 2))
        nl_term = expand(nl_expr)

        # Full forcing for O(eps^k):
        forcing_raw = -(cross_term + damp_term + nl_term)
        # Expand trig powers for clean coefficient extraction
        forcing = expand(expand(forcing_raw.rewrite(exp)).rewrite(cos))

        # Extract secular terms
        coeff_cos, coeff_sin = _secular_coefficients(forcing, T0, omega0)

        # Solvability conditions: set secular coefficients to zero
        # These are ODEs for A(T1) and B(T1)
        dA = A.diff(T1)
        dB = B.diff(T1)

        solvability_A_eq = None
        solvability_B_eq = None

        # Solve for dA from sin equation, dB from cos equation
        # Note: Derivative objects are NOT in .free_symbols — use has() instead
        if coeff_sin != 0 and coeff_sin.has(dA):
            dA_val = solve(coeff_sin, dA)
            if dA_val:
                solvability_A_eq = Eq(dA, simplify(dA_val[0]))

        if coeff_cos != 0 and coeff_cos.has(dB):
            dB_val = solve(coeff_cos, dB)
            if dB_val:
                solvability_B_eq = Eq(dB, simplify(dB_val[0]))

        # Solve the solvability ODEs symbolically with ICs.
        # Key: substitute B0_val into A's equation FIRST (decouple),
        # then solve A, then solve B (if needed).
        # For symmetric ICs (B0=0), B stays 0 if the B equation is satisfied.
        A_solved = A
        B_solved = B

        if B0_val == 0:
            # With B(0)=0, check if B=0 is a solution (it often is by symmetry)
            B_solved = Integer(0)
            # Solve A's equation with B=0 substituted (decoupled Bernoulli/linear ODE)
            if solvability_A_eq is not None:
                eq_A_decoupled = solvability_A_eq.subs(B, Integer(0))
                try:
                    A_ode_sol = dsolve(
                        eq_A_decoupled, A,
                        ics={A.subs(T1, 0): A0_val}
                    )
                    if isinstance(A_ode_sol, list):
                        A_ode_sol = A_ode_sol[0]
                    A_solved = simplify(A_ode_sol.rhs)
                except Exception:
                    pass  # leave symbolic
        else:
            # General case: try to solve both (may fail for coupled nonlinear)
            if solvability_A_eq is not None:
                try:
                    A_ode_sol = dsolve(
                        solvability_A_eq, A,
                        ics={A.subs(T1, 0): A0_val}
                    )
                    if isinstance(A_ode_sol, list):
                        A_ode_sol = A_ode_sol[0]
                    A_solved = simplify(A_ode_sol.rhs)
                except Exception:
                    pass

            if solvability_B_eq is not None:
                try:
                    B_ode_sol = dsolve(
                        solvability_B_eq.subs(A, A_solved), B,
                        ics={B.subs(T1, 0): B0_val}
                    )
                    if isinstance(B_ode_sol, list):
                        B_ode_sol = B_ode_sol[0]
                    B_solved = simplify(B_ode_sol.rhs)
                except Exception:
                    pass

        solvability_odes.append((solvability_A_eq, solvability_B_eq))

        # Remove secular terms from forcing and solve for u_k
        forcing_desec = forcing - coeff_cos*cos(omega0*T0) - coeff_sin*sin(omega0*T0)
        subs_list = []
        if solvability_A_eq is not None:
            subs_list.append((dA, solvability_A_eq.rhs))
        if solvability_B_eq is not None:
            subs_list.append((dB, solvability_B_eq.rhs))
        if subs_list:
            forcing_desec = expand(forcing_desec.subs(subs_list))

        # Expand trig BEFORE substituting A_solved (keep A, B symbolic for dsolve speed)
        # Substituting complex A_solved expressions into forcing makes dsolve very slow
        forcing_desec = expand(expand(forcing_desec.rewrite(exp)).rewrite(cos))

        # Solve for u_k with A, B still symbolic — fast dsolve
        C1, C2 = symbols('C1 C2')
        uk_T0  = Function(f'{dep}_{k}_T0')(T0)
        ode_k  = Eq(diff(uk_T0, T0, 2) + omega0**2 * uk_T0, forcing_desec)

        try:
            gen_sol  = dsolve(ode_k, uk_T0)
            gen_expr = gen_sol.rhs

            # Apply homogeneous ICs
            ic_eqs = [
                Eq(gen_expr.subs(T0, 0), 0),
                Eq(diff(gen_expr, T0).subs(T0, 0), 0),
            ]
            const_sol = solve(ic_eqs, [C1, C2])
            part_expr = gen_expr.subs(const_sol)
            # Substitute solved A, B only at the end
            part_expr = part_expr.subs(A, A_solved).subs(B, B_solved)
        except Exception:
            part_expr = Integer(0)

        # Update u0 with solved A, B for next iteration
        u0_current = u0_expr.subs(A, A_solved).subs(B, B_solved)
        A_sol_expr = A_solved
        B_sol_expr = B_solved

        h.entries.append(MultScalesOrderEntry(
            order               = k,
            pde                 = Eq(diff(uk, T0, 2) + omega0**2 * uk, forcing),
            secular_cos         = coeff_cos,
            secular_sin         = coeff_sin,
            solvability_A       = solvability_A_eq,
            solvability_B       = solvability_B_eq,
            particular_solution = part_expr,
            symbol              = uk,
        ))

    # ------------------------------------------------------------------
    # Step 5: assemble composite
    # ------------------------------------------------------------------
    h.amplitude_A      = A_sol_expr
    h.amplitude_B      = B_sol_expr
    h.solvability_odes = solvability_odes

    # Composite in (T0, T1)
    u0_part = u0_expr.subs(A, A_sol_expr).subs(B, B_sol_expr)
    composite_T = u0_part
    for k in range(1, N + 1):
        composite_T = composite_T + eps**k * h.entries[k].particular_solution
    h.composite = composite_T

    # Composite in t: T0 -> t, T1 -> eps*t
    h.composite_t = simplify(composite_T.subs([(T0, t), (T1, eps * t)]))

    h._problem = problem
    return h
