"""
asymptotics.methods.multiple_scales
===================================
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

The expansion solution is then:
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
    r"""
    Result of a method-of-multiple-scales expansion.

    A container holding the order-by-order results produced by
    :func:`expand_multiple_scales`. It bundles the leading-order amplitude
    functions :math:`A(T_1), B(T_1)`, the solvability (amplitude) equations,
    the per-order solutions, and the assembled expansion, together with the
    four-method presentation API (:meth:`show`, :meth:`eval`,
    :meth:`compare_numeric`, :meth:`to_latex`) shared by every hierarchy in
    the toolkit.

    The method of multiple scales treats the fast oscillation and the slow
    amplitude/phase modulation of a weakly perturbed oscillator as functions
    of independent time scales,

    .. math::

        T_0 = t \quad\text{(fast)}, \qquad
        T_1 = \varepsilon\, t \quad\text{(slow)},

    and writes :math:`u = u_0(T_0, T_1) + \varepsilon\,u_1(T_0, T_1) +
    \cdots`. The leading order is
    :math:`u_0 = A(T_1)\cos(\omega_0 T_0) + B(T_1)\sin(\omega_0 T_0)`, and the
    requirement that :math:`u_1` contain no secular (resonant) forcing yields
    *solvability conditions* — ODEs in the slow time for the amplitudes
    :math:`A(T_1)` and :math:`B(T_1)`.

    Indexing and length: ``sol[k]`` returns the
    :class:`MultScalesOrderEntry` for order ``k`` and ``len(sol)`` is
    ``order + 1``.

    Attributes
    ----------
    entries : list of MultScalesOrderEntry
        One entry per order, indexed ``0 .. order``. Also reachable via
        ``sol[k]``.
    expansion : sympy.Expr
        The assembled expansion :math:`u(T_0, T_1, \varepsilon)` in the two
        time scales.
    expansion_t : sympy.Expr
        The same expansion in physical time, obtained via
        :math:`T_0 = t,\ T_1 = \varepsilon t`. For the damped oscillator this
        is :math:`e^{-\varepsilon t/2}\cos t`.
    amplitude_A : sympy.Expr
        The solved slow-time amplitude :math:`A(T_1)` (or the symbolic
        ``A(T_1)`` if SymPy could not integrate the solvability ODE).
    amplitude_B : sympy.Expr
        The solved slow-time amplitude :math:`B(T_1)`.
    solvability_odes : list of tuple
        One ``(A_eq, B_eq)`` pair per positive order — the amplitude ODEs
        derived from the secularity conditions.
    small_param : sympy.Symbol
        The perturbation parameter :math:`\varepsilon`.
    independent : sympy.Symbol
        The physical independent variable :math:`t`.
    T0 : sympy.Symbol
        The fast-time symbol :math:`T_0 = t`.
    T1 : sympy.Symbol
        The slow-time symbol :math:`T_1 = \varepsilon t`.
    omega_0 : sympy.Expr
        The unperturbed frequency :math:`\omega_0`, auto-detected from the
        unperturbed equation.

    See Also
    --------
    expand_multiple_scales : Function that builds this hierarchy.
    MultScalesOrderEntry : Per-order record returned by ``sol[k]``.
    LindstedtHierarchy : Result of the related Lindstedt–Poincaré method.

    Examples
    --------
    Damped oscillator :math:`u'' + u + \varepsilon u' = 0`,
    :math:`u(0)=1,\ u'(0)=0` (exact envelope :math:`e^{-\varepsilon t/2}`):

    >>> from asymptotics import ODE
    >>> eq = ODE("u'' + u + eps*u'",
    ...          dependent='u', small_param='eps', independent='t',
    ...          conditions=["u(0) = 1", "u'(0) = 0"])
    >>> sol = eq.expand_multiple_scales(order=1)
    >>> sol.omega_0
    1
    >>> sol.amplitude_A
    exp(-T_1/2)
    >>> sol.amplitude_B
    0
    >>> sol.expansion_t
    exp(-eps*t/2)*cos(t)
    """

    def __init__(self):
        self.entries          = []
        self.expansion        = None
        self.expansion_t      = None
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
        """Return the :class:`MultScalesOrderEntry` for the given ``order``.

        Enables ``sol[k]`` access to the order-``k`` record. ``order`` is a
        plain integer index into :attr:`entries` (order ``0`` is the
        leading-order term); negative indices count from the end.

        Parameters
        ----------
        order : int
            Perturbation order, ``0`` through ``len(sol) - 1``.

        Returns
        -------
        MultScalesOrderEntry
            The record for that order.
        """
        return self.entries[order]

    def __len__(self):
        """Number of orders in the hierarchy.

        Returns
        -------
        int
            One more than the requested expansion order (``2`` for
            ``order=1``).
        """
        return len(self.entries)


    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Verify the expansion against a SciPy numerical solution and plot.

        Substitutes ``eps`` into :attr:`expansion_t`, integrates the original
        ODE numerically with :func:`scipy.integrate.solve_ivp`, and builds a
        comparison figure plus error metrics. This confirms that the
        slow-scale amplitude modulation captures the true envelope of the
        solution.

        Parameters
        ----------
        eps : float
            Value of the small parameter :math:`\varepsilon` to use.
        params : dict, optional
            Numerical values for any extra symbolic parameters in the problem.
        **kwargs
            plot_range : [a, b]
                Domain over which to sample and plot.
            n_points : int
                Number of plot points (default 300).

        Returns
        -------
        dict
            Keys include:

            ``'t'`` : ndarray
                Evaluation points.
            ``'u_pert'`` : ndarray
                The multiple-scales expansion sampled on ``t``.
            ``'u_numerical'`` : ndarray
                The SciPy reference solution.
            ``'fig'`` : matplotlib.figure.Figure
                Comparison plot.
            ``'errors'`` : dict
                L2/Linf absolute and relative errors keyed by ``eps``.
            ``'settings'`` : dict
                Solver, method, and tolerances used for reproducibility.

        Notes
        -----
        The plot is rendered with matplotlib. In a headless environment set a
        non-interactive backend (``import matplotlib; matplotlib.use('Agg')``)
        before calling.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u'",
        ...          dependent='u', small_param='eps', independent='t',
        ...          conditions=["u(0) = 1", "u'(0) = 0"])
        >>> sol = eq.expand_multiple_scales(order=1)
        >>> result = sol.compare_numeric(eps=0.1)
        >>> sorted(result.keys())
        ['errors', 'fig', 'settings', 't', 'u_numerical', 'u_pert']
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, problem=problem, **kwargs)


    def to_latex(self, environment='align', show_orders=False, filename=None):
        r"""
        Export the multiple-scales expansion as LaTeX source.

        Renders the amplitude equations and the assembled solution (in the two
        time scales and in physical time) as a LaTeX string, optionally
        written to a file. The small parameter is always typeset as
        ``\varepsilon``.

        Parameters
        ----------
        environment : str, optional
            LaTeX math environment: ``'align'`` (default), ``'equation'``, or
            ``'gather'``.
        show_orders : bool, optional
            If True, include each order :math:`u_k` separately. Default False.
        filename : str, optional
            If given, write the source to this file. Otherwise the string is
            only returned.

        Returns
        -------
        str
            The LaTeX source string.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u'",
        ...          dependent='u', small_param='eps', independent='t',
        ...          conditions=["u(0) = 1", "u'(0) = 0"])
        >>> sol = eq.expand_multiple_scales(order=1)
        >>> latex = sol.to_latex()
        >>> isinstance(latex, str)
        True
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)


    def eval(self, eps, at=None, params=None):
        r"""
        Evaluate the uniformly valid expansion numerically.

        Substitutes ``eps`` into :attr:`expansion_t` (the solution in physical
        time, with :math:`T_0 = t,\ T_1 = \varepsilon t` already applied) and
        samples it on the grid ``at``.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter :math:`\varepsilon`.
        at : array-like
            Values of the physical independent variable :math:`t` at which to
            sample. Required for this ODE hierarchy.
        params : dict, optional
            Numerical values for any additional symbolic parameters.

        Returns
        -------
        numpy.ndarray or dict
            An ``ndarray`` of samples if ``eps`` is a scalar, or a dict
            ``{eps: ndarray}`` if ``eps`` is a list.

        Examples
        --------
        >>> import numpy as np
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u'",
        ...          dependent='u', small_param='eps', independent='t',
        ...          conditions=["u(0) = 1", "u'(0) = 0"])
        >>> sol = eq.expand_multiple_scales(order=1)
        >>> u = sol.eval(eps=0.1, at=np.array([0.0, 1.0]))
        >>> float(u[0])
        1.0
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

    def show(self, orders=None, mode: str = "auto") -> None:
        r"""
        Pretty-print the hierarchy: amplitude equations and per-order solutions.

        Renders the slow/fast time scales, the solvability (amplitude) ODEs
        for :math:`A(T_1), B(T_1)`, and the solutions :math:`u_k(T_0, T_1)`.
        Output is typeset as LaTeX in a Jupyter notebook and as plain text in
        a terminal.

        Parameters
        ----------
        orders : int or iterable of int, optional
            Restrict the display to these orders. By default every order is
            shown.
        mode : {'auto', 'latex', 'text'}, optional
            Force a rendering mode. ``'auto'`` (default) chooses LaTeX in a
            notebook and plain text otherwise.

        Returns
        -------
        None
            This method prints/displays and returns nothing.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u'",
        ...          dependent='u', small_param='eps', independent='t',
        ...          conditions=["u(0) = 1", "u'(0) = 0"])
        >>> sol = eq.expand_multiple_scales(order=1)
        >>> sol.show(mode='text')          # doctest: +SKIP
        """
        from asymptotics.display.multiple_scales_display import show_multiple_scales
        show_multiple_scales(self, orders=orders, mode=mode)


class MultScalesOrderEntry:
    r"""
    One order of a multiple-scales hierarchy.

    Records everything computed at a single order :math:`\varepsilon^k`: the
    governing equation for :math:`u_k(T_0, T_1)`, the resonant (secular)
    coefficients, the solvability (amplitude) ODEs derived from removing them,
    and the resulting solution. Returned by ``sol[k]`` for a
    :class:`MultScalesHierarchy`.

    Because the unknown depends on both the fast time :math:`T_0` and the slow
    time :math:`T_1`, the order-:math:`k` equation is technically a PDE,

    .. math::

        \partial_{T_0}^2 u_k + \omega_0^2\, u_k = F_k(T_0, T_1),

    stored in :attr:`pde`. The forcing :math:`F_k` contains terms
    proportional to :math:`\cos(\omega_0 T_0)` and :math:`\sin(\omega_0 T_0)`
    whose coefficients (:attr:`secular_cos`, :attr:`secular_sin`) would drive
    secular growth. Setting them to zero gives the **solvability conditions** —
    ODEs in :math:`T_1` for the amplitudes :math:`A(T_1)` and :math:`B(T_1)`
    (:attr:`solvability_A`, :attr:`solvability_B`).

    Parameters
    ----------
    order : int
        The perturbation order :math:`k`.
    pde : sympy.Eq
        The order-:math:`k` PDE for :math:`u_k(T_0, T_1)`.
    secular_cos : sympy.Expr or None
        Coefficient of :math:`\cos(\omega_0 T_0)` in the forcing (``None`` at
        order 0).
    secular_sin : sympy.Expr or None
        Coefficient of :math:`\sin(\omega_0 T_0)` in the forcing (``None`` at
        order 0).
    solvability_A : sympy.Eq or None
        The amplitude ODE for :math:`A(T_1)` obtained from the resonant sine
        coefficient (``None`` if none was found).
    solvability_B : sympy.Eq or None
        The amplitude ODE for :math:`B(T_1)` obtained from the resonant cosine
        coefficient (``None`` if none was found).
    particular_solution : sympy.Expr
        The order-:math:`k` particular solution with homogeneous initial
        conditions applied. Also available as :attr:`solution`.
    symbol : sympy.Function
        The unknown :math:`u_k(T_0, T_1)`.

    Attributes
    ----------
    order : int
        Perturbation order :math:`k`.
    pde : sympy.Eq
        The order-:math:`k` governing PDE in :math:`(T_0, T_1)`. Exposed also
        as :attr:`ode` and :attr:`equation` so the per-order interface matches
        the other (ODE) hierarchy types.
    secular_cos, secular_sin : sympy.Expr or None
        The resonant coefficients whose vanishing gives the solvability
        conditions. For the Duffing problem
        :math:`u''+u+\varepsilon u^3=0` at order 1,
        ``secular_sin`` is
        :math:`2A'(T_1) - \tfrac34 A^2 B - \tfrac34 B^3`.
    solvability_A : sympy.Eq or None
        Amplitude ODE :math:`A'(T_1) = \ldots`. For the damped oscillator
        this is :math:`A'(T_1) = -A/2`.
    solvability_B : sympy.Eq or None
        Amplitude ODE :math:`B'(T_1) = \ldots`.
    particular_solution : sympy.Expr
        Order-:math:`k` solution after applying homogeneous initial
        conditions.
    solution : sympy.Expr
        Alias of :attr:`particular_solution`.

    See Also
    --------
    MultScalesHierarchy : The container whose ``sol[k]`` yields these entries.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("u'' + u + eps*u'",
    ...          dependent='u', small_param='eps', independent='t',
    ...          conditions=["u(0) = 1", "u'(0) = 0"])
    >>> sol = eq.expand_multiple_scales(order=1)
    >>> entry = sol[1]
    >>> entry.solvability_A
    Eq(Derivative(A(T_1), T_1), -A(T_1)/2)
    >>> entry.secular
    True
    >>> entry.ode is entry.pde
    True
    """

    def __init__(self, order, pde, secular_cos, secular_sin,
                 solvability_A, solvability_B,
                 particular_solution, symbol):
        self.order              = order
        self.pde                = pde                # PDE for u_k (T0, T1)
        self.secular_cos        = secular_cos        # coeff of cos(omega0*T0)
        self.secular_sin        = secular_sin        # coeff of sin(omega0*T0)
        self.solvability_A      = solvability_A      # Eq(dA/dT1, ...)
        self.solvability_B      = solvability_B      # Eq(dB/dT1, ...)
        self.particular_solution = particular_solution
        self.symbol             = symbol
        self.solution           = particular_solution  # alias

    # ------------------------------------------------------------------
    # Uniform API aliases — keep the public interface consistent with all
    # other hierarchy types even though the equation is technically a PDE.
    # ------------------------------------------------------------------

    @property
    def ode(self):
        """Alias for ``pde``: the equation at this order in (T0, T1)-space.
        Exposed as ``ode`` to match the uniform ``sol[k].ode`` interface."""
        return self.pde

    @property
    def equation(self):
        """Uniform per-order API alias: the order-k governing equation."""
        return self.pde

    @property
    def secular(self):
        """True if resonant (secular) terms were present at this order.
        Computed from ``secular_cos`` and ``secular_sin``."""
        from sympy import S
        return (self.secular_cos != S.Zero or self.secular_sin != S.Zero)


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
    r"""
    Apply the method of multiple scales to a nonlinear oscillator ODE.

    Solves a weakly perturbed second-order oscillator of the form

    .. math::

        u'' + \omega_0^2\, u + \varepsilon\, f(u, u', t) = 0

    by introducing independent fast and slow time scales,

    .. math::

        T_0 = t, \qquad T_1 = \varepsilon\, t,

    and expanding :math:`u = u_0(T_0, T_1) + \varepsilon\,u_1(T_0, T_1) +
    \cdots`. The leading-order solution is

    .. math::

        u_0 = A(T_1)\cos(\omega_0 T_0) + B(T_1)\sin(\omega_0 T_0),

    with amplitudes that vary on the slow scale. At order :math:`\varepsilon`
    the demand that :math:`u_1` be free of secular (resonant) forcing yields
    the **solvability conditions** — ODEs in :math:`T_1` for :math:`A` and
    :math:`B` — which are integrated with the initial data
    :math:`A(0) = u(0)`, :math:`B(0) = u'(0)/\omega_0`. Unlike
    Lindstedt–Poincaré, this method handles damping and other non-conservative
    perturbations (e.g. :math:`u'' + u + \varepsilon u' = 0`, whose amplitude
    obeys :math:`A'(T_1) = -A/2`, reproducing the decaying envelope
    :math:`e^{-\varepsilon t/2}`).

    The leading frequency :math:`\omega_0` is auto-detected from the
    unperturbed equation as :math:`\sqrt{c_u / c_{u''}}`.

    Parameters
    ----------
    problem : ODE
        A second-order oscillator containing the small parameter,
        :math:`u'' + \omega_0^2 u + \varepsilon f(u, u', t) = 0`.
    order : int, optional
        Number of :math:`\varepsilon` corrections (default 1). The returned
        hierarchy holds orders ``0`` through ``order``.

    Returns
    -------
    MultScalesHierarchy
        The order-by-order result, including the solved amplitudes
        :attr:`~MultScalesHierarchy.amplitude_A` /
        :attr:`~MultScalesHierarchy.amplitude_B` and the assembled
        physical-time solution :attr:`~MultScalesHierarchy.expansion_t`.

    Raises
    ------
    NoSmallParameterError
        If the small parameter does not appear in the equation.
    ValueError
        If the ODE is not second order, or if the unperturbed frequency
        squared is non-positive (non-oscillatory).

    Notes
    -----
    Called through the convenience method
    ``ODE.expand_multiple_scales(order)``.

    For a nonlinear conservative problem such as the Duffing oscillator the
    coupled amplitude ODEs need not have a closed-form SymPy solution; in that
    case :attr:`~MultScalesHierarchy.amplitude_A` /
    :attr:`~MultScalesHierarchy.amplitude_B` remain symbolic while the
    solvability conditions themselves are still reported per order.

    Examples
    --------
    Damped linear oscillator (closed-form slow amplitude):

    >>> from asymptotics import ODE
    >>> eq = ODE("u'' + u + eps*u'",
    ...          dependent='u', small_param='eps', independent='t',
    ...          conditions=["u(0) = 1", "u'(0) = 0"])
    >>> sol = eq.expand_multiple_scales(order=1)
    >>> sol.entries[1].solvability_A          # solvability condition on A(T_1)
    Eq(Derivative(A(T_1), T_1), -A(T_1)/2)
    >>> sol.amplitude_A
    exp(-T_1/2)
    >>> sol.expansion_t
    exp(-eps*t/2)*cos(t)
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
    # Step 5: assemble expansion
    # ------------------------------------------------------------------
    h.amplitude_A      = A_sol_expr
    h.amplitude_B      = B_sol_expr
    h.solvability_odes = solvability_odes

    # Expansion in (T0, T1)
    u0_part = u0_expr.subs(A, A_sol_expr).subs(B, B_sol_expr)
    expansion_T = u0_part
    for k in range(1, N + 1):
        expansion_T = expansion_T + eps**k * h.entries[k].particular_solution
    h.expansion = expansion_T

    # Expansion in t: T0 -> t, T1 -> eps*t
    h.expansion_t = simplify(expansion_T.subs([(T0, t), (T1, eps * t)]))

    h._problem = problem
    return h
