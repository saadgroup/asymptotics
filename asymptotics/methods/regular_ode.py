"""
asymptotics.methods.regular_ode
===============================
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


def _bc_value_at_order(cond_value, eps, k):
    """
    Return the eps^k coefficient of a boundary-condition value.

    If the value is plain (no eps), the standard rule applies:
      k=0 → the value itself, k>0 → 0.
    If the value contains eps (e.g. "1 + eps" or "eps**2"),
      extract the appropriate coefficient via series expansion so
      every order gets the right contribution.

    Examples
    --------
    _bc_value_at_order(1,          eps, 0) → 1
    _bc_value_at_order(1,          eps, 1) → 0
    _bc_value_at_order(1 + eps,    eps, 0) → 1
    _bc_value_at_order(1 + eps,    eps, 1) → 1
    _bc_value_at_order(eps**2,     eps, 0) → 0
    _bc_value_at_order(eps**2,     eps, 2) → 1
    """
    if eps in sympify(cond_value).free_symbols:
        return series(sympify(cond_value), eps, 0, k + 2).coeff(eps, k)
    return sympify(cond_value) if k == 0 else sympify(0)

from asymptotics.core.hierarchy  import OrderHierarchy, OrderEntry
from asymptotics.core.exceptions import (
    NoSmallParameterError,
    NoLeadingOrderSolutionError,
    NoHigherOrderSolutionError,
)
from asymptotics.gauge import parse_gauge, extract_coefficients


class ODEHierarchy:
    r"""
    Regular-perturbation hierarchy for a single ordinary differential equation.

    An ``ODEHierarchy`` is the object returned by :meth:`ODE.expand_regular`
    (and by the module-level :func:`expand_regular_ode`).  It stores the
    order-by-order results of the regular perturbation expansion together with
    the assembled expansion, and exposes the four-method result API shared by
    every hierarchy in :mod:`asymptotics` (:meth:`show`, :meth:`eval`,
    :meth:`compare_numeric`, :meth:`to_latex`).

    The underlying expansion writes the solution as a power series in the small
    parameter :math:`\varepsilon`,

    .. math::

        u(t, \varepsilon) = \sum_{k=0}^{N} u_k(t)\, \varepsilon^{k}
                          = u_0(t) + \varepsilon\, u_1(t)
                          + \varepsilon^{2}\, u_2(t) + \cdots ,

    where each :math:`u_k(t)` solves a *linear* ODE obtained by collecting the
    coefficient of :math:`\varepsilon^{k}` after substituting the ansatz into
    the original equation.  (A non-standard *gauge* sequence
    :math:`\{\delta_k(\varepsilon)\}` replaces :math:`\varepsilon^{k}`; see
    :func:`expand_regular_ode`.)

    Indexing (``sol[k]``) returns the :class:`ODEOrderEntry` for order ``k``,
    and ``len(sol)`` is the number of orders (``order + 1``).

    Attributes
    ----------
    entries : list of ODEOrderEntry
        One :class:`ODEOrderEntry` per order, 0-indexed, from order 0 up to
        and including ``order``.
    expansion : sympy.Expr
        The assembled expansion :math:`\sum_k u_k(t)\,\varepsilon^{k}`, with all
        integration constants already fixed by the initial/boundary conditions.
    small_param : sympy.Symbol
        The small parameter :math:`\varepsilon`.
    independent : sympy.Symbol
        The independent variable (e.g. :math:`t`).

    See Also
    --------
    expand_regular_ode : The function that builds this hierarchy.
    ODEOrderEntry : Per-order record accessed via ``sol[k]``.

    Examples
    --------
    A first-order IVP :math:`u' + u + \varepsilon u^2 = 0`, :math:`u(0)=1`:

    >>> import matplotlib; matplotlib.use('Agg')
    >>> from asymptotics import ODE
    >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
    ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
    >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
    >>> len(sol)                                             # doctest: +SKIP
    3
    >>> sol[0].particular_solution                           # doctest: +SKIP
    exp(-t)
    >>> sol.expansion                                        # doctest: +SKIP
    eps**2*(exp(-t) - 2*exp(-2*t) + exp(-3*t)) + eps*(-exp(-t) + exp(-2*t)) + exp(-t)
    """

    def __init__(self):
        self.entries      = []
        self.expansion    = None
        self.small_param  = None
        self.independent  = None
        self._method      = ""
        self._problem_repr = ""
        self._problem_type = ""   # "ivp" or "bvp"

    def __getitem__(self, order: int):
        """
        Return the :class:`ODEOrderEntry` for a given order.

        Parameters
        ----------
        order : int
            Perturbation order ``k``, ``0 <= k <= N``.  Standard Python
            indexing semantics apply (negative indices count from the end).

        Returns
        -------
        ODEOrderEntry
            The per-order record holding the order-``k`` ODE and its general and
            particular solutions.

        Examples
        --------
        >>> sol[0].particular_solution      # leading-order solution   # doctest: +SKIP
        >>> sol[1].equation                 # order-1 ODE               # doctest: +SKIP
        """
        return self.entries[order]

    def __len__(self):
        """
        Return the number of orders stored (``order + 1``).

        Returns
        -------
        int
            One more than the requested expansion order, since order 0 (the
            leading order) is included.

        Examples
        --------
        >>> len(eq.expand_regular(order=2))      # doctest: +SKIP
        3
        """
        return len(self.entries)


    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Compare this perturbation expansion against a numerical solution.

        Numerically integrates the original ODE with SciPy
        (:func:`scipy.integrate.solve_ivp` for an IVP, a BVP solver for a BVP)
        at the requested value of :math:`\varepsilon`, evaluates the assembled
        perturbation :attr:`expansion` on the same grid, and returns both
        together with error norms and a comparison figure.  Use this to gauge
        how well the truncated expansion tracks the true solution — the two
        curves should agree closely for small :math:`\varepsilon` and drift
        apart as it grows.

        Parameters
        ----------
        eps : float
            Value of the small parameter :math:`\varepsilon` at which to
            compare.
        params : dict, optional
            Numerical values for any extra free symbols appearing in the
            equation, as ``{name_or_symbol: value}``.
        **kwargs
            ``plot_range`` : ``[a, b]``
                Domain over which to integrate and plot.  Defaults to the
                domain implied by the problem's conditions.
            ``n_points`` : int
                Number of sample points (default 300).
            ``filename`` : str
                If given, save the figure to this path.

        Returns
        -------
        dict
            Dictionary with keys:

            ``'t'`` : ndarray
                Evaluation grid (independent-variable values).
            ``'u_pert'`` : dict
                Perturbation expansion sampled on the grid, keyed by the
                :math:`\varepsilon` value(s).
            ``'u_numerical'`` : ndarray
                The SciPy reference solution.
            ``'errors'`` : dict
                L2/Linf absolute and relative errors, keyed by
                :math:`\varepsilon`.
            ``'settings'`` : dict
                The SciPy solver, method, and tolerances used (for reproducible
                reporting).
            ``'fig'`` : matplotlib.figure.Figure
                The comparison plot.

        Notes
        -----
        This routine imports :mod:`matplotlib`.  In a headless environment or a
        script, select a non-interactive backend before calling, e.g.::

            import matplotlib; matplotlib.use('Agg')

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
        >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
        >>> res = sol.compare_numeric(eps=0.1)                  # doctest: +SKIP
        >>> sorted(res.keys())                                  # doctest: +SKIP
        ['errors', 'fig', 'settings', 't', 'u_numerical', 'u_pert']
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, problem=problem, **kwargs)


    def to_latex(self, environment='align', show_orders=False, filename=None):
        r"""
        Export this expansion as LaTeX source.

        Renders the assembled :attr:`expansion` (and, optionally, each order
        :math:`u_k` separately) as a LaTeX math block.  The small parameter is
        always typeset as ``\varepsilon`` regardless of the symbol name used in
        the problem.

        Parameters
        ----------
        environment : str, optional
            LaTeX math environment: ``'align'`` (default), ``'equation'``, or
            ``'gather'``.
        show_orders : bool, optional
            If True, list each order :math:`u_k(t)` on its own line in addition
            to the assembled expansion.  Default False.
        filename : str, optional
            If given, write the LaTeX source to this file.  The source string is
            returned in every case.

        Returns
        -------
        str
            The LaTeX source string.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
        >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
        >>> src = sol.to_latex()                                # doctest: +SKIP
        >>> print(src)                                          # doctest: +SKIP
        % Regular perturbation — ODE (IVP)
        ...
        \begin{align}
          u(t,\varepsilon) &= ... + \mathcal{O}(\varepsilon^{3})
        \end{align}
        >>> sol.to_latex(environment='equation', show_orders=True)  # doctest: +SKIP
        >>> sol.to_latex(filename="result.tex")                     # doctest: +SKIP
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)


    def eval(self, eps, at=None, params=None):
        r"""
        Evaluate the perturbation expansion numerically.

        Substitutes a concrete value of :math:`\varepsilon` into the assembled
        :attr:`expansion` and evaluates it over an array of independent-variable
        values, returning a NumPy array.  This is the numerical realisation of

        .. math::

            u(t, \varepsilon) \approx \sum_{k=0}^{N} u_k(t)\, \varepsilon^{k}.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter :math:`\varepsilon`.
        at : array-like
            Values of the independent variable :math:`t` on which to sample the
            expansion.  Required for ODE hierarchies.
        params : dict, optional
            Numerical values for any extra free symbols in the expansion.

        Returns
        -------
        numpy.ndarray or dict
            If ``eps`` is a scalar, a 1-D array of the expansion sampled at the
            points in ``at``.  If ``eps`` is a list, a dict mapping each
            :math:`\varepsilon` value to its sampled array.

        Examples
        --------
        >>> import numpy as np
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
        >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
        >>> t_vals = np.linspace(0, 20, 300)                     # doctest: +SKIP
        >>> u = sol.eval(eps=0.1, at=t_vals)          # ndarray  # doctest: +SKIP
        >>> multi = sol.eval(eps=[0.1, 0.2], at=t_vals)  # {0.1: array, 0.2: array}  # doctest: +SKIP
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

    def show(self, orders=None, mode: str = "auto") -> None:
        r"""
        Render the ODE hierarchy for inspection.

        Displays, for each order, the linear ODE that :math:`u_k(t)` satisfies
        together with its general solution (with free integration constants) and
        its particular solution (constants fixed by the initial/boundary
        conditions), followed by the assembled expansion.  Rendering adapts to
        the environment: rich LaTeX in Jupyter, plain text in a terminal.

        Parameters
        ----------
        orders : list of int, optional
            Which orders to display.  Default: all orders.
        mode : str, optional
            ``"auto"`` (default) uses LaTeX in Jupyter and plain text
            elsewhere; ``"jupyter"`` forces LaTeX via IPython; ``"text"`` forces
            plain text.

        Returns
        -------
        None
            Output is printed/displayed as a side effect.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
        >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
        >>> sol.show(mode="text")                                # doctest: +SKIP
        >>> sol.show(orders=[0, 1])                              # doctest: +SKIP
        """
        from asymptotics.display.ode_display import show_ode
        show_ode(self, orders=orders, mode=mode)


class ODEOrderEntry:
    r"""
    All information about a single order of an ODE perturbation hierarchy.

    One ``ODEOrderEntry`` records everything the solver produced at a given
    order ``k``: the linear ODE for :math:`u_k(t)`, its general solution (with
    free integration constants) and its particular solution (constants fixed by
    the problem's initial/boundary conditions), and whether secular terms were
    detected.  Instances are created by :func:`expand_regular_ode` and reached
    through ``sol[k]``.

    At order ``k`` the coefficient of :math:`\varepsilon^{k}` in the substituted
    equation gives a *linear* ODE of the schematic form

    .. math::

        \mathcal{L}\, u_k(t) = R_k\bigl(t;\, u_0, \dots, u_{k-1}\bigr),

    where :math:`\mathcal{L}` is the linear operator of the leading-order problem
    and the right-hand side :math:`R_k` depends only on the lower-order
    solutions already known.  Solving this ODE gives a
    :attr:`general_solution` containing free constants :math:`C_1, C_2, \dots`;
    imposing the order-``k`` conditions fixes them and yields the
    :attr:`particular_solution`.

    Parameters
    ----------
    order : int
        The perturbation order ``k`` this entry describes.
    ode : sympy.Eq
        The linear ODE that :math:`u_k(t)` satisfies, written as
        ``Eq(<expr>, 0)``.
    general_solution : sympy.Expr
        Solution of ``ode`` with free integration constants (``C1``, ``C2``,
        ...) still present.
    particular_solution : sympy.Expr
        The general solution after the integration constants have been fixed by
        the initial/boundary conditions.
    constants : dict
        Mapping ``{C1: value, ...}`` giving the constant values determined from
        the conditions.
    symbol : sympy.Function
        The unknown function :math:`u_k(t)` for this order.
    secular : bool, optional
        Whether secular (unbounded, e.g. :math:`t\sin t`) terms were detected in
        the particular solution.  Default False.

    Attributes
    ----------
    order : int
        The perturbation order ``k``.
    ode : sympy.Eq
        The order-``k`` linear ODE, ``Eq(<expr>, 0)``.
    general_solution : sympy.Expr
        Order-``k`` solution with free integration constants.
    particular_solution : sympy.Expr
        Order-``k`` solution with constants fixed by ICs/BCs.  Also available as
        ``solution`` for display compatibility.
    secular : bool
        True if secular terms were detected at this order.  Regular perturbation
        does not remove them — that is the domain of the Lindstedt–Poincaré and
        multiple-scales methods.
    constants : dict
        The determined integration constants ``{C1: value, ...}``.
    symbol : sympy.Function
        The unknown function :math:`u_k(t)`.
    equation : sympy.Eq
        Uniform per-order alias for :attr:`ode`; ``sol[k].equation`` is the
        order-``k`` equation across every hierarchy type in :mod:`asymptotics`.

    Notes
    -----
    The difference between :attr:`general_solution` and
    :attr:`particular_solution` is exactly the integration constants: the
    general solution is the full family that solves the order-``k`` ODE, while
    the particular solution is the single member of that family selected by the
    initial conditions (IVP) or boundary conditions (BVP).

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
    ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
    >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
    >>> sol[1].equation                                      # doctest: +SKIP
    Eq(u_1(t) - sinh(2*t) + cosh(2*t) + Derivative(u_1(t), t), 0)
    >>> sol[1].general_solution                              # free constant C1  # doctest: +SKIP
    (C1 + exp(-t))*exp(-t)
    >>> sol[1].particular_solution                           # C1 fixed by u(0)=0  # doctest: +SKIP
    -exp(-t) + exp(-2*t)
    >>> sol[1].secular                                       # doctest: +SKIP
    False
    """

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
        self.equation            = ode   # uniform per-order API: order-k equation



def _apply_limit_condition(cond, gen_expr, t_sym, dep_name, deriv_syms, order_k, eps=None):
    """
    Apply a limit condition to the general solution at order k.

    For each free constant C_i, check if it causes the limit expression
    to diverge — if so, set C_i = 0.
    """
    from sympy import (Symbol, diff, limit, Eq, oo, zoo, nan,
                       sympify, series)
    from asymptotics.core.problem import _preprocess_ode_string

    var_sym   = Symbol(cond.var_str)
    lim_point = cond.point
    lim_value = _bc_value_at_order(cond.value, eps if eps is not None else sympify(0), order_k)

    # Preprocess: replace u\'\', u\' with d2u, du in expr_str
    expr_proc = _preprocess_ode_string(cond.expr_str, dep_name)

    # Build namespace: map dep and derivatives to gen_expr
    ns = {dep_name: gen_expr, cond.var_str: var_sym}
    for k_d, dsym in deriv_syms.items():
        dname = "d{}{}".format(k_d if k_d > 1 else "", dep_name)
        expr_val = diff(gen_expr, t_sym, k_d)
        if str(var_sym) != str(t_sym):
            expr_val = expr_val.subs(t_sym, var_sym)
        ns[dname] = expr_val

    if str(var_sym) == str(t_sym):
        pass  # already in t_sym
    else:
        if dep_name in ns:
            ns[dep_name] = gen_expr.subs(t_sym, var_sym)

    try:
        full_expr = sympify(expr_proc, locals=ns, convert_xor=False)
    except Exception as e:
        raise RuntimeError(
            "\n\n  Could not parse limit expression: \"{}\"\n  Error: {}\n".format(
                cond.expr_str, e)
        )

    # Free constants in this general solution
    free_consts = sorted(
        [s for s in gen_expr.free_symbols
         if str(s).startswith("C") and str(s)[1:].isdigit()],
        key=lambda s: int(str(s)[1:])
    )

    if not free_consts:
        try:
            lv = limit(full_expr, var_sym, lim_point, "+")
            if lv.is_finite:
                return Eq(lv, lim_value) if lv != lim_value else True
        except Exception:
            pass
        return True

    # Try direct limit
    try:
        lv = limit(full_expr, var_sym, lim_point, "+")
        if lv.is_finite and not lv.has(oo, zoo, nan):
            return Eq(lv, lim_value)
    except Exception:
        pass

    # Identify which constants cause divergence
    for C in free_consts:
        test_subs = {c: 0 for c in free_consts if c != C}
        test_expr = full_expr.subs(test_subs)
        try:
            lv_test = limit(test_expr, var_sym, lim_point, "+")
            if not lv_test.is_finite or lv_test.has(oo, zoo, nan):
                return Eq(C, 0)
        except Exception:
            try:
                s = series(test_expr, var_sym, lim_point, 1)
                lv_s = limit(s.removeO(), var_sym, lim_point, "+")
                if not lv_s.is_finite:
                    return Eq(C, 0)
            except Exception:
                pass

    return True  # condition automatically satisfied

def expand_regular_ode(problem, order: int = 2, gauge=None) -> ODEHierarchy:
    r"""
    Apply regular perturbation theory to an ordinary differential equation.

    Constructs the regular (Poincaré) perturbation expansion of an ODE
    :math:`F(u, u', u'', \dots, t, \varepsilon) = 0` subject to its
    initial/boundary conditions, and returns the full order-by-order hierarchy.
    This is the engine behind :meth:`asymptotics.ODE.expand_regular`; call that
    method rather than this function directly in normal use.

    The solution is sought as a power series in the small parameter,

    .. math::

        u(t, \varepsilon) = \sum_{k=0}^{N} u_k(t)\, \delta_k(\varepsilon),

    where by default :math:`\delta_k(\varepsilon) = \varepsilon^{k}` (the
    standard gauge :math:`\{1, \varepsilon, \varepsilon^2, \dots\}`).
    Substituting this ansatz into :math:`F` and collecting the coefficient of
    each gauge function yields, at order ``k``, a *linear* ODE for
    :math:`u_k(t)` whose forcing depends only on the already-known lower-order
    solutions:

    .. math::

        \mathcal{L}\, u_k(t) = R_k\bigl(t;\, u_0, \dots, u_{k-1}\bigr).

    Each order is solved with :func:`sympy.dsolve` and its free integration
    constants are fixed by imposing the order-``k`` conditions (the leading
    order absorbs the inhomogeneous condition values; higher orders satisfy the
    homogeneous versions, except where a condition value itself carries an
    :math:`\varepsilon`-dependence, which is distributed across orders).
    Secular terms are detected and flagged but never removed — that is the
    province of the Lindstedt–Poincaré and multiple-scales methods.

    Parameters
    ----------
    problem : asymptotics.ODE
        The ODE problem to expand.  Supplies the equation, small parameter,
        independent/dependent variables, and the initial/boundary conditions.
    order : int, optional
        Highest order ``N`` to compute.  The resulting hierarchy has
        ``order + 1`` entries (orders 0 through ``N``).  Default 2.
    gauge : str, list of str, or None, optional
        Non-standard gauge (asymptotic) sequence overriding the default
        :math:`\{\varepsilon^{k}\}`.  A single string such as ``"sqrt(eps)"``
        seeds a geometric sequence :math:`\{\delta^{k}\}`; a list of strings
        such as ``["1", "log(eps)", "log(eps)**2"]`` specifies the sequence
        term by term.  Parsed by :func:`asymptotics.gauge.parse_gauge`.

    Returns
    -------
    ODEHierarchy
        The assembled hierarchy, indexable by order (``sol[k]``) and exposing
        the ``show`` / ``eval`` / ``compare_numeric`` / ``to_latex`` API.

    Raises
    ------
    NoSmallParameterError
        If the small parameter :math:`\varepsilon` does not appear in the
        equation, so there is nothing to expand in.
    NoLeadingOrderSolutionError
        If :func:`sympy.dsolve` cannot solve the leading-order (order 0) ODE.
    NoHigherOrderSolutionError
        If a higher-order ODE cannot be solved, or its integration constants
        cannot be determined from the conditions.

    See Also
    --------
    ODEHierarchy : The returned object.
    asymptotics.ODE.expand_regular : The public entry point that calls this.

    Examples
    --------
    First-order IVP :math:`u' + u + \varepsilon u^2 = 0`, :math:`u(0)=1`:

    >>> from asymptotics import ODE
    >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
    ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
    >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
    >>> sol[0].particular_solution                           # doctest: +SKIP
    exp(-t)
    >>> sol.expansion                                        # doctest: +SKIP
    eps**2*(exp(-t) - 2*exp(-2*t) + exp(-3*t)) + eps*(-exp(-t) + exp(-2*t)) + exp(-t)

    A linear BVP :math:`u'' + \varepsilon u = 0`, :math:`u(0)=0`, :math:`u(1)=1`,
    where the higher orders satisfy homogeneous boundary conditions:

    >>> eq = ODE("u'' + eps*u", dependent="u", small_param="eps",
    ...          independent="t", conditions=["u(0) = 0", "u(1) = 1"])  # doctest: +SKIP
    >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
    >>> sol[0].particular_solution                           # doctest: +SKIP
    t
    >>> sol[1].particular_solution                           # doctest: +SKIP
    -t**3/6 + t/6
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
    # Step 1b: build gauge sequence
    # ------------------------------------------------------------------
    gauge_seq = parse_gauge(gauge, N, eps)

    # ------------------------------------------------------------------
    # Step 2: build ansatz
    #   u(t,ε) = u_0(t)·δ_0(ε) + u_1(t)·δ_1(ε) + ...
    # ------------------------------------------------------------------
    u_ans = sum(gauge_seq[k] * u_funcs[k] for k in range(N + 1))

    # ------------------------------------------------------------------
    # Step 3: substitute ansatz into F
    # Replace derivative symbols with actual derivatives of u_ans
    # ------------------------------------------------------------------
    f_sub = f.subs(problem.dependent, u_ans)  # substitute u -> u_ans (symbol)

    # Also substitute derivative symbols
    for k_deriv, dsym in deriv_syms.items():
        f_sub = f_sub.subs(dsym, diff(u_ans, t, k_deriv))

    f_expanded = expand(f_sub)

    # ------------------------------------------------------------------
    # Step 4: collect coefficients by gauge function
    #   Use sequential limit extraction (works for power-law AND log gauges)
    # ------------------------------------------------------------------
    coeff_list = extract_coefficients(f_expanded, gauge_seq, eps)
    coeffs     = {k: coeff_list[k] for k in range(N + 1)}

    # ------------------------------------------------------------------
    # Step 5: solve order by order
    # ------------------------------------------------------------------
    h = ODEHierarchy()
    h.small_param   = eps
    h.independent   = t
    h._method       = f"Regular perturbation — ODE ({'IVP' if ptype == 'ivp' else 'BVP'})"
    h._problem_repr = f"F({dep}, {dep}', t, {eps}) = 0"
    h._problem_type = ptype
    h._gauge        = gauge_seq     # stored for display

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

        # Build order-k conditions.
        # If a BC value contains eps (e.g. "u(0) = 1 + eps"), extract the
        # eps^k coefficient so each order gets the correct contribution.
        from asymptotics.core.conditions import LimitCondition as _LimitCond
        cond_equations = []
        for cond in conds:
            if isinstance(cond, _LimitCond):
                # Limit condition: lim(expr, var, point) = value
                _eq = _apply_limit_condition(
                    cond, gen_expr, t, dep,
                    problem._deriv_syms, k, eps=eps
                )
                if _eq is not None and _eq is not True:
                    cond_equations.append(_eq)
            else:
                pt  = cond.point
                val = _bc_value_at_order(cond.value, eps, k)
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
    # Step 7: expansion expansion
    # ------------------------------------------------------------------
    h.expansion = Add(*[known_solutions[u_funcs[k]] * gauge_seq[k] for k in range(N + 1)])

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
