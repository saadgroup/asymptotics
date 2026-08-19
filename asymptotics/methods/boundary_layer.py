r"""
asymptotics.methods.boundary_layer
==================================
Matched asymptotic expansions (boundary-layer theory) for linear singularly
perturbed boundary-value problems.

This module handles second-order two-point BVPs in which a small parameter
:math:`\varepsilon \ll 1` multiplies the highest derivative,

.. math::

    \varepsilon\, u''(x) + p(x)\, u'(x) + q(x)\, u(x) = f(x),
    \qquad u(a) = \alpha, \quad u(b) = \beta .

Because :math:`\varepsilon` multiplies :math:`u''`, setting
:math:`\varepsilon = 0` drops the order of the equation from two to one: the
*reduced* (outer) problem cannot satisfy both boundary conditions. The
solution therefore develops a thin **boundary layer** of width
:math:`\mathcal{O}(\varepsilon)` at one end, across which :math:`u` changes
rapidly to pick up the boundary condition the outer solution had to abandon.

Method (leading order)
----------------------
1. **Layer location** — inferred from the sign of :math:`p` at the endpoints:

   - :math:`p(a) > 0` → layer at the left boundary :math:`x = a`;
   - :math:`p(b) < 0` → layer at the right boundary :math:`x = b`.

2. **Outer solution** :math:`u_\mathrm{out}(x)` — solve the reduced equation
   :math:`p\,u' + q\,u = f` (set :math:`\varepsilon = 0`) subject to the
   boundary condition at the *far* boundary (the one away from the layer).

3. **Inner solution** :math:`U(\xi)` — rescale with the stretched
   boundary-layer coordinate

   .. math::

       \xi = \frac{x - a}{\varepsilon} \quad\text{(left layer)}, \qquad
       \xi = \frac{b - x}{\varepsilon} \quad\text{(right layer)},

   giving the leading-order inner equation :math:`U'' + p_\ell\, U' = 0`
   (with :math:`p_\ell = p` evaluated at the layer), solved with the boundary
   condition at the *near* boundary.

4. **Matching** — the inner and outer descriptions must agree in the overlap
   region:

   .. math::

       \lim_{\xi \to \infty} U(\xi) = \lim_{x \to x_\ell} u_\mathrm{out}(x) .

5. **Composite (uniform) expansion** — add the two and subtract the common
   overlap value so neither region is double-counted:

   .. math::

       u_\mathrm{comp}(x) = u_\mathrm{out}(x) + U(\xi) - u_\mathrm{match},
       \qquad \xi = \xi(x).

Every intermediate object (ODEs, outer/inner solutions, matching constant,
composite) is stored on the returned :class:`BoundaryLayerHierarchy` and is a
live SymPy expression; see :attr:`BoundaryLayerHierarchy.components`.

Only leading order (``order=0``) is currently implemented; interior layers
(where :math:`p` changes sign inside :math:`(a, b)`) are not supported.
"""

from __future__ import annotations
from sympy import (
    Symbol, Function, symbols, Eq, dsolve, solve,
    expand, simplify, limit, oo, Integer, diff,
    sympify
)


# ---------------------------------------------------------------------------
# Hierarchy container
# ---------------------------------------------------------------------------

class BoundaryLayerHierarchy:
    r"""
    Result of a boundary-layer (matched asymptotic) expansion.

    Returned by :func:`expand_boundary_layer` and by
    :meth:`asymptotics.ODE.expand_boundary_layer`. It bundles every stage of
    the matched-asymptotics construction — outer solution, inner
    (boundary-layer) solution, matching constant, and the uniform composite —
    as inspectable SymPy expressions, and exposes the shared hierarchy API
    (:meth:`show`, :meth:`eval`, :meth:`to_latex`, :meth:`compare_numeric`).

    The central result is the additive **composite expansion**

    .. math::

        u_\mathrm{comp}(x)
            = u_\mathrm{out}(x) + U\!\big(\xi(x)\big) - u_\mathrm{match},

    stored on :attr:`expansion`, where :math:`\xi = (x-a)/\varepsilon` for a
    left layer or :math:`\xi = (b-x)/\varepsilon` for a right layer.

    Attributes
    ----------
    outer : sympy.Expr
        Leading-order outer solution :math:`u_\mathrm{out}(x)`, satisfying the
        reduced ODE and the far boundary condition.
    inner : sympy.Expr
        Leading-order inner (boundary-layer) solution :math:`U(\xi)`, written
        in the stretched coordinate :math:`\xi`.
    inner_xi : sympy.Expr
        The inner solution re-expressed in the original variable, i.e.
        :math:`U\!\big(\xi(x)\big)` with :math:`\xi` substituted back.
    match : sympy.Expr
        The matching constant :math:`u_\mathrm{match}` — the common value of
        the inner (:math:`\xi \to \infty`) and outer (at the layer) limits;
        subtracted in the composite to remove the overlap.
    expansion : sympy.Expr
        The uniform composite expansion :math:`u_\mathrm{comp}(x)`.
    layer_location : str
        Human-readable layer position, e.g. ``'x = 0'`` or ``'x = 1'``.
    layer_var : sympy.Symbol
        The stretched boundary-layer coordinate :math:`\xi`.
    outer_ode, inner_ode : sympy.Eq
        The reduced (outer) ODE and the leading-order inner ODE that were
        solved.
    outer_bc, inner_bc : str
        Which boundary condition the outer / inner solution satisfies.
    small_param : sympy.Symbol
        The small parameter :math:`\varepsilon`.
    independent : sympy.Symbol
        The independent variable :math:`x`.
    p_expr, q_expr, f_expr : sympy.Expr
        The coefficient functions :math:`p(x)`, :math:`q(x)` and right-hand
        side :math:`f(x)` extracted from the equation.

    See Also
    --------
    components : the same substeps collected into a single dict.
    expand_boundary_layer : the function that builds this object.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("eps*u'' + u' + u", dependent='u', small_param='eps',
    ...          independent='x', conditions=['u(0) = 0', 'u(1) = 1'])
    >>> sol = eq.expand_boundary_layer()
    >>> sol.layer_location
    'x = 0'
    >>> sol.outer
    exp(1 - x)
    >>> sol.expansion
    (1 - exp(x*(eps - 1)/eps))*exp(1 - x)
    """

    def __init__(self):
        self.outer          = None
        self.inner          = None
        self.inner_xi       = None
        self.match          = None
        self.expansion      = None
        self.layer_location = None
        self.layer_var      = None
        self.outer_ode      = None
        self.inner_ode      = None
        self.outer_bc       = None
        self.inner_bc       = None
        self.small_param    = None
        self.independent    = None
        self._method        = "Matched Asymptotic Expansions"
        self._problem_repr  = ""
        # p, q, f coefficients stored for display
        self.p_expr         = None
        self.q_expr         = None
        self.f_expr         = None

    @property
    def components(self):
        r"""Every intermediate substep of the expansion, as live SymPy objects.

        Rather than treating :func:`expand_boundary_layer` as a single opaque
        call, each stage of the matched-asymptotics construction can be
        inspected, plotted, or reused independently.

        Returns
        -------
        dict
            An ordered mapping with the following keys:

            ``'layer_location'`` : str
                Where the boundary layer sits, e.g. ``'x = 0'``.
            ``'outer_ode'`` : sympy.Eq
                The reduced ODE :math:`p(x)\,u' + q(x)\,u = f(x)`.
            ``'outer'`` : sympy.Expr
                Its solution :math:`u_\mathrm{out}(x)` (far boundary condition
                applied).
            ``'inner_ode'`` : sympy.Eq
                The leading-order inner equation
                :math:`U'' + p_\ell\, U' = 0`.
            ``'inner'`` : sympy.Expr
                The inner solution :math:`U(\xi)` in the stretched coordinate.
            ``'inner_xi'`` : sympy.Expr
                The inner solution back in :math:`x`,
                :math:`U\!\big(\xi(x)\big)`.
            ``'match'`` : sympy.Expr
                The matching constant :math:`u_\mathrm{match}`.
            ``'composite'`` : sympy.Expr
                The uniform composite
                :math:`u_\mathrm{out} + U(\xi) - u_\mathrm{match}`.

        Notes
        -----
        Each entry is a SymPy expression (or ``Eq``), ready for ``lambdify``,
        plotting, or further symbolic manipulation. ``'composite'`` is the same
        object as :attr:`expansion`; ``'outer'``, ``'inner'``, ``'inner_xi'``
        and ``'match'`` mirror the like-named attributes.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("eps*u'' + u' + u", dependent='u', small_param='eps',
        ...          independent='x', conditions=['u(0) = 0', 'u(1) = 1'])
        >>> sol = eq.expand_boundary_layer()
        >>> parts = sol.components
        >>> parts['outer']
        exp(1 - x)
        >>> parts['inner']            # U(xi): decays across the layer
        E - exp(1 - xi)
        >>> parts['match']
        E
        >>> sorted(parts)             # doctest: +NORMALIZE_WHITESPACE
        ['composite', 'inner', 'inner_ode', 'inner_xi', 'layer_location',
         'match', 'outer', 'outer_ode']
        """
        return {
            'layer_location': self.layer_location,
            'outer_ode'     : self.outer_ode,
            'outer'         : self.outer,
            'inner_ode'     : self.inner_ode,
            'inner'         : self.inner,
            'inner_xi'      : self.inner_xi,
            'match'         : self.match,
            'composite'     : self.expansion,
        }

    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Compare this expansion against a numerical solution.

        Parameters
        ----------
        eps : float
            Value of the small parameter ε to use.
        problem : ODE, optional
            The original problem object. Defaults to the equation
            used to create this hierarchy — usually not needed.
        **kwargs
            Extra options: ``plot_range`` (``[a, b]``, plotting domain) and
            ``n_points`` (int, number of plot points, default 300).

        Returns
        -------
        dict
            Dictionary with keys ``'x'`` (evaluation points), ``'u_pert'``
            (perturbation composite), ``'u_numerical'`` (SciPy reference),
            ``'fig'`` (matplotlib figure), ``'errors'``, and ``'settings'``.
            Boundary-layer problems additionally return ``'u_outer'``,
            ``'u_inner'``, and ``'u_expansion'``.

        Notes
        -----
        For a boundary-layer problem the reference is obtained with
        ``scipy.integrate.solve_bvp``; the composite expansion is expected to
        track it to :math:`\mathcal{O}(\varepsilon)`, with the largest
        discrepancy inside the thin layer.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("eps*u'' + u' + u", dependent='u', small_param='eps',
        ...          independent='x', conditions=['u(0) = 0', 'u(1) = 1'])
        >>> sol = eq.expand_boundary_layer()
        >>> out = sol.compare_numeric(eps=0.05)
        >>> sorted(out)                                  # doctest: +SKIP
        ['errors', 'fig', 'settings', 'u_numerical', 'u_pert', 'x']
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

    def show(self, mode: str = "auto") -> None:
        """
        Pretty-print the full matched-asymptotics construction.

        Displays the detected layer location, the outer and inner ODEs and
        their solutions, the matching constant, and the composite expansion.

        Parameters
        ----------
        mode : {'auto', 'text', 'latex'}, optional
            Rendering mode. ``'auto'`` (default) renders LaTeX in a Jupyter
            notebook and falls back to plain text in a terminal.

        Returns
        -------
        None
            Output is displayed as a side effect.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("eps*u'' + u' + u", dependent='u', small_param='eps',
        ...          independent='x', conditions=['u(0) = 0', 'u(1) = 1'])
        >>> sol = eq.expand_boundary_layer()
        >>> sol.show(mode='text')                        # doctest: +SKIP
        """
        from asymptotics.display.boundary_layer_display import show_boundary_layer
        show_boundary_layer(self, mode=mode)


# ---------------------------------------------------------------------------
# Layer detection
# ---------------------------------------------------------------------------

def _detect_layer(p_expr, x_sym, a, b):
    """
    Detect which boundary has the layer.

    For eps*u'' + p*u' + ... = 0:
    - p(a) > 0  => characteristics flow from left, layer at x=a
    - p(b) < 0  => characteristics flow from right, layer at x=b
    """
    try:
        pa = float(p_expr.subs(x_sym, a))
        pb = float(p_expr.subs(x_sym, b))
    except Exception:
        pa = float(p_expr.subs(x_sym, a).evalf())
        pb = float(p_expr.subs(x_sym, b).evalf())

    if pa > 0:
        return a, 'left'
    elif pb < 0:
        return b, 'right'
    else:
        raise ValueError(
            f"\n\n  Could not determine layer location.\n"
            f"  p({a}) = {pa:.4f},  p({b}) = {pb:.4f}\n\n"
            f"  For a left layer: need p({a}) > 0\n"
            f"  For a right layer: need p({b}) < 0\n"
            f"  If p changes sign in ({a},{b}), there may be an interior layer\n"
            f"  (not currently supported).\n"
        )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def expand_boundary_layer(problem, order: int = 0) -> BoundaryLayerHierarchy:
    r"""
    Solve a singularly perturbed BVP by matched asymptotic expansions.

    Builds the leading-order boundary-layer solution of

    .. math::

        \varepsilon\, u''(x) + p(x)\, u'(x) + q(x)\, u(x) = f(x),
        \qquad u(a) = \alpha, \quad u(b) = \beta ,

    by detecting the layer location, solving the reduced (outer) problem away
    from the layer, solving the stretched (inner) problem across it, matching
    the two, and assembling the uniform composite

    .. math::

        u_\mathrm{comp}(x) = u_\mathrm{out}(x) + U\!\big(\xi(x)\big)
                             - u_\mathrm{match},

    where :math:`\xi = (x-a)/\varepsilon` for a left layer or
    :math:`\xi = (b-x)/\varepsilon` for a right layer.

    Parameters
    ----------
    problem : ODE
        A second-order BVP of the form above, with the small parameter
        multiplying :math:`u''` and Dirichlet conditions at two distinct
        points.
    order : int, optional
        Expansion order. Only ``order=0`` (leading order) is currently
        supported. Default ``0``.

    Returns
    -------
    BoundaryLayerHierarchy
        Container holding the outer/inner ODEs and solutions, the matching
        constant, and the composite expansion, plus the shared display and
        evaluation API. See :attr:`BoundaryLayerHierarchy.components`.

    Raises
    ------
    NoSmallParameterError
        If the small parameter does not appear in the equation.
    ValueError
        If the ODE is not second order, is not a two-point BVP, is missing
        Dirichlet conditions at both endpoints, or if the layer location
        cannot be determined from the sign of :math:`p` (e.g. an interior
        layer, which is unsupported).
    NotImplementedError
        If ``order > 0`` is requested.
    RuntimeError
        If SymPy fails to solve the outer or inner ODE.

    Notes
    -----
    The layer is placed at the left boundary when :math:`p(a) > 0` and at the
    right boundary when :math:`p(b) < 0`. The far boundary condition fixes the
    outer solution; the near boundary condition fixes the inner solution.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("eps*u'' + u' + u", dependent='u', small_param='eps',
    ...          independent='x', conditions=['u(0) = 0', 'u(1) = 1'])
    >>> sol = eq.expand_boundary_layer()      # layer at x = 0 (p = 1 > 0)
    >>> sol.outer
    exp(1 - x)
    >>> sol.match
    E
    >>> sol.expansion
    (1 - exp(x*(eps - 1)/eps))*exp(1 - x)

    A right-hand layer arises when the sign of :math:`p` flips:

    >>> eq2 = ODE("eps*u'' - u' + u", dependent='u', small_param='eps',
    ...           independent='x', conditions=['u(0) = 0', 'u(1) = 1'])
    >>> sol2 = eq2.expand_boundary_layer()    # layer at x = 1 (p = -1 < 0)
    >>> sol2.layer_location
    'x = 1'
    >>> sol2.outer
    0
    """
    from asymptotics.core.exceptions import NoSmallParameterError

    eps        = problem.small_param
    x          = problem._indep_sym
    dep        = problem._dependent_name
    f_orig     = problem.equation
    deriv_syms = problem._deriv_syms
    conds      = problem.conditions

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if eps not in f_orig.free_symbols:
        raise NoSmallParameterError(eps, f_orig)

    if problem.ode_order != 2:
        raise ValueError(
            "\n\n  Boundary layer expansion requires a 2nd-order ODE.\n"
            f"  Got order {problem.ode_order}.\n"
        )

    if problem.problem_type != 'bvp':
        raise ValueError(
            "\n\n  Boundary layer expansion requires a BVP.\n"
            "  Provide conditions at two different points, e.g.:\n"
            "    conditions=['u(0) = 0', 'u(1) = 1']\n"
        )

    if order > 0:
        raise NotImplementedError(
            "\n\n  Only leading-order (order=0) boundary layer expansion\n"
            "  is currently supported.\n"
        )

    du_sym  = deriv_syms.get(1)
    d2u_sym = deriv_syms.get(2)
    u_sym   = Symbol(dep)
    C1, C2  = symbols('C1 C2')

    # ------------------------------------------------------------------
    # Extract p, q, f from equation: d2u_coeff*eps*u'' + p*u' + q*u - f = 0
    # ------------------------------------------------------------------
    f_at_eps0 = f_orig.subs(eps, Integer(0))

    # Coefficient of d2u_sym should be eps (or eps * something)
    d2u_coeff = f_orig.coeff(d2u_sym)
    p_expr    = f_at_eps0.coeff(du_sym)   if du_sym  else Integer(0)
    q_expr    = f_at_eps0.coeff(u_sym)
    # rhs f: everything else at eps=0
    f_expr    = -(f_at_eps0 - p_expr*du_sym - q_expr*u_sym) if du_sym else \
                -(f_at_eps0 - q_expr*u_sym)

    # ------------------------------------------------------------------
    # Get boundary points and values from conditions
    # ------------------------------------------------------------------
    # conds are ParsedCondition objects: .point, .value, .deriv_order
    bc_pts  = sorted(set(c.point for c in conds), key=lambda p: float(p.evalf()))
    a, b    = bc_pts[0], bc_pts[1]

    # Find u(a) and u(b) values
    bc_dict = {}
    for c in conds:
        if c.deriv_order == 0:
            bc_dict[c.point] = c.value

    alpha = bc_dict.get(a)   # u(a)
    beta  = bc_dict.get(b)   # u(b)

    if alpha is None or beta is None:
        raise ValueError(
            "\n\n  Could not find Dirichlet BCs u(a) and u(b).\n"
            "  Boundary layer expansion requires u(a)=alpha, u(b)=beta.\n"
        )

    # ------------------------------------------------------------------
    # Detect layer location
    # ------------------------------------------------------------------
    layer_pt, layer_side = _detect_layer(p_expr, x, a, b)
    xi = Symbol('xi')   # stretched coordinate

    if layer_side == 'left':
        # Layer at x=a, far BC at x=b
        far_pt, far_val   = b, beta
        near_pt, near_val = a, alpha
        xi_expr = (x - a) / eps
        # p at the layer point (for inner ODE)
        p_layer = p_expr.subs(x, a)
        layer_str = f"x = {a}"
    else:
        # Layer at x=b, far BC at x=a
        far_pt, far_val   = a, alpha
        near_pt, near_val = b, beta
        xi_expr = (b - x) / eps
        # For right layer: xi = (b-x)/eps, so d/dx = -1/eps * d/dxi
        # u'' = 1/eps^2 * U'',  u' = -1/eps * U'
        # Inner ODE: U'' - p(b)*U' = 0  (note sign flip)
        p_layer = -p_expr.subs(x, b)
        layer_str = f"x = {b}"

    h = BoundaryLayerHierarchy()
    h.small_param    = eps
    h.independent    = x
    h.layer_var      = xi
    h.layer_location = layer_str
    h.p_expr         = p_expr
    h.q_expr         = q_expr
    h.f_expr         = f_expr
    h._method        = "Matched Asymptotic Expansions"
    h._problem_repr  = (
        f"{eps}·{dep}'' + ({p_expr})·{dep}' + ({q_expr})·{dep} = {f_expr}"
    )

    # ------------------------------------------------------------------
    # Step 1: Outer solution
    # Solve reduced ODE (eps=0): p(x)*u' + q(x)*u = f(x)
    # with BC at the far boundary
    # ------------------------------------------------------------------
    u_fn = Function(dep)
    outer_ode = Eq(
        p_expr * u_fn(x).diff(x) + q_expr * u_fn(x),
        f_expr
    )
    h.outer_ode = outer_ode
    h.outer_bc  = f"u({far_pt}) = {far_val}"

    try:
        outer_sol = dsolve(outer_ode, u_fn(x), ics={u_fn(far_pt): far_val})
        if isinstance(outer_sol, list):
            outer_sol = outer_sol[0]
        outer_expr = simplify(outer_sol.rhs)
    except Exception as e:
        raise RuntimeError(
            f"\n\n  Could not solve outer ODE: {outer_ode}\n"
            f"  Error: {e}\n"
        ) from e

    h.outer = outer_expr

    # ------------------------------------------------------------------
    # Step 2: Inner solution
    # Stretched coordinate xi, leading-order inner ODE: U'' + p_layer*U' = 0
    # with BC U(0) = near_val
    # ------------------------------------------------------------------
    U_fn = Function('U')
    inner_ode = Eq(
        U_fn(xi).diff(xi, 2) + p_layer * U_fn(xi).diff(xi),
        Integer(0)
    )
    h.inner_ode = inner_ode
    h.inner_bc  = f"U(0) = {near_val}"

    try:
        inner_gen = dsolve(inner_ode, U_fn(xi))
        if isinstance(inner_gen, list):
            inner_gen = inner_gen[0]
        inner_gen_expr = inner_gen.rhs
    except Exception as e:
        raise RuntimeError(
            f"\n\n  Could not solve inner ODE: {inner_ode}\n"
            f"  Error: {e}\n"
        ) from e

    # ------------------------------------------------------------------
    # Step 3: Matching
    # As xi -> inf, U(xi) -> outer(layer_pt)
    # ------------------------------------------------------------------
    outer_at_layer = simplify(outer_expr.subs(x, layer_pt))
    h.match = outer_at_layer

    # The inner general solution is C1 + C2*exp(-p_layer*xi)
    # (assuming p_layer > 0 so exp term decays)
    # Matching: lim_{xi->inf} U = C1 = outer_at_layer
    # Near BC: U(0) = C1 + C2 = near_val => C2 = near_val - C1

    try:
        lim_val = limit(inner_gen_expr, xi, oo)
    except Exception:
        lim_val = C1   # fallback

    # Solve for constants
    match_eq  = Eq(lim_val, outer_at_layer)
    near_eq   = Eq(inner_gen_expr.subs(xi, 0), near_val)

    try:
        const_sol = solve([match_eq, near_eq], [C1, C2])
        inner_particular = simplify(inner_gen_expr.subs(const_sol))
    except Exception:
        # Fallback: assume standard form C1 + C2*exp(...)
        C1_val = outer_at_layer
        C2_val = near_val - outer_at_layer
        inner_particular = inner_gen_expr.subs([(C1, C1_val), (C2, C2_val)])
        inner_particular = simplify(inner_particular)

    h.inner     = inner_particular
    h.inner_xi  = inner_particular.subs(xi, xi_expr)

    # ------------------------------------------------------------------
    # Step 4: Expansion solution
    # u_comp = u_outer(x) + U(xi) - match_value
    # with xi = xi_expr
    # ------------------------------------------------------------------
    expansion = outer_expr + inner_particular.subs(xi, xi_expr) - outer_at_layer
    h.expansion = simplify(expansion)

    h._problem = problem
    return h
