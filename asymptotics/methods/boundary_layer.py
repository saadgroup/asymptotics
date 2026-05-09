"""
asymptotics.methods.boundary_layer
================================
Matched asymptotic expansions for linear singular perturbation BVPs.

For equations of the form:
    eps * u'' + p(x) * u' + q(x) * u = f(x)
    u(a) = alpha,  u(b) = beta

where eps << 1 multiplies the highest derivative.

Algorithm
---------
1. Detect layer location from sign of p(x) at boundaries:
   - p(a) > 0  =>  layer at x = a  (left boundary)
   - p(b) < 0  =>  layer at x = b  (right boundary)

2. Outer solution: solve reduced ODE (eps=0), apply far BC.

3. Inner solution: introduce stretched coordinate
   - Layer at x=a: xi = (x-a)/eps
   - Layer at x=b: xi = (b-x)/eps
   Solve leading-order inner ODE with near BC.

4. Matching: inner limit as xi->inf must equal outer limit at boundary.

5. Composite: u_comp = u_outer + u_inner - u_match

Each step is stored for inspection.
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
    """
    Result of a boundary layer (matched asymptotic) expansion.

    Attributes
    ----------
    outer           : Expr   — outer solution u_out(x)
    inner           : Expr   — inner solution U(xi)
    inner_xi        : Expr   — inner in original variable: U((x-a)/eps)
    match           : Expr   — matching value (constant)
    composite       : Expr   — u_out + U(xi) - match, with xi substituted
    layer_location  : str    — 'x=a' or 'x=b'
    layer_var       : Symbol — the stretched coordinate xi
    outer_bc        : str    — which BC the outer solution satisfies
    inner_bc        : str    — which BC the inner solution satisfies
    small_param     : Symbol
    independent     : Symbol
    omega_0         : None   (unused, for API consistency)
    """

    def __init__(self):
        self.outer          = None
        self.inner          = None
        self.inner_xi       = None
        self.match          = None
        self.composite      = None
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

    def show(self, mode: str = "auto") -> None:
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
    """
    Apply matched asymptotic expansions to a singular perturbation BVP.

    Parameters
    ----------
    problem : ODE
        Must be a 2nd-order BVP: eps*u'' + p(x)*u' + q(x)*u = f(x)
        with conditions at two distinct points.
    order : int
        Currently only order=0 (leading order) is supported.

    Returns
    -------
    BoundaryLayerHierarchy
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
    # Step 4: Composite solution
    # u_comp = u_outer(x) + U(xi) - match_value
    # with xi = xi_expr
    # ------------------------------------------------------------------
    composite = outer_expr + inner_particular.subs(xi, xi_expr) - outer_at_layer
    h.composite = simplify(composite)

    h._problem = problem
    return h
