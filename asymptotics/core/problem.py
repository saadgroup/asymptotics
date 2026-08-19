"""
asymptotics.core.problem
========================
Problem definition layer. All inputs are strings — asymptotics creates the
SymPy symbols internally. Users never need to call symbols() themselves.
"""

from __future__ import annotations
from sympy import symbols, sympify, Symbol, Expr


def _parse_algebraic(equation: str, dependent: str, small_param: str) -> tuple:
    """
    Parse a string equation into SymPy objects.

    Creates symbols for the dependent variable and small parameter,
    then sympifies the equation string in that namespace.

    Returns (equation_expr, dep_symbol, param_symbol)
    """
    # Standard math functions available automatically via sympify
    # We only need to declare the user-defined names
    ns = {}
    dep_sym   = symbols(dependent)
    param_sym = symbols(small_param)
    ns[dependent]   = dep_sym
    ns[small_param] = param_sym

    try:
        eq_expr = sympify(equation, locals=ns, convert_xor=False)
    except Exception as e:
        raise ValueError(
            f"\n\n  Could not parse equation: '{equation}'\n"
            f"  SymPy error: {e}\n\n"
            f"  Tips:\n"
            f"    - Use ** for powers:   x**3 not x^3\n"
            f"    - Use * for products:  eps*x not eps·x\n"
            f"    - Functions like cos(), sin(), exp(), log() work out of the box\n"
        ) from e

    return eq_expr, dep_sym, param_sym


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PerturbationEquation:
    r"""
    Base class for every perturbation problem in :mod:`asymptotics`.

    A perturbation problem is written in the residual (equal-to-zero) form

    .. math::

        F(u,\,\varepsilon) = 0,

    where :math:`u` is the unknown (a scalar, a function, or a vector) and
    :math:`\varepsilon` is the small parameter. This base class stores the
    parsed SymPy residual together with the SymPy symbols for the unknown
    and the small parameter. It performs *no* expansion itself — the
    order-by-order machinery lives on the concrete subclasses
    (:class:`AlgebraicEquation`, :class:`ODE`, :class:`AlgebraicSystem`, and
    the coupled-ODE type ``ODESystem``), each of which exposes the
    ``expand_*`` methods appropriate to its structure.

    All user-facing subclasses accept plain **strings** and build the
    required SymPy symbols internally, so callers never invoke
    :func:`sympy.symbols` themselves.

    Parameters
    ----------
    equation : sympy.Expr
        The residual :math:`F` as an already-parsed SymPy expression,
        understood to be set equal to zero.
    small_param : sympy.Symbol
        The symbol used for the small parameter :math:`\varepsilon`.
    dependent : sympy.Symbol
        The symbol for the dependent variable / unknown :math:`u`.

    Attributes
    ----------
    equation : sympy.Expr
        The stored residual expression.
    small_param : sympy.Symbol
        The small-parameter symbol.
    dependent : sympy.Symbol
        The dependent-variable symbol.

    Notes
    -----
    This class is not intended to be instantiated directly; use one of the
    subclasses, which parse strings and attach the correct expansion
    methods.
    """

    def __init__(self, equation: Expr, small_param: Symbol, dependent: Symbol):
        self.equation    = equation
        self.small_param = small_param
        self.dependent   = dependent

    def __repr__(self):
        return f"{self.__class__.__name__}({self.equation} = 0)"


# ---------------------------------------------------------------------------
# Algebraic equations:  f(x, eps) = 0
# ---------------------------------------------------------------------------

class AlgebraicEquation(PerturbationEquation):
    r"""
    A single scalar algebraic perturbation equation :math:`f(x,\varepsilon)=0`.

    Represents one nonlinear (possibly transcendental) equation in one
    unknown that depends on a small parameter. Regular perturbation theory
    seeks the root as a power series in :math:`\varepsilon`,

    .. math::

        x(\varepsilon) \;=\; x_0 + x_1\,\varepsilon + x_2\,\varepsilon^2
        + \cdots ,

    substitutes it into :math:`f`, expands in :math:`\varepsilon`, and solves
    the resulting sequence of equations one order at a time. The
    leading-order equation :math:`f(x_0,0)=0` fixes :math:`x_0`; each higher
    order is *linear* in the new coefficient :math:`x_k` and is solved in
    closed form.

    All inputs are plain strings — no :func:`sympy.symbols` calls needed. The
    constructor creates the SymPy symbols internally.

    Parameters
    ----------
    equation : str
        The equation, understood as set equal to zero. Use ``**`` for
        powers and ``*`` for products (``x^3`` is rejected). Standard
        functions (``cos``, ``sin``, ``exp``, ``log``, ``tan``, ``sqrt``,
        ...) are recognised automatically.
    dependent : str
        Name of the unknown variable, e.g. ``'x'``.
    small_param : str
        Name of the small parameter, e.g. ``'eps'``. It is always rendered
        as :math:`\varepsilon` in LaTeX output.
    root_hint : float or int, optional
        Numerical value of the leading-order root :math:`x_0` to follow.
        When several real roots of :math:`f(x_0,0)=0` exist this selects the
        branch nearest ``root_hint``. If ``None`` (default) the solver falls
        back to ``root_index`` (see :meth:`expand_regular`).

    Attributes
    ----------
    equation : sympy.Expr
        The parsed residual :math:`f`.
    small_param : sympy.Symbol
        The small-parameter symbol.
    dependent : sympy.Symbol
        The unknown symbol.
    params : set of str
        Names of any extra symbolic parameters detected in the equation
        (symbols other than the unknown and the small parameter). If
        non-empty, values must be supplied at ``eval``/``compare_numeric``
        time via ``params={...}``.

    Raises
    ------
    ValueError
        If the equation string cannot be parsed (e.g. ``^`` used for a
        power, or an unbalanced expression).

    See Also
    --------
    AlgebraicSystem : coupled algebraic systems.
    ODE : differential equations.

    Examples
    --------
    >>> from asymptotics import AlgebraicEquation
    >>> eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
    >>> sol = eq.expand_regular(order=3)
    >>> sol.expansion
    eps**3/81 - eps/3 + 1

    Transcendental equations work too:

    >>> eq  = AlgebraicEquation("tan(x) - 1 - eps*x**2", dependent="x", small_param="eps")
    >>> sol = eq.expand_regular(order=2)
    >>> from sympy import pi, simplify
    >>> simplify(sol[0].solution - pi/4)
    0
    """

    def __init__(
        self,
        equation: str,
        dependent: str,
        small_param: str,
        root_hint=None,
    ):
        eq_expr, dep_sym, param_sym = _parse_algebraic(equation, dependent, small_param)
        super().__init__(eq_expr, param_sym, dep_sym)
        self.root_hint = root_hint

        # Keep string names for display
        self._dependent_name   = dependent
        self._small_param_name = small_param

        # Detect symbolic parameters
        raw_params = _detect_params(str(self.equation), dependent, '', small_param)
        self.params = raw_params
        if raw_params:
            _params_warning(raw_params, method='eval', has_at=False)


    def expand_regular(self, order: int = 3, root_index: int = 0, gauge=None):
        r"""
        Solve this algebraic equation by regular perturbation theory.

        Posits the ansatz

        .. math::

            x(\varepsilon) \;=\; \sum_{k=0}^{N} x_k\,\delta_k(\varepsilon),
            \qquad N = \texttt{order},

        with gauge functions :math:`\delta_k` (the standard sequence
        :math:`\{1,\varepsilon,\varepsilon^2,\dots\}` unless ``gauge`` is
        given), substitutes it into :math:`f(x,\varepsilon)=0`, expands, and
        collects powers of the gauge. The leading order

        .. math::

            f(x_0, 0) = 0

        determines :math:`x_0`. Each subsequent order is linear in the new
        coefficient: writing :math:`f_x = \partial f/\partial x`, the
        :math:`O(\varepsilon^k)` balance has the form

        .. math::

            f_x(x_0,0)\,x_k \;=\; (\text{known terms in } x_0,\dots,x_{k-1}),

        which is solved directly for :math:`x_k`.

        Parameters
        ----------
        order : int, optional
            Highest gauge index :math:`N` to compute (inclusive). Default 3,
            producing coefficients :math:`x_0,\dots,x_3`.
        root_index : int, optional
            Which real root of the leading-order equation
            :math:`f(x_0,0)=0` to follow, after the real roots are sorted in
            descending order (``0`` = largest real root, the default).
            Ignored when ``root_hint`` was supplied to the constructor.
        gauge : str, list of str, or None, optional
            Non-standard asymptotic gauge sequence :math:`\delta_k`.

            - ``None`` (default) — standard sequence
              :math:`\{1,\varepsilon,\varepsilon^2,\dots\}`.
            - ``str`` — a pattern for geometric inference, e.g.
              ``'sqrt(eps)'`` generates
              :math:`\{1,\varepsilon^{1/2},\varepsilon,\varepsilon^{3/2},\dots\}`.
            - ``list`` — an explicit sequence of length ``order + 1``, e.g.
              ``['1', 'eps*log(eps)', 'eps']``.

        Returns
        -------
        OrderHierarchy
            Indexable container of per-order results. ``sol[k].solution`` is
            the coefficient :math:`x_k`, ``sol[k].equation`` the
            :math:`O(\varepsilon^k)` equation, ``sol.expansion`` the
            assembled series, and ``sol.eval(eps=...)`` a float evaluation.

        Raises
        ------
        NoSmallParameterError
            If :math:`\varepsilon` does not appear in the equation.
        NoLeadingOrderSolutionError
            If the leading-order equation has no solution.
        OnlyComplexRootsError
            If the leading-order equation has no real roots to follow.
        NoHigherOrderSolutionError
            If a higher-order equation cannot be solved for its coefficient.

        Examples
        --------
        >>> from asymptotics import AlgebraicEquation
        >>> eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
        >>> sol = eq.expand_regular(order=3)
        >>> sol[0].solution, sol[1].solution
        (1, -1/3)
        >>> sol.expansion
        eps**3/81 - eps/3 + 1
        >>> round(sol.eval(eps=0.1), 6)
        0.966679

        Selecting a different leading-order branch with ``root_index``:

        >>> eq2 = AlgebraicEquation("x**2 + eps*x - 1", dependent="x", small_param="eps")
        >>> eq2.expand_regular(order=2, root_index=0)[0].solution
        1
        >>> eq2.expand_regular(order=2, root_index=1)[0].solution
        -1
        """
        from asymptotics.methods.regular_algebraic import expand_regular_algebraic
        return expand_regular_algebraic(self, order=order, root_index=root_index,
                                        gauge=gauge)


# ---------------------------------------------------------------------------
# Inference helpers for ODE
# ---------------------------------------------------------------------------

import re as _re

_INDEP_CANDIDATES = {'x', 'y', 'z', 't', 'r'}

# Single definition — used by both inference and parameter-detection helpers.
_MATH_NAMES = {
    'sin','cos','exp','log','tan','cot','sec','csc',
    'sqrt','pi','E','I','oo','Abs','sign','floor','ceiling',
    'sinh','cosh','tanh','asin','acos','atan','atan2',
    'lim',  # limit condition keyword
}

def _infer_dependent(conditions):
    """Extract dependent variable name from condition strings.

    Skips ``lim(...)`` conditions — they start with the keyword 'lim', not
    the dependent variable name, and would cause a false match.
    """
    for cond in conditions:
        stripped = cond.strip()
        if stripped.lower().startswith('lim('):
            continue
        m = _re.match(r"^([a-zA-Z_]\w*)", stripped)
        if m:
            return m.group(1)
    raise ValueError(
        "\n\n  Could not infer dependent variable from conditions.\n"
        "  Specify it explicitly: ODE(..., dependent='u')\n"
    )

def _infer_independent(eq_str, dep_name, small_param, problem_type):
    """
    Infer independent variable from equation string.
    Only considers {x, y, z, t} as candidates — everything else
    is treated as a parameter.
    """
    tokens     = set(_re.findall(r"[a-zA-Z_]\w*", eq_str))
    exclude    = {dep_name, small_param} | _MATH_NAMES
    candidates = (tokens & _INDEP_CANDIDATES) - exclude
    if len(candidates) == 1:
        return candidates.pop()
    # Fallback: IVP -> 't', BVP -> 'x'
    return 't' if problem_type == 'ivp' else 'x'

def _print_inference(dep, indep, dep_supplied, indep_supplied):
    """Print a clear statement of what was inferred vs supplied."""
    dep_note   = "supplied" if dep_supplied   else "inferred from conditions"
    indep_note = "supplied" if indep_supplied else "inferred from equation"
    print(
        f"  ℹ️  dependent = '{dep}' ({dep_note}), "
        f"independent = '{indep}' ({indep_note})\n"
        f"     To override: ODE(..., dependent='{dep}', independent='{indep}')"
    )

# ---------------------------------------------------------------------------
# Parameter detection
# ---------------------------------------------------------------------------

def _params_warning(params, method='eval', has_at=False):
    """Print a consistent warning when symbolic parameters are detected."""
    param_dict = ', '.join(f"'{p}': value" for p in sorted(params))
    at_arg = ', at=t_vals' if has_at else ''
    print(
        f"  ⚠️  symbolic parameters detected: {set(sorted(params))}\n"
        f"     Provide values at eval/compare time:\n"
        f"     sol.{method}(eps=0.1{at_arg}, params={{{param_dict}}})"
    )


def _params_error(params, method='eval', has_at=False):
    """Return a consistent error message for missing symbolic parameters."""
    param_dict = ', '.join(f"'{p}': value" for p in sorted(params))
    at_arg = ', at=t_vals' if has_at else ''
    return (
        f"\n\n  Equation has symbolic parameters: {set(sorted(params))}\n"
        f"  Provide values:\n"
        f"    sol.{method}(eps=..{at_arg}, params={{{param_dict}}})\n"
    )

def _detect_params(eq_str, dependent, independent, small_param):
    """
    Detect symbolic parameters in an equation string.
    Parameters are free symbols that are not: dependent, independent,
    small_param, math function names, or derivative notation artifacts.
    """
    tokens = set(_re.findall(r"[a-zA-Z_]\w*", eq_str))
    exclude = {dependent, independent or '', small_param} | _MATH_NAMES
    # Exclude derivative notation: du, d2u, d3u, d4u, d5u, d6u
    exclude |= {f'd{dependent}'} | {f'd{n}{dependent}' for n in range(2, 7)}
    return tokens - exclude


def _check_ambiguous_params(params, indep_candidates, indep_supplied, dependent):
    """
    If any detected parameter is in {x,y,z,t} and independent was inferred
    (not explicitly supplied), raise a hard error.
    """
    ambiguous = params & indep_candidates
    if ambiguous and not indep_supplied:
        raise ValueError(
            f"\n\n  Ambiguous symbols detected: {ambiguous}\n"
            f"  These could be the independent variable OR symbolic parameters.\n"
            f"  Please specify 'independent' explicitly to resolve the ambiguity:\n"
            f"    ODE(..., independent='t')   # if {list(ambiguous)[0]} is a parameter\n"
            f"    ODE(..., independent='{list(ambiguous)[0]}')  # if it's the independent variable\n"
        )


# ---------------------------------------------------------------------------
# ODE equations
# ---------------------------------------------------------------------------

def _preprocess_ode_string(eq_str: str, dep: str) -> str:
    """
    Convert prime notation to internal derivative symbols.
    Process longest first: u''''->d4u, u'''->d3u, u''->d2u, u'->du
    """
    for n in range(6, 0, -1):
        primes = "'" * n
        sym = f"d{n}{dep}" if n > 1 else f"d{dep}"
        eq_str = eq_str.replace(dep + primes, sym)
    return eq_str


def _detect_ode_order(eq_str: str, dep: str) -> int:
    """Detect ODE order from prime notation in the equation string."""
    for n in range(6, 0, -1):
        primes = "'" * n
        if dep + primes in eq_str:
            return n
    return 0


class ODE(PerturbationEquation):
    r"""
    A single ordinary differential equation :math:`F(u,u',u'',\dots,t,\varepsilon)=0`.

    Represents an ODE of order 1–6 that depends on a small parameter,
    together with the initial or boundary conditions that close it. The
    class recognises the structure of the problem (its order, and whether it
    is an initial-value or boundary-value problem) and offers several
    expansion strategies as methods:

    - :meth:`expand_regular` — straightforward power-series (regular)
      perturbation, valid when the expansion stays uniformly ordered.
    - :meth:`expand_lindstedt` — Lindstedt–Poincaré strained-coordinate
      method for periodic solutions of weakly nonlinear oscillators.
    - :meth:`expand_multiple_scales` — method of multiple scales, resolving
      slow modulation (damping, amplitude/phase drift) of oscillators.
    - :meth:`expand_boundary_layer` — matched asymptotic expansions for
      singularly perturbed BVPs :math:`\varepsilon u'' + p(x)u' + q(x)u = f`.
    - :meth:`begin_expansion` — step-by-step control: extract the
      order equations symbolically and solve (or supply) them one at a time.

    All inputs are plain strings — no :func:`sympy.symbols` needed. Use prime
    notation for derivatives: ``u'`` for :math:`du/dt`, ``u''`` for
    :math:`d^2u/dt^2`, up to sixth order.

    Parameters
    ----------
    equation : str
        The ODE, understood as set equal to zero, e.g.
        ``"u'' + u + eps*u**3"`` or ``"u' + eps*u**2 + u"``. Use ``**`` for
        powers and ``*`` for products; standard functions are recognised.
    small_param : str
        Name of the small parameter, e.g. ``'eps'``. Rendered as
        :math:`\varepsilon` in LaTeX output.
    conditions : list of str
        Initial or boundary conditions, one per string, e.g.
        ``["u(0) = 1", "u'(0) = 0"]`` (IVP) or ``["u(0) = 0", "u(1) = 1"]``
        (BVP). The number of conditions must equal the detected ODE order.
        Whether the problem is treated as an IVP or a BVP is inferred from
        whether the conditions are all imposed at a single point.
    dependent : str, optional
        Name of the dependent variable, e.g. ``'u'``. If omitted it is
        inferred from the first symbol appearing in ``conditions``.
    independent : str, optional
        Name of the independent variable, e.g. ``'t'``. If omitted it is
        inferred from the equation (candidates ``x, y, z, t, r``); the
        fallback is ``'t'`` for an IVP and ``'x'`` for a BVP.

    Attributes
    ----------
    equation : sympy.Expr
        The parsed residual, with derivatives represented by internal
        symbols ``du``, ``d2u``, ...
    ode_order : int
        Detected order of the ODE (from the highest prime count).
    problem_type : {'ivp', 'bvp'}
        Whether the conditions define an initial- or boundary-value problem.
    conditions : list
        Parsed and validated condition objects.
    small_param, dependent : sympy.Symbol
        The small-parameter and dependent-variable symbols.
    params : set of str
        Names of extra symbolic parameters detected in the equation or in
        the condition values; values for these must be supplied at
        ``eval``/``compare_numeric`` time via ``params={...}``.

    Raises
    ------
    ValueError
        If no derivative of the dependent variable is found, if the equation
        cannot be parsed, or if a symbol is ambiguous between the
        independent variable and a parameter.
    ConditionError
        If the conditions are malformed, conflicting, or of the wrong count
        for the ODE order.

    Notes
    -----
    On construction a short summary line is printed reporting which of
    ``dependent``/``independent`` were supplied versus inferred, so that any
    misparse is visible immediately.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> # First-order IVP
    >>> eq = ODE(
    ...     "u' + u + eps*u**2",
    ...     dependent="u", small_param="eps", independent="t",
    ...     conditions=["u(0) = 1"],
    ... )                                          # doctest: +SKIP
    >>> sol = eq.expand_regular(order=2)           # doctest: +SKIP
    >>> sol[0].particular_solution                 # doctest: +SKIP
    exp(-t)

    >>> # Second-order BVP
    >>> eq = ODE(
    ...     "u'' + eps*u**3",
    ...     dependent="u", small_param="eps", independent="t",
    ...     conditions=["u(0) = 0", "u(1) = 1"],
    ... )                                          # doctest: +SKIP
    """

    def __init__(
        self,
        equation:    str,
        small_param: str,
        conditions:  list,
        dependent:   str = None,
        independent: str = None,
    ):
        from sympy import symbols as _symbols, Function, sympify as _sympify
        from asymptotics.core.conditions import parse_and_validate_conditions

        # ------------------------------------------------------------------
        # Infer dependent and independent if not supplied
        # ------------------------------------------------------------------
        dep_supplied   = dependent   is not None
        indep_supplied = independent is not None

        if not dep_supplied:
            dependent = _infer_dependent(conditions)

        # Need ODE order to detect IVP/BVP before inferring independent
        self.ode_order = _detect_ode_order(equation, dependent)
        if self.ode_order == 0:
            raise ValueError(
                f"\n\n  No derivatives of '{dependent}' found in equation '{equation}'.\n"
                f"  Use prime notation: u' for first derivative, u'' for second.\n"
            )

        # Parse and validate conditions (needed to determine IVP/BVP)
        self.conditions, self.problem_type = parse_and_validate_conditions(
            conditions, dependent, self.ode_order
        )

        if not indep_supplied:
            independent = _infer_independent(
                equation, dependent, small_param, self.problem_type
            )

        # Print inference summary
        _print_inference(dependent, independent, dep_supplied, indep_supplied)

        # Detect symbolic parameters — scan both equation AND condition values
        raw_params = _detect_params(equation, dependent, independent, small_param)

        # Also detect symbols in condition values (e.g. u(0) = A)
        # Skip limit conditions — they have complex syntax handled separately
        for cond_str in conditions:
            if cond_str.strip().lower().startswith('lim('):
                continue
            cond_params = _detect_params(cond_str, dependent, independent, small_param)
            raw_params |= cond_params

        # Remove any symbols that are clearly points (numbers) not parameters
        # Keep only symbols that SymPy would treat as unknowns
        from sympy import sympify as _sympify
        confirmed_params = set()
        for p in raw_params:
            try:
                val = _sympify(p)
                if val.is_Symbol:
                    confirmed_params.add(p)
            except Exception:
                pass
        raw_params = confirmed_params

        _check_ambiguous_params(raw_params, _INDEP_CANDIDATES, indep_supplied, dependent)
        self.params = raw_params   # set of symbol names

        if raw_params:
            _params_warning(raw_params, method='eval', has_at=True)

        # Store names
        self._equation_str     = equation
        self._dependent_name   = dependent
        self._small_param_name = small_param
        self._independent_name = independent

        # Build internal symbols
        dep_sym   = _symbols(dependent)
        param_sym = _symbols(small_param)
        indep_sym = _symbols(independent)

        # Build derivative symbols: du, d2u, ...
        self._indep_sym = indep_sym
        self._deriv_syms = {}   # order -> symbol
        for k in range(1, self.ode_order + 1):
            prefix = "d" * k if k == 1 else f"d{k}"
            self._deriv_syms[k] = _symbols(f"{prefix}{dependent}")

        # Parse equation string: replace primes, then sympify
        processed = _preprocess_ode_string(equation, dependent)
        ns = {
            dependent:   dep_sym,
            small_param: param_sym,
            independent: indep_sym,
        }
        for k, sym in self._deriv_syms.items():
            prefix = "d" * k if k == 1 else f"d{k}"
            ns[f"{prefix}{dependent}"] = sym

        try:
            eq_expr = sympify(processed, locals=ns, convert_xor=False)
        except Exception as e:
            raise ValueError(
                f"\n\n  Could not parse ODE: '{equation}'\n"
                f"  SymPy error: {e}\n\n"
                f"  Tips:\n"
                f"    - Use prime notation: u', u''\n"
                f"    - Use ** for powers, * for products\n"
                f"    - Functions like cos(), sin(), exp() work out of the box\n"
            ) from e

        super().__init__(eq_expr, param_sym, dep_sym)

    def _validate_order(self, order, method_name):
        """Validate the order argument is a non-negative integer."""
        if not isinstance(order, int):
            raise TypeError(
                f"\n\n  '{method_name}' order must be an integer, got {type(order).__name__}: {order!r}\n"
                f"  Example: eq.{method_name}(order=2)\n"
            )
        if order < 0:
            raise ValueError(
                f"\n\n  '{method_name}' order must be non-negative, got {order}.\n"
            )

    def expand_regular(self, order: int = 2, gauge=None):
        r"""
        Solve this ODE by regular (Poincaré) perturbation theory.

        Expands the solution as a power series in the small parameter,

        .. math::

            u(t,\varepsilon) \;=\; \sum_{k=0}^{N} u_k(t)\,\delta_k(\varepsilon),
            \qquad N = \texttt{order},

        substitutes it into the ODE and the conditions, and collects like
        powers of the gauge. This yields a *sequence of linear ODEs* for the
        coefficient functions :math:`u_k(t)`. The leading order reproduces
        the unperturbed problem :math:`F(u_0,u_0',\dots,t,0)=0`; each higher
        order is a linear ODE with the *same* leading differential operator
        :math:`L` and a right-hand side built from lower orders,

        .. math::

            L\,u_k \;=\; R_k[u_0,\dots,u_{k-1}],

        solved subject to the homogeneous form of the original conditions
        (all inhomogeneity is carried by :math:`u_0`). Works for both IVPs
        and BVPs of order 1–6.

        Regular perturbation is only valid when the expansion is uniformly
        ordered. For weakly nonlinear oscillators it produces *secular*
        terms (unbounded :math:`t\sin t`, :math:`t\cos t`) that grow without
        bound; each order flags this via ``sol[k].secular``. When secular
        terms appear, use :meth:`expand_lindstedt` or
        :meth:`expand_multiple_scales` instead.

        Parameters
        ----------
        order : int, optional
            Highest gauge index :math:`N` to compute (inclusive). Default 2.
        gauge : str, list of str, or None, optional
            Non-standard asymptotic gauge sequence :math:`\delta_k`.

            - ``None`` (default) — standard
              :math:`\{1,\varepsilon,\varepsilon^2,\dots\}`.
            - ``str`` — a pattern for geometric inference, e.g.
              ``'sqrt(eps)'`` gives
              :math:`\{1,\varepsilon^{1/2},\varepsilon,\dots\}`.
            - ``list`` — an explicit sequence of length ``order + 1``.

        Returns
        -------
        ODEHierarchy
            Per-order results. ``sol[k].ode`` is the :math:`O(\varepsilon^k)`
            ODE, ``sol[k].general_solution`` carries the free constants,
            ``sol[k].particular_solution`` fixes them from the conditions,
            ``sol[k].secular`` flags secular terms, and ``sol.expansion`` is
            the assembled series.

        Raises
        ------
        TypeError
            If ``order`` is not an integer.
        ValueError
            If ``order`` is negative.
        NoSmallParameterError
            If :math:`\varepsilon` does not appear in the equation.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1"])   # doctest: +SKIP
        >>> sol = eq.expand_regular(order=2)                     # doctest: +SKIP
        >>> sol[0].particular_solution                          # doctest: +SKIP
        exp(-t)
        >>> sol[1].particular_solution                          # doctest: +SKIP
        -exp(-t) + exp(-2*t)
        """
        self._validate_order(order, "expand_regular")
        from asymptotics.methods.regular_ode import expand_regular_ode
        return expand_regular_ode(self, order=order, gauge=gauge)

    def begin_expansion(self, order: int = 2):
        r"""
        Set up a step-by-step regular expansion without solving it.

        Performs the same substitution and power-collection as
        :meth:`expand_regular` — writing
        :math:`u=\sum_{k=0}^{N}u_k(t)\varepsilon^k` and extracting the
        :math:`O(\varepsilon^k)` equations symbolically — but stops before
        solving them. Control then returns to the caller, who solves the
        hierarchy one order at a time:

        - ``sol[k].solve()`` — attempt a SymPy solution (fails gracefully).
        - ``sol[k].set_solution(expr)`` — supply a solution manually, as a
          string or a SymPy expression.
        - ``sol.solve_all()`` — attempt every remaining order.

        This is useful when SymPy cannot close a step automatically, when you
        want to inspect or hand-solve an intermediate ODE, or for teaching
        the mechanics of the method.

        Parameters
        ----------
        order : int, optional
            Highest power of :math:`\varepsilon` to expand to. Default 2,
            producing equations for :math:`u_0,\dots,u_2`.

        Returns
        -------
        StepwiseHierarchy
            Container whose entries expose ``.ode``, ``.solve()``,
            ``.set_solution()`` and ``.is_solved``, plus ``sol.n_solved`` /
            ``sol.n_pending`` counters. ``sol.expansion`` becomes available
            once every order is solved.

        Raises
        ------
        TypeError
            If ``order`` is not an integer.
        ValueError
            If ``order`` is negative.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u**3", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1", "u'(0) = 0"])  # doctest: +SKIP
        >>> sol = eq.begin_expansion(order=2)          # doctest: +SKIP
        >>> sol.n_pending, sol.n_solved                # doctest: +SKIP
        (3, 0)
        >>> sol[0].solve()                             # try SymPy      # doctest: +SKIP
        >>> sol[0].is_solved                           # doctest: +SKIP
        True
        >>> sol[1].set_solution("cos(t)/32 - cos(t)**3/32")  # or manually  # doctest: +SKIP
        >>> sol.solve_all()                            # attempt remaining  # doctest: +SKIP
        >>> sol.expansion                              # once all solved    # doctest: +SKIP
        """
        self._validate_order(order, "begin_expansion")
        from asymptotics.methods.stepwise import begin_expansion_ode
        return begin_expansion_ode(self, order=order)

    def expand_boundary_layer(self, order: int = 0):
        r"""
        Solve this singularly perturbed BVP by matched asymptotic expansions.

        For a boundary-value problem in which the small parameter multiplies
        the highest derivative,

        .. math::

            \varepsilon\,u'' + p(x)\,u' + q(x)\,u = f(x),
            \qquad u(a)=\alpha,\ u(b)=\beta,

        the leading-order outer solution obtained by setting
        :math:`\varepsilon = 0` cannot satisfy both boundary conditions, so a
        thin **boundary layer** forms where :math:`u` varies rapidly. This
        method builds the standard leading-order matched approximation:

        1. **Outer solution** :math:`u_\text{out}(x)`: solve the reduced
           first-order problem :math:`p u' + q u = f` satisfying the boundary
           condition away from the layer.
        2. **Inner solution**: rescale with the stretched coordinate
           :math:`\xi = (x-x_0)/\varepsilon` about the layer location
           :math:`x_0` and solve the boundary-layer equation.
        3. **Matching + composite**: form the additive composite
           :math:`u \approx u_\text{out} + u_\text{inner} - u_\text{match}`,
           uniformly valid across the domain.

        The layer sits at the left boundary when :math:`p>0` there and at the
        right boundary when :math:`p<0`; the location is detected
        automatically from the sign of :math:`p` at the boundaries.

        Parameters
        ----------
        order : int, optional
            Expansion order. Currently only ``order=0`` (leading-order
            matched approximation) is supported. Default 0.

        Returns
        -------
        BoundaryLayerHierarchy
            Exposes ``sol.layer_location`` (e.g. ``'x = 0'``), ``sol.outer``
            (the outer solution), and ``sol.expansion`` (the uniformly valid
            composite).

        Raises
        ------
        ValueError
            If the problem is an IVP rather than a BVP.
        NoSmallParameterError
            If :math:`\varepsilon` does not multiply the highest derivative.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("eps*u'' + u' + u", dependent="u", small_param="eps",
        ...          independent="x", conditions=["u(0) = 0", "u(1) = 1"])   # doctest: +SKIP
        >>> sol = eq.expand_boundary_layer()           # doctest: +SKIP
        >>> sol.layer_location                         # doctest: +SKIP
        'x = 0'
        >>> sol.outer                                  # doctest: +SKIP
        exp(1 - x)
        >>> sol.expansion                              # doctest: +SKIP
        (1 - exp(x*(eps - 1)/eps))*exp(1 - x)
        """
        self._validate_order(order, "expand_boundary_layer")
        from asymptotics.methods.boundary_layer import expand_boundary_layer
        return expand_boundary_layer(self, order=order)

    def expand_multiple_scales(self, order: int = 1):
        r"""
        Solve this oscillator ODE by the method of multiple scales.

        Treats the fast oscillation and the slow modulation of a weakly
        nonlinear oscillator as independent variables. Introduces the fast
        time :math:`T_0=t` and the slow time :math:`T_1=\varepsilon t`
        (and, at higher order, :math:`T_2=\varepsilon^2 t`, ...), so that

        .. math::

            u(t,\varepsilon) = u_0(T_0,T_1) + \varepsilon\,u_1(T_0,T_1)
            + \cdots,
            \qquad
            \frac{d}{dt} = \partial_{T_0} + \varepsilon\,\partial_{T_1}
            + \cdots.

        The leading order is a harmonic oscillation in :math:`T_0` whose
        amplitude and phase are undetermined functions of the slow time,
        written here as :math:`A(T_1)` and :math:`B(T_1)` (equivalently a
        slowly varying complex amplitude). At :math:`O(\varepsilon)` the
        forcing of :math:`u_1` contains resonant (secular) terms
        proportional to :math:`\cos T_0` and :math:`\sin T_0`; requiring
        their coefficients to vanish gives the **solvability conditions** —
        ODEs in the slow time for :math:`A` and :math:`B`. For example, the
        damped oscillator :math:`u''+u+\varepsilon u'=0` yields

        .. math::

            \frac{dA}{dT_1} = -\tfrac{1}{2}A, \qquad
            \frac{dB}{dT_1} = -\tfrac{1}{2}B,

        so :math:`A=e^{-T_1/2}` and the amplitude decays as
        :math:`e^{-\varepsilon t/2}`.

        Parameters
        ----------
        order : int, optional
            Number of :math:`\varepsilon` corrections (slow scales) to
            introduce. Default 1.

        Returns
        -------
        MultScalesHierarchy
            Exposes ``sol.T0``, ``sol.T1``, ``sol.omega_0``,
            ``sol.amplitude_A``, ``sol.amplitude_B`` (solved slow-time
            functions when available), ``sol.expansion_t`` (the expansion
            re-expressed in the physical time :math:`t`), and per-order
            ``sol[k].solvability_A`` / ``sol[k].solvability_B`` and
            ``sol[k].pde``.

        Raises
        ------
        ValueError
            If the ODE is not second order.
        NoSmallParameterError
            If :math:`\varepsilon` does not appear in the equation.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u'", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1", "u'(0) = 0"])  # doctest: +SKIP
        >>> sol = eq.expand_multiple_scales(order=1)   # doctest: +SKIP
        >>> sol.amplitude_A                            # doctest: +SKIP
        exp(-T_1/2)
        >>> sol.expansion_t                            # doctest: +SKIP
        exp(-eps*t/2)*cos(t)
        """
        self._validate_order(order, "expand_multiple_scales")
        from asymptotics.methods.multiple_scales import expand_multiple_scales
        return expand_multiple_scales(self, order=order)

    def expand_lindstedt(self, order: int = 2):
        r"""
        Solve this nonlinear oscillator by the Lindstedt–Poincaré method.

        Suitable for periodic solutions of weakly nonlinear oscillators of
        the form

        .. math::

            u'' + \omega_0^2\,u + \varepsilon\,f(u,u') = 0 .

        A naive regular expansion produces secular terms
        (:math:`t\sin\omega_0 t`) because the nonlinearity shifts the true
        frequency. Lindstedt–Poincaré removes them by **straining the time
        coordinate**: it introduces :math:`\tau = \omega(\varepsilon)\,t` and
        expands *both* the solution and the frequency in :math:`\varepsilon`,

        .. math::

            u(\tau) = \sum_{k=0}^{N} u_k(\tau)\,\varepsilon^k,
            \qquad
            \omega(\varepsilon) = \omega_0 + \omega_1\varepsilon
            + \omega_2\varepsilon^2 + \cdots .

        At each order the frequency correction :math:`\omega_k` is chosen to
        cancel the resonant forcing (the *secularity condition*), keeping
        every :math:`u_k` bounded and :math:`2\pi`-periodic in :math:`\tau`.
        The natural frequency :math:`\omega_0` is detected automatically from
        the leading-order equation. For the Duffing oscillator
        :math:`u''+u+\varepsilon u^3=0` with :math:`u(0)=1,\,u'(0)=0` this
        recovers the classical results :math:`\omega_1=\tfrac{3}{8}` and
        :math:`u_1=\tfrac{1}{32}\!\left(\cos 3\tau-\cos\tau\right)`.

        Parameters
        ----------
        order : int, optional
            Highest power of :math:`\varepsilon` to compute (inclusive).
            Default 2.

        Returns
        -------
        LindstedtHierarchy
            Exposes ``sol.omega_0``, ``sol.omega_expansion`` (the strained
            frequency :math:`\omega(\varepsilon)`), ``sol.expansion``
            (solution in :math:`\tau`), ``sol.expansion_t`` (in physical
            time), and per-order ``sol[k].omega_k_val``,
            ``sol[k].omega_k_sym`` and ``sol[k].secularity_condition``.

        Raises
        ------
        ValueError
            If the ODE is not second order.
        NoSmallParameterError
            If :math:`\varepsilon` does not appear in the equation.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u**3", dependent="u", small_param="eps",
        ...          independent="t", conditions=["u(0) = 1", "u'(0) = 0"])  # doctest: +SKIP
        >>> sol = eq.expand_lindstedt(order=2)         # doctest: +SKIP
        >>> sol[1].omega_k_val                         # doctest: +SKIP
        3/8
        >>> sol.omega_expansion                        # doctest: +SKIP
        -21*eps**2/256 + 3*eps/8 + 1
        >>> sol[1].particular_solution                 # doctest: +SKIP
        -cos(tau)/32 + cos(3*tau)/32
        """
        self._validate_order(order, "expand_lindstedt")
        from asymptotics.methods.lindstedt import expand_lindstedt
        return expand_lindstedt(self, order=order)


# ---------------------------------------------------------------------------
# Coupled algebraic systems:  f(x,y,...,eps) = 0,  g(x,y,...,eps) = 0, ...
# ---------------------------------------------------------------------------

class AlgebraicSystem:
    r"""
    A coupled system of algebraic perturbation equations.

    Represents :math:`n` simultaneous nonlinear equations

    .. math::

        f_i(x_1,\dots,x_n,\varepsilon) = 0, \qquad i = 1,\dots,n,

    in :math:`n` unknowns depending on a small parameter. Regular
    perturbation expands every unknown as a power series,

    .. math::

        x_j(\varepsilon) = \sum_{k\ge 0} x_{j,k}\,\varepsilon^k,

    substitutes into all equations, and collects powers of
    :math:`\varepsilon`. The leading order is the (generally nonlinear)
    system :math:`f_i(x_{1,0},\dots,x_{n,0},0)=0`; every higher order is a
    *linear* system in the new coefficients whose matrix is the Jacobian
    :math:`\partial f_i/\partial x_j` evaluated at the leading-order
    solution, solved order by order.

    All inputs are plain strings — no :func:`sympy.symbols` needed.

    Parameters
    ----------
    equations : list of str
        The equations, each understood as set equal to zero, e.g.
        ``["x**2 + eps*y - 1", "y**2 + eps*x - 1"]``.
    dependents : list of str
        Names of the unknowns, e.g. ``["x", "y"]``. Must have the same
        length as ``equations``.
    small_param : str
        Name of the small parameter, e.g. ``"eps"``.
    root_hint : dict, optional
        Leading-order solution branch to follow, e.g. ``{"x": 1, "y": -1}``.
        If ``None`` (default) the solver selects the real solution with the
        largest norm.

    Attributes
    ----------
    equations : list of sympy.Expr
        The parsed residuals.
    dependents : list of sympy.Symbol
        The unknown symbols.
    small_param : sympy.Symbol
        The small-parameter symbol.
    params : set of str
        Names of extra symbolic parameters detected across the equations.

    Raises
    ------
    ValueError
        If ``equations`` and ``dependents`` differ in length, or if any
        equation string cannot be parsed.

    See Also
    --------
    AlgebraicEquation : the single-equation case.

    Examples
    --------
    >>> from asymptotics import AlgebraicSystem
    >>> sys = AlgebraicSystem(
    ...     equations   = ["x**2 + eps*y - 1", "y**2 + eps*x - 1"],
    ...     dependents  = ["x", "y"],
    ...     small_param = "eps",
    ... )
    >>> sol = sys.expand_regular(order=3)
    >>> sol["x"].expansion
    eps**2/8 - eps/2 + 1
    >>> sol["y"][1].solution
    -1/2
    """

    def __init__(
        self,
        equations:   list,
        dependents:  list,
        small_param: str,
        root_hint:   dict = None,
    ):
        if len(equations) != len(dependents):
            raise ValueError(
                f"Number of equations ({len(equations)}) must match "
                f"number of dependents ({len(dependents)})."
            )

        from sympy import symbols, sympify

        # Create all symbols
        param_sym  = symbols(small_param)
        dep_syms   = [symbols(d) for d in dependents]
        ns         = {d: s for d, s in zip(dependents, dep_syms)}
        ns[small_param] = param_sym

        # Parse all equations
        parsed = []
        for i, eq_str in enumerate(equations):
            try:
                parsed.append(sympify(eq_str, locals=ns, convert_xor=False))
            except Exception as e:
                raise ValueError(
                    f"\n\n  Could not parse equation {i+1}: '{eq_str}'\n"
                    f"  SymPy error: {e}\n\n"
                    f"  Tips:\n"
                    f"    - Use ** for powers:  x**3 not x^3\n"
                    f"    - Use * for products: eps*x not eps·x\n"
                    f"    - Functions like cos(), sin(), exp(), log() work out of the box\n"
                ) from e

        self.equations        = parsed
        self.dependents       = dep_syms
        self.small_param      = param_sym
        self._dependent_names = dependents
        self._small_param_name = small_param
        self.root_hint        = root_hint

        # Detect symbolic parameters across all equations
        raw_params = set()
        for eq_str in equations:
            raw_params |= _detect_params(eq_str, '', '', small_param)
        # Remove dependent variable names
        for dep in dependents:
            raw_params.discard(dep)
        self.params = raw_params
        if raw_params:
            _params_warning(raw_params, method='eval', has_at=False)


    def __repr__(self):
        eqs = ", ".join(str(e) for e in self.equations)
        return f"AlgebraicSystem([{eqs}] = 0)"

    def expand_regular(self, order: int = 3):
        r"""
        Solve this coupled system by regular perturbation theory.

        Expands each unknown as
        :math:`x_j = \sum_{k=0}^{N} x_{j,k}\varepsilon^k` with
        :math:`N=\texttt{order}`, substitutes into every equation, and
        collects powers of :math:`\varepsilon`. The leading order solves the
        nonlinear system :math:`f_i(\mathbf{x}_0,0)=0`; each higher order
        :math:`k\ge 1` is the linear system

        .. math::

            J(\mathbf{x}_0)\,\mathbf{x}_k = \mathbf{b}_k,

        where :math:`J=[\partial f_i/\partial x_j]` is the Jacobian at the
        leading-order solution and :math:`\mathbf{b}_k` collects the known
        lower-order contributions.

        Parameters
        ----------
        order : int, optional
            Highest power of :math:`\varepsilon` to compute (inclusive).
            Default 3.

        Returns
        -------
        SystemHierarchy
            Indexable by variable name: ``sol["x"]`` is that variable's
            per-order hierarchy, ``sol["x"].expansion`` its assembled series,
            and ``sol["x"][k].solution`` the coefficient :math:`x_k`.
            ``sol.variables`` lists the variable names.

        Raises
        ------
        TypeError
            If ``order`` is not a non-negative integer.
        NoSmallParameterError
            If :math:`\varepsilon` appears in none of the equations.
        OnlyComplexRootsError
            If the leading-order system has no real solution to follow.

        Examples
        --------
        >>> from asymptotics import AlgebraicSystem
        >>> sys = AlgebraicSystem(
        ...     equations   = ["x**2 + eps*y - 1", "y**2 + eps*x - 1"],
        ...     dependents  = ["x", "y"],
        ...     small_param = "eps",
        ... )
        >>> sol = sys.expand_regular(order=3)
        >>> sol.variables
        ['x', 'y']
        >>> sol["x"][0].solution, sol["x"][1].solution
        (1, -1/2)
        >>> sol["x"].expansion
        eps**2/8 - eps/2 + 1
        """
        if not isinstance(order, int) or order < 0:
            raise TypeError(f"\n\n  order must be a non-negative integer, got: {order!r}\n")
        from asymptotics.methods.regular_algebraic_system import expand_regular_system
        return expand_regular_system(self, order=order)
