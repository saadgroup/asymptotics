"""
asymptotics.core.problem
====================
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
    """
    Base class for all perturbation equations.

    Holds the parsed SymPy equation and symbols.
    Expansion methods are defined only on subclasses that implement them.
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
    """
    A perturbation equation of the form  f(x, ε) = 0.

    All inputs are plain strings — no symbols() needed.

    Parameters
    ----------
    equation : str
        The equation set equal to zero.
        Use ** for powers, * for products.
        Standard functions (cos, sin, exp, log, tan, sqrt, ...) work out of the box.
    dependent : str
        Name of the unknown variable, e.g. 'x'.
    small_param : str
        Name of the small parameter, e.g. 'eps'.
    root_hint : float or int, optional
        Leading-order root to follow. If None the solver picks the
        largest real root of the O(1) equation.

    Supported methods
    -----------------
    expand_regular(order)

    Examples
    --------
    >>> from asymptotics import AlgebraicEquation
    >>> eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
    >>> sol = eq.expand_regular(order=3)
    >>> sol.show()

    >>> # Transcendental equations work too
    >>> eq  = AlgebraicEquation("tan(x) - 1 - eps*x**2", dependent="x", small_param="eps")
    >>> sol = eq.expand_regular(order=2)
    >>> sol.show()
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
        """
        Apply regular perturbation theory to this algebraic equation.

        Parameters
        ----------
        order : int
            Highest power of ε to compute (inclusive). Default 3.
        root_index : int
            Which real root of the O(1) equation to follow (default 0 =
            largest real root). Ignored if root_hint is set.
        gauge : str, list of str, or None
            Non-standard asymptotic gauge sequence.
            - None (default) → standard {1, ε, ε², ...}
            - str  → pattern for geometric inference, e.g. 'sqrt(eps)'
                     generates {1, √ε, ε, ε^(3/2), ...}
            - list → explicit sequence of length order+1, e.g.
                     ['1', 'eps*log(eps)', 'eps']

        Returns
        -------
        OrderHierarchy

        Raises
        ------
        NoSmallParameterError
        NoLeadingOrderSolutionError
        OnlyComplexRootsError
        NoHigherOrderSolutionError
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
    """
    A perturbation ODE of the form  F(u, u', u'', t, ε) = 0.

    All inputs are plain strings — no symbols() needed.
    Use prime notation for derivatives: u' for du/dt, u'' for d²u/dt².

    Parameters
    ----------
    equation : str
        The ODE set equal to zero.
        e.g. "u'' + u + eps*u**3"  or  "u' + eps*u**2 + u"
    dependent : str
        Name of the dependent variable, e.g. 'u'.
    small_param : str
        Name of the small parameter, e.g. 'eps'.
    independent : str
        Name of the independent variable, e.g. 't'.
    conditions : list of str
        Boundary or initial conditions, e.g.:
          IVP: ["u(0) = 1", "u'(0) = 0"]
          BVP: ["u(0) = 0", "u(1) = 1"]
        Number of conditions must equal the ODE order.

    Supported methods
    -----------------
    expand_regular(order)

    Examples
    --------
    >>> from asymptotics import ODE
    >>> # IVP
    >>> eq = ODE(
    ...     "u'' + u + eps*u**3",
    ...     dependent="u", small_param="eps", independent="t",
    ...     conditions=["u(0) = 1", "u'(0) = 0"],
    ... )
    >>> sol = eq.expand_regular(order=2)
    >>> sol.show()

    >>> # BVP
    >>> eq = ODE(
    ...     "u'' + eps*u**3",
    ...     dependent="u", small_param="eps", independent="t",
    ...     conditions=["u(0) = 0", "u(1) = 1"],
    ... )
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
        """
        Apply regular perturbation theory to this ODE.

        Parameters
        ----------
        order : int
            Highest power of ε to compute (inclusive). Default 2.
        gauge : str, list of str, or None
            Non-standard asymptotic gauge sequence.
            - None (default) → standard {1, ε, ε², ...}
            - str  → pattern for geometric inference, e.g. 'sqrt(eps)'
                     generates {1, √ε, ε, ε^(3/2), ...}
            - list → explicit sequence of length order+1

        Returns
        -------
        ODEHierarchy
        """
        self._validate_order(order, "expand_regular")
        from asymptotics.methods.regular_ode import expand_regular_ode
        return expand_regular_ode(self, order=order, gauge=gauge)

    def begin_expansion(self, order: int = 2):
        """
        Set up a step-by-step perturbation expansion without solving.

        The order-by-order equations are extracted symbolically immediately.
        Solutions are obtained one at a time via:
            sol[k].solve()           — try SymPy
            sol[k].set_solution(expr) — provide manually

        Parameters
        ----------
        order : int
            Highest power of eps to expand to. Default 2.

        Returns
        -------
        StepwiseHierarchy

        Examples
        --------
        >>> sol = eq.begin_expansion(order=2)
        >>> sol.show()                          # see all equations
        >>>
        >>> sol[0].solve()                      # try SymPy
        >>> sol[0].set_solution("4*eta - 4*eta**2")  # or manually
        >>>
        >>> sol[1].solve()
        >>> sol.solve_all()                     # try all remaining
        >>>
        >>> sol.expansion                       # available once all solved
        >>> sol.eval(eps=0.1, at=eta_vals)
        """
        self._validate_order(order, "begin_expansion")
        from asymptotics.methods.stepwise import begin_expansion_ode
        return begin_expansion_ode(self, order=order)

    def expand_boundary_layer(self, order: int = 0):
        """
        Apply matched asymptotic expansions to this singular perturbation BVP.

        Valid for equations of the form:
            eps*u'' + p(x)*u' + q(x)*u = f(x)
        with BVP conditions. The layer location is detected automatically
        from the sign of p(x) at the boundaries.

        Parameters
        ----------
        order : int
            Currently only order=0 (leading order) is supported.

        Returns
        -------
        BoundaryLayerHierarchy
        """
        self._validate_order(order, "expand_boundary_layer")
        from asymptotics.methods.boundary_layer import expand_boundary_layer
        return expand_boundary_layer(self, order=order)

    def expand_multiple_scales(self, order: int = 1):
        """
        Apply the method of multiple scales to this oscillator ODE.

        Introduces fast time T_0=t and slow time T_1=ε·t.
        At each order, eliminates secular terms via solvability conditions
        — ODEs for the amplitude and phase in slow time.

        Parameters
        ----------
        order : int
            Number of epsilon corrections (default 1).

        Returns
        -------
        MultScalesHierarchy
        """
        self._validate_order(order, "expand_multiple_scales")
        from asymptotics.methods.multiple_scales import expand_multiple_scales
        return expand_multiple_scales(self, order=order)

    def expand_lindstedt(self, order: int = 2):
        """
        Apply the Lindstedt–Poincaré method to this nonlinear oscillator.

        Eliminates secular terms by straining the time coordinate.
        Valid for equations of the form:
            u'' + ω₀²·u + ε·f(u, u') = 0

        The natural frequency ω₀ is detected automatically from the O(1) equation.

        Parameters
        ----------
        order : int
            Highest power of ε to compute. Default 2.

        Returns
        -------
        LindstedtHierarchy
        """
        self._validate_order(order, "expand_lindstedt")
        from asymptotics.methods.lindstedt import expand_lindstedt
        return expand_lindstedt(self, order=order)


# ---------------------------------------------------------------------------
# Coupled algebraic systems:  f(x,y,...,eps) = 0,  g(x,y,...,eps) = 0, ...
# ---------------------------------------------------------------------------

class AlgebraicSystem:
    """
    A system of coupled algebraic perturbation equations.

    All inputs are plain strings — no symbols() needed.

    Parameters
    ----------
    equations : list of str
        The equations, each set equal to zero.
        e.g. ["x**2 + eps*y - 1", "y**2 + eps*x - 1"]
    dependents : list of str
        Names of the unknowns, e.g. ["x", "y"].
        Must have the same length as equations.
    small_param : str
        Name of the small parameter, e.g. "eps".
    root_hint : dict, optional
        Leading-order solution to follow, e.g. {"x": 1, "y": -1}.
        If None, the solver picks the real solution with largest norm.

    Supported methods
    -----------------
    expand_regular(order)

    Examples
    --------
    >>> from asymptotics import AlgebraicSystem
    >>> sys = AlgebraicSystem(
    ...     equations   = ["x**2 + eps*y - 1", "y**2 + eps*x - 1"],
    ...     dependents  = ["x", "y"],
    ...     small_param = "eps",
    ... )
    >>> sol = sys.expand_regular(order=3)
    >>> sol.show()
    >>> sol["x"].expansion
    >>> sol["y"][1].solution
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
        """
        Apply regular perturbation theory to this coupled system.

        Parameters
        ----------
        order : int
            Highest power of ε to compute (inclusive). Default 3.

        Returns
        -------
        SystemHierarchy
        """
        if not isinstance(order, int) or order < 0:
            raise TypeError(f"\n\n  order must be a non-negative integer, got: {order!r}\n")
        from asymptotics.methods.regular_algebraic_system import expand_regular_system
        return expand_regular_system(self, order=order)
