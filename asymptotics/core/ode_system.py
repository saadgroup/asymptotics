"""
asymptotics.core.ode_system
===========================
ODESystem — a coupled system of ODEs for perturbation expansion.

All inputs are plain strings — no symbols() needed.

Example
-------
>>> from asymptotics import ODESystem
>>> import io, contextlib
>>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
...     sys = ODESystem(
...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
...         small_param = "eps",
...         conditions  = ["u(0) = 1", "v(0) = 1"],
...     )
>>> sol = sys.expand_regular(order=2)
>>> sol["u"][0].particular_solution
exp(-t)
>>> sol["v"][1].particular_solution
-t*exp(-2*t)
"""

from __future__ import annotations
import re as _re
from sympy import (
    Symbol, Function, symbols, sympify, diff,
    series, expand, simplify, dsolve, solve,
    Add, Integer, Eq
)
from asymptotics.core.conditions import parse_and_validate_conditions, ConditionError
from asymptotics.core.problem import _INDEP_CANDIDATES, _MATH_NAMES


# ---------------------------------------------------------------------------
# Helper: parse ODE string for a system
# ---------------------------------------------------------------------------

def _detect_order(eq_str: str, dep: str) -> int:
    """Detect the order of *dep* in *eq_str* (up to 6th order)."""
    for n in range(6, 0, -1):
        if dep + "'" * n in eq_str:
            return n
    return 0


def _preprocess(eq_str: str, dep: str) -> str:
    """Replace prime notation with internal derivative symbols (up to 6th order)."""
    for n in range(6, 0, -1):
        primes = "'" * n
        token  = f"d{n}{dep}" if n > 1 else f"d{dep}"
        eq_str = eq_str.replace(dep + primes, token)
    return eq_str


def _infer_dependents_system(equations: list, conditions: list) -> tuple[list, list]:
    """
    Infer dependent variable names from conditions and match each equation
    to its owner dependent variable.

    Strategy
    --------
    1. Extract unique dependent names from condition strings in order of
       first appearance (skipping ``lim(...)`` conditions).
    2. For each inferred name, find the equation in which *that* variable
       has at least one derivative (prime notation).  If two variables both
       appear with primes in the same equation the pairing is ambiguous and
       a ``ValueError`` is raised.

    Returns
    -------
    (dependents, reordered_equations) : (list[str], list[str])
        ``reordered_equations[i]`` is the equation that belongs to
        ``dependents[i]``, regardless of the order they were supplied.
    """
    # Step 1 — collect unique dep names from conditions
    dep_names = []
    for cond in conditions:
        stripped = cond.strip()
        if stripped.lower().startswith('lim('):
            continue
        m = _re.match(r'^([a-zA-Z_]\w*)', stripped)
        if m:
            name = m.group(1)
            if name not in dep_names:
                dep_names.append(name)

    if not dep_names:
        raise ValueError(
            "\n\n  Could not infer dependent variables from conditions.\n"
            "  Specify them explicitly: ODESystem(..., dependents=['u', 'v'])\n"
        )

    # Step 2 — match each equation to its owner
    matched = {}   # dep_name -> equation string
    for eq in equations:
        owners = [d for d in dep_names if d + "'" in eq]
        if len(owners) == 0:
            raise ValueError(
                f"\n\n  Could not match equation '{eq}' to any dependent variable.\n"
                f"  No derivatives found for any of: {dep_names}\n"
                f"  Check that prime notation is used (e.g. u', u'').\n"
            )
        if len(owners) > 1:
            raise ValueError(
                f"\n\n  Equation '{eq}' has derivatives of multiple variables: {owners}.\n"
                f"  Cannot infer ownership unambiguously.\n"
                f"  Specify dependents explicitly: ODESystem(..., dependents={dep_names})\n"
            )
        dep = owners[0]
        if dep in matched:
            raise ValueError(
                f"\n\n  Multiple equations contain derivatives of '{dep}'.\n"
                f"  Specify dependents explicitly: ODESystem(..., dependents={dep_names})\n"
            )
        matched[dep] = eq

    # Check every dep has an equation
    missing = [d for d in dep_names if d not in matched]
    if missing:
        raise ValueError(
            f"\n\n  No equation found with a derivative of: {missing}\n"
            f"  Specify dependents explicitly: ODESystem(..., dependents={dep_names})\n"
        )

    reordered = [matched[d] for d in dep_names]
    return dep_names, reordered


def _infer_independent_system(equations: list, dependents: list,
                               small_param: str) -> tuple[str, bool]:
    """
    Infer the independent variable for an ODE system.

    Scans all equation strings for tokens in ``_INDEP_CANDIDATES``
    after excluding all dependent names, the small parameter, derivative
    notation, and known math function names.

    Returns
    -------
    (name, was_inferred) : (str, bool)
        ``was_inferred`` is True when the name came from scanning rather
        than being supplied by the caller.
    """
    combined = ' '.join(equations)
    tokens   = set(_re.findall(r'[a-zA-Z_]\w*', combined))

    exclude = set(dependents) | {small_param} | _MATH_NAMES
    # Also exclude derivative notation: du, d2u, ..., d6u for every dep
    for dep in dependents:
        exclude.add(f'd{dep}')
        for n in range(2, 7):
            exclude.add(f'd{n}{dep}')

    candidates = (tokens & _INDEP_CANDIDATES) - exclude

    if len(candidates) == 1:
        return candidates.pop(), True
    # Ambiguous or absent — fall back to 't' (most ODE systems are IVPs).
    # If 't' is itself a dependent name the user must supply independent= explicitly.
    fallback = 't'
    if fallback in set(dependents):
        raise ValueError(
            f"\n\n  Could not infer the independent variable: the fallback 't' "
            f"is already used as a dependent variable name.\n"
            f"  Please supply it explicitly: ODESystem(..., independent='...')\n"
        )
    return fallback, True


# ---------------------------------------------------------------------------
# ODESystem class
# ---------------------------------------------------------------------------

class ODESystem:
    r"""
    A coupled system of ODEs prepared for regular perturbation expansion.

    An :class:`ODESystem` holds :math:`N` coupled ordinary differential
    equations in the dependent variables :math:`u^{(1)}, \dots, u^{(N)}`,
    each written implicitly as :math:`F_i = 0`, together with a designated
    small parameter :math:`\varepsilon` and one initial/boundary condition
    per order of each variable's equation.  All inputs are plain strings —
    no :func:`sympy.symbols` calls are needed.  Use prime notation for
    derivatives: ``u'`` for :math:`du/dt`, ``u''`` for :math:`d^2u/dt^2`
    (up to sixth order).

    Regular perturbation seeks each unknown as a power series in
    :math:`\varepsilon`,

    .. math::

        u^{(i)}(t, \varepsilon) = \sum_{k=0}^{\infty}
            \varepsilon^{k}\, u^{(i)}_k(t),

    which, substituted into every :math:`F_i = 0` and collected power by
    power of :math:`\varepsilon`, yields a triangular family of ODEs solved
    order by order (see :meth:`expand_regular`).  The construction only
    validates and parses the problem; the expansion is performed lazily by
    :meth:`expand_regular`.

    Parameters
    ----------
    equations : list of str
        Each equation is understood as set equal to zero, e.g.
        ``["u' + u + eps*v", "v' + 2*v + eps*u**2"]``.  There must be
        exactly one equation per dependent variable, and each equation must
        contain at least one derivative of its owner variable.
    small_param : str
        Name of the small parameter, e.g. ``"eps"``.  It must appear in at
        least one equation.
    conditions : list of str
        Initial or boundary conditions for ALL variables, e.g.
        ``["u(0) = 1", "v(0) = 0"]``.  Exactly one condition is required per
        order of each variable's ODE (a first-order equation needs one
        condition, a second-order equation two, and so on).
    dependents : list of str, optional
        Names of the dependent variables, e.g. ``["u", "v"]``.  If omitted,
        they are inferred from the leading identifiers in *conditions*
        (e.g. ``"u(0) = 1"`` → ``"u"``) and each equation is matched to the
        variable whose derivative it contains; an informational message
        reporting the inferred names is printed.
    independent : str, optional
        Name of the independent variable, e.g. ``"t"``.  If omitted, it is
        inferred by scanning the equations for tokens in
        :math:`\{r, x, y, z, t\}`, falling back to ``'t'`` when ambiguous
        or absent.

    Raises
    ------
    ValueError
        If the number of equations does not match the number of dependent
        variables, if a variable has no derivative in its equation, if the
        dependent/independent variables cannot be inferred unambiguously, or
        if an equation fails to parse.
    ConditionError
        If a variable does not have exactly as many conditions as the order
        of its ODE, or a condition cannot be matched to a variable.

    See Also
    --------
    expand_regular : Solve the system order by order.
    asymptotics.ODE : Single-equation counterpart.

    Notes
    -----
    This solver targets *weakly* coupled systems in which the leading-order
    (:math:`\varepsilon^0`) operator is diagonal — i.e. inter-variable
    coupling enters only at :math:`O(\varepsilon)` or higher, as with the
    ``eps*v`` and ``eps*u**2`` terms below.  Systems whose equations remain
    coupled at :math:`O(1)` are rejected by :meth:`expand_regular`; see its
    ``Raises`` section.

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> import io, contextlib
    >>> # dependents and independent are inferred automatically; the
    >>> # constructor prints an informational banner (hidden here).
    >>> with contextlib.redirect_stdout(io.StringIO()):
    ...     sys = ODESystem(
    ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...         small_param = "eps",
    ...         conditions  = ["u(0) = 1", "v(0) = 1"],
    ...     )
    >>> sol = sys.expand_regular(order=2)
    >>> sol["u"][0].particular_solution
    exp(-t)
    >>> sol["v"][1].particular_solution
    -t*exp(-2*t)
    """

    def __init__(
        self,
        equations:   list,
        small_param: str,
        conditions:  list,
        dependents:  list = None,
        independent: str  = None,
    ):
        from sympy import sympify as _sympify

        # ------------------------------------------------------------------
        # Infer dependent variables if not supplied
        # ------------------------------------------------------------------
        deps_supplied = dependents is not None
        if not deps_supplied:
            dependents, equations = _infer_dependents_system(equations, conditions)
            print(
                f"  ℹ️  dependents = {dependents} (inferred from conditions)\n"
                f"     To override: ODESystem(..., dependents={dependents})"
            )

        if len(equations) != len(dependents):
            raise ValueError(
                f"\n\n  Number of equations ({len(equations)}) must match "
                f"number of dependents ({len(dependents)}).\n"
            )

        # ------------------------------------------------------------------
        # Infer independent variable if not supplied
        # ------------------------------------------------------------------
        indep_supplied = independent is not None
        if not indep_supplied:
            independent, _ = _infer_independent_system(equations, dependents,
                                                        small_param)
            print(
                f"  ℹ️  independent = '{independent}' (inferred from equations)\n"
                f"     To override: ODESystem(..., independent='{independent}')"
            )
        else:
            print(
                f"  ℹ️  independent = '{independent}' (supplied)"
            )

        self.dependent_names = dependents
        self._small_param_name = small_param
        self._independent_name = independent

        # Build symbols
        self.small_param = Symbol(small_param)
        self.independent = Symbol(independent)
        eps = self.small_param
        t   = self.independent

        # Detect ODE order per variable
        self.ode_orders = {}
        for dep, eq_str in zip(dependents, equations):
            order = _detect_order(eq_str, dep)
            if order == 0:
                raise ValueError(
                    f"\n\n  No derivatives of '{dep}' found in equation '{eq_str}'.\n"
                    f"  Use prime notation: {dep}' or {dep}''.\n"
                )
            self.ode_orders[dep] = order

        # Parse conditions — all together, then distribute per variable
        all_conds, _ = _parse_system_conditions(conditions, dependents)
        self.conditions = all_conds   # dict: {dep_name: [ParsedCondition, ...]}

        # Validate each variable has the right number of conditions
        for dep in dependents:
            n_conds  = len(self.conditions.get(dep, []))
            n_needed = self.ode_orders[dep]
            if n_conds != n_needed:
                raise ConditionError(
                    f"\n\n  Variable '{dep}' is order {n_needed} but has "
                    f"{n_conds} condition(s). Need exactly {n_needed}.\n"
                )

        # Build derivative symbols and parse equations
        self._deriv_syms = {}   # dep -> {order: symbol}
        self._dep_syms   = {}   # dep -> symbol
        self.equations   = {}   # dep -> SymPy expression

        ns = {small_param: eps, independent: t}

        for dep, eq_str in zip(dependents, equations):
            dep_sym  = Symbol(dep)
            self._dep_syms[dep] = dep_sym
            ns[dep] = dep_sym

            deriv_dict = {}
            for k in range(1, self.ode_orders[dep] + 1):
                prefix = "d" * k if k == 1 else f"d{k}"
                dsym   = Symbol(f"{prefix}{dep}")
                deriv_dict[k] = dsym
                ns[f"{prefix}{dep}"] = dsym
            self._deriv_syms[dep] = deriv_dict

        # Parse all equations (after full ns is built)
        for dep, eq_str in zip(dependents, equations):
            processed = _preprocess(eq_str, dep)
            for dep2 in dependents:
                processed = _preprocess(processed, dep2)
            try:
                self.equations[dep] = sympify(processed, locals=ns, convert_xor=False)
            except Exception as e:
                raise ValueError(
                    f"\n\n  Could not parse equation for '{dep}': '{eq_str}'\n"
                    f"  Error: {e}\n"
                ) from e

    def expand_regular(self, order: int = 2):
        r"""
        Apply regular perturbation theory to this coupled ODE system.

        Substitutes the power-series ansatz
        :math:`u^{(i)} = \sum_k \varepsilon^k u^{(i)}_k(t)` into every
        equation, collects like powers of :math:`\varepsilon`, and solves the
        resulting hierarchy order by order.  Because the leading-order
        operator is diagonal, the order-:math:`k` equations decouple into
        :math:`N` independent scalar ODEs

        .. math::

            \mathcal{L}_i\, u^{(i)}_k(t)
                = g^{(i)}_k\!\bigl(u^{(j)}_0, \dots, u^{(j)}_{k-1}\bigr),

        each forced only by already-known lower-order solutions.  The
        leading order (:math:`k=0`) uses the supplied initial/boundary
        conditions; every higher order is solved with homogeneous
        conditions so that the conditions are met exactly at :math:`O(1)`.

        Parameters
        ----------
        order : int
            Highest power of :math:`\varepsilon` to compute. Default 2.
            Must be a non-negative integer.

        Returns
        -------
        ODESystemHierarchy
            The solved hierarchy, indexable by variable name
            (``sol["u"]``) and by order (``sol[k]``).

        Raises
        ------
        TypeError
            If *order* is not a non-negative integer.
        NoSmallParameterError
            If :math:`\varepsilon` appears in none of the equations.
        RuntimeError
            If, after substituting all known lower-order solutions, an
            order-:math:`k` equation still references another variable's
            unknown — i.e. the system is coupled at :math:`O(1)` and does
            not decouple as this solver requires.

        Examples
        --------
        >>> from asymptotics import ODESystem
        >>> import io, contextlib
        >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
        ...     sys = ODESystem(
        ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
        ...         dependents  = ["u", "v"], independent = "t",
        ...         small_param = "eps",
        ...         conditions  = ["u(0) = 1", "v(0) = 1"],
        ...     )
        >>> sol = sys.expand_regular(order=2)
        >>> len(sol)                                # orders 0, 1, 2
        3
        """
        if not isinstance(order, int) or order < 0:
            raise TypeError(
                f"\n\n  order must be a non-negative integer, got: {order!r}\n"
            )
        from asymptotics.methods.regular_ode_system import expand_regular_ode_system
        return expand_regular_ode_system(self, order=order)


# ---------------------------------------------------------------------------
# Condition parser for systems — distributes conditions to each variable
# ---------------------------------------------------------------------------

def _parse_system_conditions(conditions: list, dependents: list) -> tuple:
    """
    Parse a flat list of condition strings and distribute to each variable.

    Returns
    -------
    (dict: {dep_name: [ParsedCondition]}, str problem_type)
    """
    from asymptotics.core.conditions import parse_condition

    result = {dep: [] for dep in dependents}

    for cond_str in conditions:
        # Detect which variable this condition belongs to
        matched = False
        for dep in dependents:
            try:
                parsed = parse_condition(cond_str, dep)
                result[dep].append(parsed)
                matched = True
                break
            except ConditionError:
                continue

        if not matched:
            raise ConditionError(
                f"\n\n  Could not match condition '{cond_str}' to any variable.\n"
                f"  Variables are: {dependents}\n"
                f"  Condition must start with one of: "
                f"{', '.join(dep + '(' for dep in dependents)}\n"
            )

    return result, "ivp"
