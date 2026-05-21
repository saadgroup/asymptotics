"""
asymptotics.core.ode_system
========================
ODESystem — a coupled system of ODEs for perturbation expansion.

All inputs are plain strings — no symbols() needed.

Example
-------
>>> sys = ODESystem(
...     equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
...     small_param = "eps",
...     conditions  = ["u(0) = 1", "v(0) = 1"],
... )
>>> sol = sys.expand_regular(order=2)
>>> sol["u"].expansion
>>> sol["v"][1].particular_solution
>>> sol.show()
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
    """
    A coupled system of ODEs for perturbation expansion.

    All inputs are plain strings — no symbols() needed.
    Use prime notation for derivatives: u' for du/dt, u'' for d²u/dt².

    Parameters
    ----------
    equations : list of str
        Each equation set equal to zero, e.g.:
          ["u' + u + eps*v", "v' + 2*v + eps*u**2"]
    small_param : str
        Name of the small parameter, e.g. "eps".
    conditions : list of str
        Initial or boundary conditions for ALL variables, e.g.:
          ["u(0) = 1", "v(0) = 0"]
        One condition per variable per order of that variable's ODE.
    dependents : list of str, optional
        Names of the dependent variables, e.g. ["u", "v"].
        If omitted, inferred from the leading identifiers in *conditions*
        (e.g. ``"u(0) = 1"`` → ``"u"``).
    independent : str, optional
        Name of the independent variable, e.g. "t".
        If omitted, inferred by scanning the equations for tokens in
        {r, x, y, z, t}; falls back to 't' when ambiguous or absent.

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> # both dependents and independent inferred automatically
    >>> sys = ODESystem(
    ...     equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...     small_param = "eps",
    ...     conditions  = ["u(0) = 1", "v(0) = 1"],
    ... )
    >>> sol = sys.expand_regular(order=2)
    >>> sol.show()
    >>> sol["u"].expansion
    >>> sol["v"][1].particular_solution
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
        """
        Apply regular perturbation theory to this coupled ODE system.

        Parameters
        ----------
        order : int
            Highest power of ε to compute. Default 2.

        Returns
        -------
        ODESystemHierarchy
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
