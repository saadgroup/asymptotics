"""
asymptotics.core.ode_system
========================
ODESystem — a coupled system of ODEs for perturbation expansion.

All inputs are plain strings — no symbols() needed.

Example
-------
>>> sys = ODESystem(
...     equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
...     dependents  = ["u", "v"],
...     small_param = "eps",
...     independent = "t",
...     conditions  = ["u(0) = 1", "v(0) = 1"],
... )
>>> sol = sys.expand_regular(order=2)
>>> sol["u"].composite
>>> sol["v"][1].particular_solution
>>> sol.show()
"""

from __future__ import annotations
from sympy import (
    Symbol, Function, symbols, sympify, diff,
    series, expand, simplify, dsolve, solve,
    Add, Integer, Eq
)
from asymptotics.core.conditions import parse_and_validate_conditions, ConditionError


# ---------------------------------------------------------------------------
# Helper: parse ODE string for a system
# ---------------------------------------------------------------------------

def _detect_order(eq_str: str, dep: str) -> int:
    if dep + "''" in eq_str:
        return 2
    elif dep + "'" in eq_str:
        return 1
    return 0


def _preprocess(eq_str: str, dep: str) -> str:
    eq_str = eq_str.replace(dep + "''", f"d2{dep}")
    eq_str = eq_str.replace(dep + "'",  f"d{dep}")
    return eq_str


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
    dependents : list of str
        Names of the dependent variables, e.g. ["u", "v"].
    small_param : str
        Name of the small parameter, e.g. "eps".
    independent : str
        Name of the independent variable, e.g. "t".
    conditions : list of str
        Initial or boundary conditions for ALL variables, e.g.:
          ["u(0) = 1", "v(0) = 0"]
        One condition per variable per order of that variable's ODE.

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> sys = ODESystem(
    ...     equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...     dependents  = ["u", "v"],
    ...     small_param = "eps",
    ...     independent = "t",
    ...     conditions  = ["u(0) = 1", "v(0) = 1"],
    ... )
    >>> sol = sys.expand_regular(order=2)
    >>> sol.show()
    >>> sol["u"].composite
    >>> sol["v"][1].particular_solution
    """

    def __init__(
        self,
        equations:   list,
        dependents:  list,
        small_param: str,
        independent: str,
        conditions:  list,
    ):
        from sympy import sympify as _sympify

        if len(equations) != len(dependents):
            raise ValueError(
                f"\n\n  Number of equations ({len(equations)}) must match "
                f"number of dependents ({len(dependents)}).\n"
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
