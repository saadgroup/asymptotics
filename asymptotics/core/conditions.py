"""
asymptotics.core.conditions
========================
Parsing and validation of ODE boundary/initial conditions.

Supports string conditions like:
    "u(0) = 1"       — value of u at t=0
    "u'(0) = 0"      — value of u' at t=0
    "u''(pi) = 2"    — value of u'' at t=pi
    "u(1) = sqrt(2)" — symbolic values
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple
from sympy import sympify, Symbol, pi, E, sqrt, Rational, nsimplify


class ParsedCondition:
    """A single parsed boundary/initial condition."""

    def __init__(self, var: str, deriv_order: int, point, value):
        self.var         = var          # variable name e.g. 'u'
        self.deriv_order = deriv_order  # 0 = u, 1 = u', 2 = u''
        self.point       = point        # SymPy expression, e.g. 0, pi
        self.value       = value        # SymPy expression, e.g. 1, sqrt(2)

    def __repr__(self):
        primes = "'" * self.deriv_order
        return f"{self.var}{primes}({self.point}) = {self.value}"


class ConditionError(Exception):
    """Raised when conditions are missing, over-specified, or inconsistent."""
    pass


def parse_condition(cond_str: str, dep_name: str) -> ParsedCondition:
    """
    Parse a single condition string.

    Parameters
    ----------
    cond_str : str
        e.g. "u(0) = 1", "u'(0) = 0", "u''(pi) = sqrt(2)"
    dep_name : str
        Name of the dependent variable, e.g. "u"

    Returns
    -------
    ParsedCondition

    Raises
    ------
    ConditionError
        If the string cannot be parsed or refers to the wrong variable.
    """
    s = cond_str.strip().replace(' ', '')

    m = re.match(r"^([a-zA-Z_]\w*)('*)\(([^)]+)\)=(.+)$", s)
    if not m:
        raise ConditionError(
            f"\n\n  Cannot parse condition: '{cond_str}'\n\n"
            f"  Expected format:\n"
            f"    u(0) = 1        — value of u at t=0\n"
            f"    u'(0) = 0       — value of u' at t=0\n"
            f"    u''(pi) = 2     — value of u'' at t=π\n"
            f"    u(1) = sqrt(2)  — symbolic values are fine\n"
        )

    var_name    = m.group(1)
    primes      = m.group(2)
    point_str   = m.group(3)
    value_str   = m.group(4)
    deriv_order = len(primes)

    # Check variable name matches
    if var_name != dep_name:
        raise ConditionError(
            f"\n\n  Condition '{cond_str}' refers to variable '{var_name}'\n"
            f"  but the dependent variable is '{dep_name}'.\n\n"
            f"  Did you mean '{dep_name}({point_str}) = {value_str}'?\n"
        )

    # Parse point and value as SymPy expressions
    try:
        point = sympify(point_str)
    except Exception:
        raise ConditionError(
            f"\n\n  Cannot parse point '{point_str}' in condition '{cond_str}'.\n"
            f"  Use numbers (0, 1, 0.5) or symbolic constants (pi, E).\n"
        )

    try:
        value = sympify(value_str)
        # Convert floats to rationals so dsolve works reliably
        # e.g. 0.9 -> 9/10, 1.5 -> 3/2
        if value.is_Float:
            value = nsimplify(value, rational=True)
    except Exception:
        raise ConditionError(
            f"\n\n  Cannot parse value '{value_str}' in condition '{cond_str}'.\n"
            f"  Use numbers (0, 1) or expressions (sqrt(2), pi/4).\n"
        )

    return ParsedCondition(var_name, deriv_order, point, value)


def parse_and_validate_conditions(
    conditions: List[str],
    dep_name:   str,
    ode_order:  int,
) -> Tuple[List[ParsedCondition], str]:
    """
    Parse all condition strings and validate them for an ODE of given order.

    Parameters
    ----------
    conditions : list of str
    dep_name : str
    ode_order : int

    Returns
    -------
    (parsed_conditions, problem_type)
        problem_type is "ivp" or "bvp"

    Raises
    ------
    ConditionError
        For any validation failure with a clear message.
    """
    if len(conditions) != ode_order:
        raise ConditionError(
            f"\n\n  {len(conditions)} condition(s) provided but the ODE is "
            f"order {ode_order} — need exactly {ode_order}.\n\n"
            f"  {'Hint: for a 2nd-order ODE you need 2 conditions, e.g.:' if ode_order == 2 else ''}\n"
            f"  {'  IVP: [\"u(0) = 1\", \"u' + chr(39) + '(0) = 0\"]' if ode_order == 2 else ''}\n"
            f"  {'  BVP: [\"u(0) = 0\", \"u(1) = 1\"]' if ode_order == 2 else ''}\n"
        )

    # Parse all conditions
    parsed = [parse_condition(c, dep_name) for c in conditions]

    # Find distinct points
    points = list(dict.fromkeys(p.point for p in parsed))  # preserves order, deduplicates

    # Check for more than 2 distinct points
    if len(points) > 2:
        pts_str = ", ".join(f"t={p}" for p in points)
        raise ConditionError(
            f"\n\n  Conditions at {len(points)} distinct points ({pts_str}).\n"
            f"  Regular perturbation supports at most 2 boundary points.\n"
        )

    # Check for conflicting conditions (same point, same derivative order, different values)
    seen = {}
    for p in parsed:
        key = (p.point, p.deriv_order)
        if key in seen:
            if seen[key] != p.value:
                primes = "'" * p.deriv_order
                raise ConditionError(
                    f"\n\n  Conflicting conditions at t={p.point}:\n"
                    f"    {dep_name}{primes}({p.point}) = {seen[key]}\n"
                    f"    {dep_name}{primes}({p.point}) = {p.value}\n"
                )
        seen[key] = p.value

    # Check for duplicate (same point + same derivative order = redundant)
    if len(seen) < len(parsed):
        raise ConditionError(
            f"\n\n  Duplicate conditions detected — two conditions specify the same "
            f"derivative at the same point.\n"
        )

    # Determine IVP vs BVP
    if len(points) == 1:
        problem_type = "ivp"
    else:
        problem_type = "bvp"

    # For BVP: validate that together the conditions fully constrain the system
    # (each point must have the right number of conditions summing to ode_order)
    if problem_type == "bvp":
        counts = {pt: sum(1 for p in parsed if p.point == pt) for pt in points}
        total  = sum(counts.values())
        if total != ode_order:
            raise ConditionError(
                f"\n\n  BVP conditions don't add up to ODE order {ode_order}.\n"
                f"  Got: {dict(counts)}\n"
            )

    return parsed, problem_type
