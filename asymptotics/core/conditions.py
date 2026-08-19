"""
asymptotics.core.conditions
========================
Parsing and validation of ODE boundary/initial conditions.

Supports string conditions like:
    "u(0) = 1"                    — value of u at t=0
    "u'(0) = 0"                   — value of u' at t=0
    "u''(pi) = 2"                 — value of u'' at t=pi
    "u(1) = sqrt(2)"              — symbolic values
    "lim(sqrt(t)*u'', t, 0) = 0" — limit condition at singular point
"""

from __future__ import annotations
import re
from typing import List, Tuple
from sympy import sympify, Symbol, pi, E, sqrt, Rational, nsimplify


class ParsedCondition:
    """A standard point boundary/initial condition."""

    def __init__(self, var: str, deriv_order: int, point, value):
        self.var         = var
        self.deriv_order = deriv_order  # 0=u, 1=u', 2=u'', ...
        self.point       = point
        self.value       = value
        self.is_limit    = False

    def __repr__(self):
        primes = "'" * self.deriv_order
        return f"{self.var}{primes}({self.point}) = {self.value}"


class LimitCondition:
    """
    A limit boundary condition of the form:
        lim(expr, var, point) = value

    where expr may contain the dependent variable and its derivatives.

    Attributes
    ----------
    expr_str : str   — the expression string inside lim(...)
    var_str  : str   — the limit variable name
    point    : sympy — the limit point
    value    : sympy — the value the limit must equal
    is_limit : bool  — always True
    deriv_order : int — -1 (not a standard point condition)
    """

    def __init__(self, expr_str: str, var_str: str, point, value):
        self.expr_str    = expr_str
        self.var_str     = var_str
        self.point       = point
        self.value       = value
        self.is_limit    = True
        self.deriv_order = -1   # sentinel: not a point condition

    def __repr__(self):
        return f"lim({self.expr_str}, {self.var_str}, {self.point}) = {self.value}"


class ConditionError(Exception):
    """
    Invalid initial/boundary conditions for an ODE problem.

    Raised while parsing or validating the ``conditions`` supplied to an
    :class:`~asymptotics.ODE`.  Typical triggers are:

    - a condition string that cannot be parsed (e.g. ``"u[0] = 1"`` instead of
      ``"u(0) = 1"``, or a malformed ``lim(...)`` limit condition);
    - a condition that names a variable other than the dependent one;
    - the wrong *number* of conditions for the ODE order (an order-:math:`n`
      ODE needs exactly :math:`n` conditions);
    - conditions at more than two distinct points (regular perturbation
      supports at most two boundary points); or
    - conflicting conditions that assign different values to the same
      derivative at the same point.

    The exception message describes the specific problem and the expected
    condition formats.
    """
    pass


def _parse_limit_condition(cond_str: str, dep_name: str) -> LimitCondition:
    """
    Parse a limit condition string: lim(expr, var, point) = value

    Example: "lim(sqrt(2*eta)*F'', eta, 0) = 0"
    """
    s = cond_str.strip()

    # Match: lim(expr, var, point) = value
    # expr may contain commas inside functions like sqrt(...), so we need
    # to find the outermost structure manually
    if not s.lower().startswith('lim('):
        raise ConditionError(f"Not a limit condition: '{cond_str}'")

    # Find matching closing paren
    depth = 0
    lim_end = -1
    for i, ch in enumerate(s[3:], 3):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                lim_end = i
                break

    if lim_end == -1:
        raise ConditionError(
            f"\n\n  Cannot parse limit condition: '{cond_str}'\n"
            f"  Expected: lim(expr, var, point) = value\n"
        )

    inner    = s[4:lim_end]          # everything inside lim(...)
    rest     = s[lim_end+1:].strip() # should be = value

    if not rest.startswith('='):
        raise ConditionError(
            f"\n\n  Cannot parse limit condition: '{cond_str}'\n"
            f"  Missing '= value' after lim(...)\n"
        )

    value_str = rest[1:].strip()

    # Split inner by commas — but only top-level commas
    parts = []
    depth = 0
    current = []
    for ch in inner:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current).strip())

    if len(parts) != 3:
        raise ConditionError(
            f"\n\n  Cannot parse limit condition: '{cond_str}'\n"
            f"  Expected: lim(expr, var, point) = value\n"
            f"  Got {len(parts)} arguments inside lim(): {parts}\n"
        )

    expr_str  = parts[0]
    var_str   = parts[1].strip()
    point_str = parts[2].strip()

    try:
        point = sympify(point_str)
    except Exception:
        raise ConditionError(
            f"\n\n  Cannot parse limit point '{point_str}' in: '{cond_str}'\n"
        )

    try:
        value = sympify(value_str)
        if hasattr(value, 'is_Float') and value.is_Float:
            value = nsimplify(value, rational=True)
    except Exception:
        raise ConditionError(
            f"\n\n  Cannot parse limit value '{value_str}' in: '{cond_str}'\n"
        )

    return LimitCondition(expr_str, var_str, point, value)


def parse_condition(cond_str: str, dep_name: str):
    """
    Parse a single condition string — either a standard point condition
    or a limit condition.

    Returns ParsedCondition or LimitCondition.
    """
    s = cond_str.strip()

    # Check for limit condition
    if s.lower().startswith('lim('):
        return _parse_limit_condition(s, dep_name)

    # Standard point condition
    s_nospace = s.replace(' ', '')
    m = re.match(r"^([a-zA-Z_]\w*)('*)\(([^)]+)\)=(.+)$", s_nospace)
    if not m:
        raise ConditionError(
            f"\n\n  Cannot parse condition: '{cond_str}'\n\n"
            f"  Expected formats:\n"
            f"    u(0) = 1                    — point condition\n"
            f"    u'(0) = 0                   — derivative condition\n"
            f"    lim(sqrt(t)*u'', t, 0) = 0  — limit condition\n"
        )

    var_name    = m.group(1)
    primes      = m.group(2)
    point_str   = m.group(3)
    value_str   = m.group(4)
    deriv_order = len(primes)

    if var_name != dep_name:
        raise ConditionError(
            f"\n\n  Condition '{cond_str}' refers to variable '{var_name}'\n"
            f"  but the dependent variable is '{dep_name}'.\n"
        )

    try:
        point = sympify(point_str)
    except Exception:
        raise ConditionError(
            f"\n\n  Cannot parse point '{point_str}' in condition '{cond_str}'.\n"
        )

    try:
        value = sympify(value_str)
        if hasattr(value, 'is_Float') and value.is_Float:
            value = nsimplify(value, rational=True)
    except Exception:
        raise ConditionError(
            f"\n\n  Cannot parse value '{value_str}' in condition '{cond_str}'.\n"
        )

    return ParsedCondition(var_name, deriv_order, point, value)


def parse_and_validate_conditions(
    conditions: List[str],
    dep_name:   str,
    ode_order:  int,
) -> Tuple[list, str]:
    """
    Parse all condition strings and validate them for an ODE of given order.

    Limit conditions count toward the total but not toward boundary points.

    Returns (parsed_conditions, problem_type)
    """
    if len(conditions) != ode_order:
        _hint = (
            "\n  Hint: for a 2nd-order ODE you need 2 conditions, e.g.:"
            "\n    IVP: [\"u(0) = 1\", \"u'(0) = 0\"]"
            "\n    BVP: [\"u(0) = 0\", \"u(1) = 1\"]"
        ) if ode_order == 2 else ""
        raise ConditionError(
            f"\n\n  {len(conditions)} condition(s) provided but the ODE is "
            f"order {ode_order} — need exactly {ode_order}.\n"
            f"{_hint}\n"
        )

    # Parse all conditions
    parsed = [parse_condition(c, dep_name) for c in conditions]

    # Separate limit and point conditions
    point_conds = [p for p in parsed if not p.is_limit]
    limit_conds = [p for p in parsed if p.is_limit]

    # Find distinct boundary points (limit conditions don't count)
    points = list(dict.fromkeys(p.point for p in point_conds))

    # Check for more than 2 distinct points
    if len(points) > 2:
        pts_str = ", ".join(f"t={p}" for p in points)
        raise ConditionError(
            f"\n\n  Conditions at {len(points)} distinct points ({pts_str}).\n"
            f"  Regular perturbation supports at most 2 boundary points.\n"
        )

    # Check for conflicting point conditions
    seen = {}
    for p in point_conds:
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

    # Determine IVP vs BVP
    # A limit condition at a different point than standard conditions = BVP
    lim_points = [lc.point for lc in limit_conds]
    all_points  = list(dict.fromkeys(points + lim_points))
    distinct    = list(dict.fromkeys(float(p.evalf()) for p in all_points))
    if len(distinct) >= 2:
        problem_type = "bvp"
    elif len(points) <= 1:
        problem_type = "ivp"
    else:
        problem_type = "bvp"

    return parsed, problem_type
