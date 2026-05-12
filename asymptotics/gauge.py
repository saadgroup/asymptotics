"""
asymptotics.gauge
=================
Utilities for non-standard asymptotic gauge sequences.

A gauge sequence {δ_0(ε), δ_1(ε), ..., δ_N(ε)} is an asymptotic sequence if
    δ_{k+1}(ε) / δ_k(ε) → 0   as ε → 0⁺
for all k.  The standard choice is δ_k = ε^k.

API
---
parse_gauge(gauge, order, eps)
    Accept either a string (pattern → infer) or a list (explicit), and
    return a list of N+1 SymPy expressions.

extract_coefficients(expr, gauge, eps)
    Given a SymPy expression that is a linear combination of gauge functions
    (plus higher-order remainders), return the list of coefficients
    [c_0, c_1, ..., c_N] via sequential limit extraction.

gauge_to_latex(gauge, eps)
    Return a list of LaTeX strings for each gauge function, with ε rendered
    as \\varepsilon.

is_standard_gauge(gauge, eps)
    Return True if gauge is the standard power sequence {1, ε, ε², ...}.
"""

from __future__ import annotations
from typing import List, Union

from sympy import (
    Symbol, Expr, Integer, Rational, limit, expand, simplify,
    sympify, latex, oo, zoo, nan, log, sqrt, S
)
from sympy.parsing.sympy_parser import parse_expr


# ---------------------------------------------------------------------------
# Public: parse_gauge
# ---------------------------------------------------------------------------

def parse_gauge(
    gauge: Union[str, List[str], None],
    order: int,
    eps: Symbol,
) -> List[Expr]:
    """
    Build the gauge sequence from user input.

    Parameters
    ----------
    gauge : str, list of str, or None
        - None   → standard sequence [1, ε, ε², ..., ε^order]
        - str    → pattern: infer geometric sequence with ratio = parse(gauge)
                   e.g. 'sqrt(eps)' → [1, √ε, ε, ε^(3/2), ...]
        - list   → explicit sequence, must have exactly order+1 elements

    order : int
        Expansion order (inclusive).  Returned list has order+1 elements.

    eps : Symbol
        The small parameter symbol.

    Returns
    -------
    list of SymPy Expr, length order+1

    Raises
    ------
    ValueError
        If the list has wrong length, or the sequence is not asymptotic.
    """
    N = order

    if gauge is None:
        # Standard sequence: 1, ε, ε², ...
        return [eps**k for k in range(N + 1)]

    if isinstance(gauge, str):
        return _infer_gauge(gauge, N, eps)

    if isinstance(gauge, (list, tuple)):
        return _explicit_gauge(list(gauge), N, eps)

    raise TypeError(
        f"gauge must be a string, list of strings, or None; got {type(gauge).__name__}"
    )


# ---------------------------------------------------------------------------
# Public: extract_coefficients
# ---------------------------------------------------------------------------

def extract_coefficients(
    expr: Expr,
    gauge: List[Expr],
    eps: Symbol,
) -> List[Expr]:
    """
    Extract the coefficient of each gauge function from expr via sequential
    limit extraction.

    Algorithm (works for both power-law and logarithmic sequences):
        remainder = expr
        for each δ_k in gauge:
            c_k = lim_{ε→0⁺}  remainder / δ_k
            remainder -= c_k * δ_k

    Parameters
    ----------
    expr : Expr
        The expression to decompose (after substituting the ansatz).
    gauge : list of Expr
        The gauge sequence, length N+1.
    eps : Symbol
        The small parameter.

    Returns
    -------
    list of Expr, length N+1
        c_k such that expr ≈ Σ c_k δ_k(ε).
    """
    coeffs    = []
    remainder = expand(expr)

    for dk in gauge:
        try:
            ck = limit(remainder / dk, eps, 0, "+")
        except Exception:
            ck = S.Zero

        # Guard: if limit is ±∞ the gauge is wrong — return zero and warn
        if ck in (oo, -oo, zoo, nan) or (hasattr(ck, 'is_finite') and ck.is_finite is False):
            ck = S.Zero

        coeffs.append(expand(ck))
        remainder = expand(remainder - ck * dk)

    return coeffs


# ---------------------------------------------------------------------------
# Public: gauge_to_latex
# ---------------------------------------------------------------------------

def gauge_to_latex(gauge: List[Expr], eps: Symbol) -> List[str]:
    """
    Return LaTeX strings for each gauge function, with ε → \\varepsilon.
    """
    eps_str = latex(eps)
    result  = []
    for dk in gauge:
        lx = latex(dk)
        lx = lx.replace(eps_str, r"\varepsilon")
        result.append(lx)
    return result


# ---------------------------------------------------------------------------
# Public: is_standard_gauge
# ---------------------------------------------------------------------------

def is_standard_gauge(gauge: List[Expr], eps: Symbol) -> bool:
    """Return True if this is the standard {1, ε, ε², ...} sequence."""
    for k, dk in enumerate(gauge):
        if not simplify(dk - eps**k) == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_single(s: str, eps: Symbol) -> Expr:
    """Parse one gauge string into a SymPy expression."""
    local = {
        str(eps): eps,
        "eps":    eps,
        "log":    log,
        "sqrt":   sqrt,
        "Rational": Rational,
    }
    try:
        return parse_expr(s.strip(), local_dict=local, transformations="all")
    except Exception as e:
        raise ValueError(
            f"\n\n  Could not parse gauge term: '{s}'\n"
            f"  SymPy error: {e}\n\n"
            f"  Tips:\n"
            f"    - Use ** for powers:   eps**(1/2)  not  eps^(1/2)\n"
            f"    - Use sqrt(eps) or eps**(1/2) for square root\n"
            f"    - Use log(eps) for logarithm\n"
        ) from e


def _infer_gauge(pattern: str, order: int, eps: Symbol) -> List[Expr]:
    """
    Build geometric gauge from a string pattern.

    The ratio r = parse(pattern).
    gauge[k] = r^k   (so gauge[0] = 1, gauge[1] = r, gauge[2] = r², ...)

    Special case: if pattern parses to something that already starts at
    a non-trivial power (e.g. 'eps**(1/3)'), we use it as the ratio.
    """
    r = _parse_single(pattern, eps)

    # Build [1, r, r², ..., r^order]
    gauge = [simplify(r**k) for k in range(order + 1)]

    _validate_gauge(gauge, eps, source=f"inferred from '{pattern}'")
    return gauge


def _explicit_gauge(terms: List[str], order: int, eps: Symbol) -> List[Expr]:
    """Parse and validate an explicit gauge list."""
    N = order
    if len(terms) != N + 1:
        raise ValueError(
            f"\n\n  gauge list has {len(terms)} term(s) but order={N} requires "
            f"exactly {N + 1} terms.\n\n"
            f"  Either:\n"
            f"    - Provide all {N + 1} terms explicitly, OR\n"
            f"    - Use a string pattern for automatic inference:\n"
            f"        gauge='sqrt(eps)'   →  {{1, √ε, ε, ε^(3/2), ...}}\n"
        )

    gauge = [_parse_single(t, eps) for t in terms]
    _validate_gauge(gauge, eps, source="explicit gauge list")
    return gauge


def _validate_gauge(gauge: List[Expr], eps: Symbol, source: str = "") -> None:
    """
    Check that the sequence is asymptotic: δ_{k+1}/δ_k → 0 as ε → 0⁺.

    Issues a warning rather than raising, because SymPy's limit() can
    sometimes fail on valid sequences.
    """
    import warnings
    for k in range(len(gauge) - 1):
        dk, dk1 = gauge[k], gauge[k + 1]
        try:
            ratio = limit(dk1 / dk, eps, 0, "+")
            if ratio not in (S.Zero,) and ratio.is_zero is not True:
                # ratio != 0 means not asymptotic
                if ratio.is_finite is False or ratio in (oo, -oo, zoo):
                    raise ValueError(
                        f"\n\n  gauge sequence ({source}) is NOT asymptotic:\n"
                        f"    δ_{k+1}/δ_{k} = {dk1}/{dk} → {ratio} as ε→0\n"
                        f"  (should → 0).  Check the ordering of your gauge terms.\n"
                    )
                # ratio is a nonzero finite constant — warn but allow
                warnings.warn(
                    f"gauge term ratio δ_{k+1}/δ_{k} → {ratio} (expected 0). "
                    f"Sequence may not be asymptotic.",
                    stacklevel=4,
                )
        except ValueError:
            raise
        except Exception:
            # SymPy limit failed — skip validation silently
            pass


def gauge_term_unicode(dk, eps) -> str:
    """Render a gauge term as a unicode string for text display."""
    from sympy import latex, Integer
    if dk == Integer(1):
        return "1"
    s = str(dk)
    # simple replacements for common cases
    s = s.replace("eps", "ε")
    s = s.replace("**", "^")
    s = s.replace("sqrt(ε)", "√ε")
    s = s.replace("(1/2)", "^(1/2)")
    return s


def gauge_term_latex(dk, eps) -> str:
    """Render a gauge term as LaTeX with ε → \\varepsilon."""
    from sympy import latex, Integer
    s = latex(dk)
    s = s.replace(latex(eps), r"\varepsilon")
    return s
