"""
asymptotics.methods.regular_algebraic
==================================
Regular perturbation expansion for algebraic equations f(x, eps) = 0.

Algorithm
---------
1. Build the ansatz  x = x0 + eps*x1 + eps^2*x2 + ...  (symbolic symbols x_0, x_1, ...)
2. Substitute into f(x, eps)
3. Taylor-expand in eps around eps=0 and collect coefficients of eps^k
4. At each order k, solve the resulting polynomial equation for x_k
   (previous x_j for j<k are already known and substituted in)
5. Assemble the expansion expansion

Design principles
-----------------
- Every intermediate object is stored on the returned OrderHierarchy
- Users can override x_k manually before subsequent orders are solved
- root_hint on the problem selects which leading-order root to follow
"""

from __future__ import annotations
from typing import Optional, List

from sympy import (
    Symbol, symbols, Eq, series, solve, Rational,
    Add, Mul, Pow, Integer, sympify, pretty, latex,
    expand, collect, factor, simplify, cancel
)

from asymptotics.core.problem    import AlgebraicEquation
from asymptotics.core.hierarchy  import OrderHierarchy, OrderEntry
from asymptotics.core.exceptions import (
    NoSmallParameterError,
    NoLeadingOrderSolutionError,
    NoHigherOrderSolutionError,
    OnlyComplexRootsError,
)
from asymptotics.gauge import parse_gauge, extract_coefficients, is_standard_gauge


def _make_order_symbols(base_name: str, n: int) -> List[Symbol]:
    """Create x_0, x_1, ..., x_n as distinct SymPy symbols."""
    return [Symbol(f"{base_name}_{k}") for k in range(n + 1)]


def expand_regular_algebraic(
    problem: AlgebraicEquation,
    order: int = 3,
    root_index: int = 0,
    gauge=None,
) -> OrderHierarchy:
    """
    Apply regular perturbation theory to an algebraic problem.

    Parameters
    ----------
    problem : AlgebraicEquation
    order : int
        Highest power of eps to compute (inclusive).
    root_index : int
        Which root of the O(1) equation to follow (0 = first real root).
        Overridden by problem.root_hint if set.

    Returns
    -------
    OrderHierarchy
        Fully populated hierarchy with all intermediate expressions.

    Raises
    ------
    NoSmallParameterError
        If the small parameter does not appear in the equation.
    NoLeadingOrderSolutionError
        If SymPy cannot solve the O(1) equation symbolically.
    OnlyComplexRootsError
        If the O(1) equation has only complex roots and no root_hint given.
    NoHigherOrderSolutionError
        If SymPy cannot solve the equation at some order k > 0.
    """
    eps = problem.small_param
    x   = problem.dependent
    f   = problem.equation
    N   = order

    # ------------------------------------------------------------------
    # Check 1: small parameter must appear in the equation
    # ------------------------------------------------------------------
    if eps not in f.free_symbols:
        raise NoSmallParameterError(eps, f)

    h = OrderHierarchy()
    h._method            = "Regular perturbation — algebraic"
    h._problem_repr      = f"f({x}, {eps}) = {f} = 0"
    h.small_param        = eps
    h._original_equation = f        # stored for compare_numeric
    h._dependent_sym     = x

    # ------------------------------------------------------------------
    # Step 1: build order symbols  x_0, x_1, ..., x_N
    # ------------------------------------------------------------------
    x_syms = _make_order_symbols(str(x), N)

    # ------------------------------------------------------------------
    # Step 1b: build gauge sequence
    # ------------------------------------------------------------------
    gauge_seq = parse_gauge(gauge, N, eps)
    h._gauge  = gauge_seq           # stored for display

    # ------------------------------------------------------------------
    # Step 2: build ansatz as a sum over gauge functions
    #   x(ε) = x_0·δ_0(ε) + x_1·δ_1(ε) + ... + x_N·δ_N(ε)
    # ------------------------------------------------------------------
    x_ans = sum(x_syms[k] * gauge_seq[k] for k in range(N + 1))

    # ------------------------------------------------------------------
    # Step 3: substitute ansatz into f and expand
    # ------------------------------------------------------------------
    f_substituted = f.subs(x, x_ans)
    f_expanded    = expand(f_substituted)

    h.substituted_equation = f_substituted

    # ------------------------------------------------------------------
    # Step 4: collect coefficients by gauge function
    #   Use sequential limit extraction (works for power-law AND log gauges)
    # ------------------------------------------------------------------
    coeff_list = extract_coefficients(f_expanded, gauge_seq, eps)
    coeff      = {k: coeff_list[k] for k in range(N + 1)}
    h.collected = coeff

    # ------------------------------------------------------------------
    # Step 5: solve order by order
    #
    # With non-standard gauges the coefficient equations may not contain
    # x_k — instead they constrain an earlier x_j that first appears at
    # this order.  We use a deferred-solve strategy:
    #
    #   At order k: substitute known values, find the lowest-index
    #   undetermined symbol present in the equation, solve for it.
    #   Skip if equation is identically 0 (gauge order not present).
    # ------------------------------------------------------------------
    known = {}      # x_sym -> value (built up incrementally)

    for k in range(N + 1):
        eq_expr  = expand(coeff[k].subs(known))
        equation = Eq(eq_expr, 0)

        # Trivially satisfied — this gauge order contributes no constraint.
        # Don't add a display entry; the symbol will be determined at a
        # later order and back-filled there.
        if eq_expr == Integer(0):
            continue

        # Find the lowest-index undetermined symbol present in eq_expr.
        # In a standard gauge this is always x_syms[k]; with non-standard
        # gauges it may be an earlier x_syms[j] that first becomes
        # constrained at this order.
        undetermined = [s for s in x_syms if s not in known]
        target = None
        for s in undetermined:
            if s in eq_expr.free_symbols:
                target = s
                break

        if target is None:
            # No undetermined symbol — check the equation is satisfiable
            if eq_expr != Integer(0):
                raise NoHigherOrderSolutionError(k, x_syms[k], eq_expr, known)
            entry = OrderEntry(
                order    = k,
                equation = equation,
                solution = Integer(0),
                symbol   = x_syms[k],
                note     = "trivially satisfied",
            )
            h.entries.append(entry)
            continue

        # Solve for target
        try:
            sols = solve(eq_expr, target)
        except NotImplementedError:
            sols = []

        # ------------------------------------------------------------------
        # Check 2: no solution found
        # ------------------------------------------------------------------
        if not sols:
            if k == 0 or target == x_syms[0]:
                raise NoLeadingOrderSolutionError(eq_expr, eps, x)
            else:
                raise NoHigherOrderSolutionError(k, target, eq_expr, known)

        # ------------------------------------------------------------------
        # Root selection — applied when solving for the leading unknown x_0
        # (which may happen at order k>0 with non-standard gauges).
        # ------------------------------------------------------------------
        if target == x_syms[0]:
            if problem.root_hint is not None:
                hint         = sympify(problem.root_hint)
                real_sols_h  = [s for s in sols if s.is_real]
                target_list  = real_sols_h if real_sols_h else sols
                x_k_val      = min(target_list, key=lambda s: abs(complex(s - hint)))
            else:
                # is_real returns True (real), False (complex), or None (unknown)
                # None happens when symbolic parameters are present.
                # Strategy: substitute a small positive test value for all
                # free symbols (except eps) to numerically identify real roots.

                from sympy import im, Abs

                # First try: definitively real
                real_sols = [s for s in sols if s.is_real is True]

                # Second try: numerically identify real roots by test substitution
                if not real_sols:
                    free = set()
                    for s in sols:
                        free |= s.free_symbols
                    free -= {eps}   # remove small param
                    test_subs = {sym: 1 for sym in free}  # substitute 1 for all params

                    def _is_numerically_real(expr):
                        try:
                            val = complex(expr.subs(test_subs).evalf())
                            return abs(val.imag) < 1e-10 * (abs(val) + 1e-10)
                        except Exception:
                            return False

                    real_sols = [s for s in sols
                                 if s.is_real is not False
                                 and _is_numerically_real(s)]

                # Third try: all non-complex
                if not real_sols:
                    real_sols = [s for s in sols if s.is_real is not False]

                if not real_sols:
                    raise OnlyComplexRootsError(eq_expr, sols)

                try:
                    real_sols = sorted(
                        real_sols,
                        key=lambda s: float(s.evalf()),
                        reverse=True,
                    )
                except Exception:
                    pass
                x_k_val = real_sols[root_index % len(real_sols)]
        else:
            x_k_val = sols[0]

        x_k_val = simplify(x_k_val)
        known[target] = x_k_val

        note = ""
        if target == x_syms[0] and len(sols) > 1:
            others         = [s for s in sols if s != x_k_val]
            real_others    = [s for s in others if s.is_real]
            complex_others = [s for s in others if not s.is_real]
            parts = []
            if real_others:
                parts.append(f"other real roots: {real_others}")
            if complex_others:
                parts.append(f"complex roots omitted ({len(complex_others)} total)")
            note = "; ".join(parts)

        entry = OrderEntry(
            order    = k,
            equation = equation,
            solution = x_k_val,
            symbol   = target,
            note     = note,
        )
        h.entries.append(entry)

    # ------------------------------------------------------------------
    # Step 6: assemble expansion expansion
    # ------------------------------------------------------------------
    from sympy import Integer as _Int
    expansion_terms = [known.get(x_syms[k], _Int(0)) * gauge_seq[k] for k in range(N + 1)]
    h.expansion = Add(*expansion_terms)

    h._problem = problem
    return h
