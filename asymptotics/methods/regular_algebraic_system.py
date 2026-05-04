"""
asymptotics.methods.regular_algebraic_system
==========================================
Regular perturbation expansion for coupled algebraic systems.

Algorithm
---------
For a system f_i(x_1,...,x_n, eps) = 0, i=1..n:

1. Build ansatz for each variable:
      x_i(eps) = x_i0 + eps*x_i1 + eps^2*x_i2 + ...

2. Substitute all ansatze into all equations and series-expand in eps.

3. At each order k, collect the eps^k coefficient from each equation.
   This gives a system of n equations in n unknowns (x_1k,...,x_nk).

4. Solve the system. At order 0 it may be nonlinear; at k>0 it is
   always linear (regular perturbation guarantee).

5. Assemble one OrderHierarchy per variable, wrap in SystemHierarchy.
"""

from __future__ import annotations
from typing import List

from sympy import (
    Symbol, symbols, Eq, series, solve, Add,
    expand, simplify
)

from asymptotics.core.hierarchy        import OrderHierarchy, OrderEntry
from asymptotics.core.system_hierarchy import SystemHierarchy
from asymptotics.core.exceptions       import (
    NoSmallParameterError,
    NoLeadingOrderSolutionError,
    NoHigherOrderSolutionError,
    OnlyComplexRootsError,
)


def expand_regular_system(problem, order: int = 3) -> SystemHierarchy:
    """
    Apply regular perturbation theory to a coupled AlgebraicSystem.

    Parameters
    ----------
    problem : AlgebraicSystem
    order : int

    Returns
    -------
    SystemHierarchy
    """
    eps      = problem.small_param
    dep_syms = problem.dependents          # [x, y, ...]
    dep_names= problem._dependent_names    # ["x", "y", ...]
    eqs      = problem.equations           # [f, g, ...]
    N        = order
    n        = len(dep_syms)

    # ------------------------------------------------------------------
    # Check: every equation must contain eps
    # ------------------------------------------------------------------
    for i, f in enumerate(eqs):
        if eps not in f.free_symbols:
            raise NoSmallParameterError(eps, f)

    # ------------------------------------------------------------------
    # Step 1: build order symbols for each variable
    #   x_0, x_1, ..., x_N  and  y_0, y_1, ..., y_N  etc.
    # ------------------------------------------------------------------
    order_syms = {}   # dep_name -> [sym_0, sym_1, ..., sym_N]
    for name in dep_names:
        order_syms[name] = [Symbol(f"{name}_{k}") for k in range(N + 1)]

    # ------------------------------------------------------------------
    # Step 2: build ansatz for each variable
    # ------------------------------------------------------------------
    ansatze = {}   # dep_sym -> ansatz_expr
    for name, dep in zip(dep_names, dep_syms):
        ansatze[dep] = sum(eps**k * order_syms[name][k] for k in range(N + 1))

    # ------------------------------------------------------------------
    # Step 3: substitute all ansatze into all equations and expand
    # ------------------------------------------------------------------
    expanded = []
    for f in eqs:
        f_sub = f
        for dep, ans in ansatze.items():
            f_sub = f_sub.subs(dep, ans)
        expanded.append(series(f_sub, eps, 0, N + 1))

    # ------------------------------------------------------------------
    # Step 4: collect coefficients at each order
    #   coeffs[k] = list of n expressions (one per equation)
    # ------------------------------------------------------------------
    coeffs = {}
    for k in range(N + 1):
        coeffs[k] = [ex.coeff(eps, k) for ex in expanded]

    # ------------------------------------------------------------------
    # Step 5: solve order by order
    # ------------------------------------------------------------------
    known = {}   # all solved order symbols -> values

    for k in range(N + 1):
        # Unknowns at this order
        unknowns = [order_syms[name][k] for name in dep_names]

        # Equations at this order with known values substituted
        eqs_k = [expand(c.subs(known)) for c in coeffs[k]]

        try:
            sol = solve(eqs_k, unknowns)
        except Exception:
            sol = []

        # ------------------------------------------------------------------
        # Handle solution forms
        # ------------------------------------------------------------------
        if not sol:
            if k == 0:
                raise NoLeadingOrderSolutionError(eqs_k, eps, dep_syms)
            else:
                raise NoHigherOrderSolutionError(k, unknowns, eqs_k, known)

        # solve() returns list of tuples for systems, or a dict
        if isinstance(sol, list):
            # Multiple solutions — pick one
            sol_dict = _pick_solution(sol, unknowns, problem.root_hint, dep_names, k)
        elif isinstance(sol, dict):
            sol_dict = sol
        else:
            raise NoHigherOrderSolutionError(k, unknowns, eqs_k, known)

        # Simplify and store
        for sym in unknowns:
            val = simplify(sol_dict.get(sym, sym))
            known[sym] = val

    # ------------------------------------------------------------------
    # Step 6: assemble SystemHierarchy
    # ------------------------------------------------------------------
    sys_hier             = SystemHierarchy()
    sys_hier.small_param = eps
    sys_hier._method     = "Regular perturbation — algebraic system"
    sys_hier.variables   = dep_names

    # Store the full coupled system at each order for display.
    # We want to show the coupling: substitute lower-order solutions for
    # previous orders, but keep the current-order unknowns (x_k, y_k, ...)
    # AND keep lower-order symbols (x_0, y_0, ...) visible at order 0 only.
    # For k > 0: substitute numeric values for all orders < k so that
    # the coupling like "x1 + y0**2" shows as "x1 + 4" revealing what drove it.
    # Better: substitute j < k but show j == k-1 symbolically to expose coupling.
    coupled_orders = {}
    for k in range(N + 1):
        # Substitute all orders strictly less than k-1 numerically,
        # keep order k-1 symbols so coupling is visible
        if k == 0:
            # Nothing to substitute — show raw O(1) equations
            eqs_k_display = [expand(c) for c in coeffs[k]]
        else:
            # Substitute orders 0..k-2 numerically, keep k-1 symbolic
            partial_known = {
                order_syms[nm][j]: known[order_syms[nm][j]]
                for nm in dep_names
                for j in range(max(0, k - 1))
            }
            eqs_k_display = [expand(c.subs(partial_known)) for c in coeffs[k]]

        sols_k = {dep_names[i]: known[order_syms[dep_names[i]][k]]
                  for i in range(n)}
        syms_k = [order_syms[nm][k] for nm in dep_names]
        coupled_orders[k] = {
            "equations" : [Eq(e, 0) for e in eqs_k_display],
            "unknowns"  : syms_k,
            "solutions" : sols_k,
        }
    sys_hier.coupled_orders = coupled_orders

    # Build one OrderHierarchy per variable (for sol["x"] access)
    for name in dep_names:
        h             = OrderHierarchy()
        h.small_param = eps
        h._method     = f"Regular perturbation — {name}"
        h._problem_repr = name

        for k in range(N + 1):
            sym = order_syms[name][k]
            val = known[sym]
            # Store the individual equation for this variable at order k
            lower_known_k = {s: v for s, v in known.items()
                             if str(s) != f"{name}_{k}"}
            eq_expr = expand(coeffs[k][dep_names.index(name)].subs(lower_known_k))
            entry = OrderEntry(
                order    = k,
                equation = Eq(eq_expr, 0),
                solution = val,
                symbol   = sym,
                note     = "",
            )
            h.entries.append(entry)

        h.composite = Add(*[known[order_syms[name][k]] * eps**k for k in range(N + 1)])
        h.collected = {k: coeffs[k][dep_names.index(name)] for k in range(N + 1)}
        sys_hier.hierarchies[name] = h

    return sys_hier


def _pick_solution(sol_list, unknowns, root_hint, dep_names, order):
    """
    Pick one solution from a list of solution tuples.

    At order 0: prefer real solutions, pick by root_hint or largest norm.
    At order k>0: should be unique (linear system).
    """
    from sympy import im, re, Abs

    # Normalise to list of dicts
    if isinstance(sol_list[0], (list, tuple)):
        dicts = [dict(zip(unknowns, s)) for s in sol_list]
    elif isinstance(sol_list[0], dict):
        dicts = sol_list
    else:
        # Single solution wrapped in list
        return dict(zip(unknowns, sol_list))

    if order > 0:
        return dicts[0]

    # Filter to real solutions
    def _is_real_solution(d):
        return all(v.is_real for v in d.values())

    real_dicts = [d for d in dicts if _is_real_solution(d)]
    candidates = real_dicts if real_dicts else dicts

    if not real_dicts and order == 0:
        raise OnlyComplexRootsError(unknowns, [list(d.values()) for d in dicts])

    # Apply root_hint if given
    if root_hint is not None:
        best = None
        best_dist = float('inf')
        for d in candidates:
            dist = sum(
                float(abs(complex(d[unknowns[i]] - root_hint.get(dep_names[i], 0))))
                for i in range(len(unknowns))
                if dep_names[i] in root_hint
            )
            if dist < best_dist:
                best_dist = dist
                best = d
        return best

    # Pick solution with largest sum of real parts (principal solution)
    try:
        candidates = sorted(
            candidates,
            key=lambda d: sum(float(v.evalf().as_real_imag()[0]) for v in d.values()),
            reverse=True,
        )
    except Exception:
        pass

    return candidates[0]
