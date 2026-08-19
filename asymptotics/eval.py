"""
asymptotics.eval
================
Evaluate perturbation expansions at given eps and independent variable values.

Usage
-----
>>> u = sol.eval(eps=0.1, at=np.linspace(0, 20, 300))
>>> u = sol.eval(eps=[0.1, 0.2, 0.3], at=np.linspace(0, 20, 300))

>>> x = sol.eval(eps=0.1)           # algebraic — scalar
>>> x = sol.eval(eps=[0.1, 0.2])    # algebraic — ndarray
"""

from __future__ import annotations
import numpy as np
from sympy import lambdify


def eval_hierarchy(h, eps, at=None, params=None):
    r"""
    Evaluate a perturbation expansion numerically at given ``eps`` values.

    This is the engine behind ``sol.eval(...)``.  It substitutes the small
    parameter (and any symbolic parameters) into the assembled expansion and
    returns concrete numbers: a plain ``float`` for algebraic problems, or a
    NumPy array sampled at the requested points for differential problems.
    For the Lindstedt and multiple-scales methods the physical-time expansion
    (``expansion_t``, with the strained frequency already folded in) is used,
    so ``at`` is ordinary physical time.

    Parameters
    ----------
    h : perturbation hierarchy
        Any supported hierarchy: ``OrderHierarchy`` (algebraic),
        ``SystemHierarchy`` (algebraic system), ``ODEHierarchy``,
        ``LindstedtHierarchy``, ``MultScalesHierarchy``,
        ``BoundaryLayerHierarchy``, ``ODESystemHierarchy``, or
        ``StepwiseHierarchy``.
    eps : float or list of float
        Value(s) of the small parameter.  A scalar selects the "scalar" return
        forms below; a list selects the "list" forms.
    at : array-like, optional
        Values of the independent variable at which to sample.  **Required**
        for every differential hierarchy (ODE, oscillator, boundary layer, ODE
        system); ignored for algebraic problems.  A ``ValueError`` is raised if
        it is missing for a differential hierarchy.
    params : dict, optional
        Numerical values for symbolic parameters, e.g. ``{'a': 0.5, 'b': 2.0}``.
        Required (with all keys present) whenever the original equation carried
        symbolic parameters; otherwise a ``ValueError`` is raised.

    Returns
    -------
    float or numpy.ndarray or dict
        The return type depends on the hierarchy and on whether ``eps`` is
        scalar or a list:

        - Algebraic equation, scalar ``eps`` -> ``float``.
        - Algebraic equation, list ``eps`` -> ``ndarray`` (one value per eps).
        - Algebraic system -> ``dict`` ``{variable: float}`` (scalar ``eps``)
          or ``{variable: ndarray}`` (list ``eps``).
        - ODE / oscillator / boundary layer, scalar ``eps`` -> ``ndarray``
          sampled on ``at``.
        - ODE / oscillator / boundary layer, list ``eps`` ->
          ``dict`` ``{eps_val: ndarray}``.
        - ODE system, scalar ``eps`` -> ``dict`` ``{variable: ndarray}``;
          list ``eps`` -> ``dict`` ``{eps_val: {variable: ndarray}}``.

    Raises
    ------
    ValueError
        If ``at`` is required (differential hierarchy) but not given, if
        symbolic parameters are unresolved, or if the expansion still contains
        free symbols after substitution.

    Examples
    --------
    Algebraic problem — a scalar ``eps`` returns a ``float``, a list returns an
    array:

    >>> import numpy as np
    >>> from asymptotics import AlgebraicEquation, eval as evaluate
    >>> eq  = AlgebraicEquation("x**3 + eps*x - 1",
    ...                         dependent="x", small_param="eps")
    >>> sol = eq.expand_regular(order=3)
    >>> round(evaluate(sol, eps=0.1), 6)
    0.966679
    >>> np.round(evaluate(sol, eps=[0.1, 0.2]), 6)
    array([0.966679, 0.933432])

    ODE problem — ``at`` is required and the result is an ``ndarray`` sampled
    on that grid:

    >>> u = evaluate(sol, eps=0.1)          # algebraic: no 'at' needed
    >>> type(u).__name__
    'float'
    """
    from asymptotics.methods.regular_ode       import ODEHierarchy
    from asymptotics.methods.lindstedt         import LindstedtHierarchy
    from asymptotics.methods.multiple_scales   import MultScalesHierarchy
    from asymptotics.methods.boundary_layer    import BoundaryLayerHierarchy
    from asymptotics.methods.regular_ode_system import ODESystemHierarchy
    from asymptotics.core.hierarchy            import OrderHierarchy
    from sympy import Symbol

    # Normalize eps
    scalar = not hasattr(eps, '__iter__')
    eps_list = [float(eps)] if scalar else [float(e) for e in eps]

    # Check for symbolic parameters on the stored problem
    problem = getattr(h, '_problem', None)
    prob_params = getattr(problem, 'params', set()) if problem else set()

    if prob_params:
        if params is None:
            param_str = ', '.join(f"'{p}': value" for p in sorted(prob_params))
            has_at    = not isinstance(h, OrderHierarchy)
            at_str    = ', at=t_vals' if has_at else ''
            raise ValueError(
                f"\n\n  Equation has symbolic parameters: {set(sorted(prob_params))}\n"
                f"  Provide values:\n"
                f"    sol.eval(eps=..{at_str}, params={{{param_str}}})\n"
            )
        missing = prob_params - set(params.keys())
        if missing:
            param_str = ', '.join(f"'{p}': value" for p in sorted(prob_params))
            has_at    = not isinstance(h, OrderHierarchy)
            at_str    = ', at=t_vals' if has_at else ''
            raise ValueError(
                f"\n\n  Missing parameter values: {set(sorted(missing))}\n"
                f"  Required: {set(sorted(prob_params))}\n"
                f"  Provide all values:\n"
                f"    sol.eval(eps=..{at_str}, params={{{param_str}}})\n"
            )

    # Build params substitution dict {Symbol: value}
    param_subs = {Symbol(k): v for k, v in params.items()} if params else {}

    from asymptotics.core.system_hierarchy import SystemHierarchy
    from asymptotics.methods.stepwise import StepwiseHierarchy
    if isinstance(h, StepwiseHierarchy):
        return _eval_ode(h, eps_list, at, scalar, param_subs)
    elif isinstance(h, SystemHierarchy):
        return _eval_algebraic_system(h, eps_list, scalar, param_subs)
    elif isinstance(h, OrderHierarchy):
        return _eval_algebraic(h, eps_list, scalar, param_subs)
    elif isinstance(h, ODESystemHierarchy):
        return _eval_ode_system(h, eps_list, at, scalar, param_subs)
    elif isinstance(h, BoundaryLayerHierarchy):
        return _eval_boundary_layer(h, eps_list, at, scalar, param_subs)
    else:
        # ODEHierarchy, LindstedtHierarchy, MultScalesHierarchy
        return _eval_ode(h, eps_list, at, scalar, param_subs)


# ---------------------------------------------------------------------------
# Algebraic
# ---------------------------------------------------------------------------

def _eval_algebraic(h, eps_list, scalar, param_subs=None):
    eps_sym = h.small_param

    param_subs = param_subs or {}
    # Check for remaining symbolic parameters after eps and param substitution
    test_expr = h.expansion.subs(eps_sym, eps_list[0]).subs(param_subs)
    remaining = test_expr.free_symbols
    if remaining:
        syms = ', '.join(str(s) for s in sorted(remaining, key=str))
        raise ValueError(
            f"\n\n  Cannot evaluate — expansion still contains symbolic parameters: {syms}\n"
            f"  Substitute them first using sol.expansion.subs(b, 2.0)\n"
            f"  Or evaluate manually: float(sol.expansion.subs(eps, 0.1).subs(b, 2.0))\n"
        )

    vals = np.array([
        complex(h.expansion.subs(eps_sym, ev).subs(param_subs).evalf()).real
        for ev in eps_list
    ])
    return float(vals[0]) if scalar else vals


# ---------------------------------------------------------------------------
# ODE (regular, Lindstedt, multiple scales)
# ---------------------------------------------------------------------------

def _eval_ode(h, eps_list, at, scalar, param_subs=None):
    if at is None:
        raise ValueError(
            "\n\n  sol.eval() requires 'at' for ODE hierarchies.\n"
            "  Example: sol.eval(eps=0.1, at=np.linspace(0, 10, 300))\n"
        )

    at = np.asarray(at, dtype=float)
    eps_sym = h.small_param
    t_sym   = h.independent

    # Use expansion_t for Lindstedt/MS — this has tau=omega(eps)*t already
    # substituted, so passing physical time t_vals is correct.
    # omega(eps) is embedded as a coefficient of t after eps substitution.
    # For regular perturbation, expansion is already in physical time.
    expansion  = getattr(h, 'expansion_t', h.expansion)
    param_subs = param_subs or {}

    results = {}
    for ev in eps_list:
        expr = expansion.subs(eps_sym, ev).subs(param_subs)
        fn   = lambdify(t_sym, expr, 'numpy')
        try:
            vals = np.real(np.array([complex(fn(ti)) for ti in at]))
        except Exception:
            vals = np.real(np.array(fn(at), dtype=complex))
        results[ev] = vals

    if scalar:
        return results[eps_list[0]]
    return results


# ---------------------------------------------------------------------------
# Boundary layer
# ---------------------------------------------------------------------------

def _eval_boundary_layer(h, eps_list, at, scalar, param_subs=None):
    if at is None:
        raise ValueError(
            "\n\n  sol.eval() requires 'at' for boundary layer hierarchies.\n"
            "  Example: sol.eval(eps=0.05, at=np.linspace(0, 1, 300))\n"
        )

    at = np.asarray(at, dtype=float)
    eps_sym = h.small_param
    x_sym   = h.independent

    results = {}
    for ev in eps_list:
        expr = h.expansion.subs(eps_sym, ev)
        fn   = lambdify(x_sym, expr, 'numpy')
        vals = fn(at)
        # broadcast scalar (e.g. outer=0)
        vals = np.real(np.broadcast_to(
            np.atleast_1d(np.array(vals, dtype=complex)), at.shape
        ))
        results[ev] = vals

    if scalar:
        return results[eps_list[0]]
    return results


# ---------------------------------------------------------------------------
# ODE System
# ---------------------------------------------------------------------------

def _eval_ode_system(h, eps_list, at, scalar, param_subs=None):
    if at is None:
        raise ValueError(
            "\n\n  sol.eval() requires 'at' for ODESystem hierarchies.\n"
            "  Example: sol.eval(eps=0.1, at=np.linspace(0, 5, 300))\n"
        )

    at = np.asarray(at, dtype=float)
    eps_sym   = h.small_param
    t_sym     = h.independent
    variables = h.variables

    results = {}
    for ev in eps_list:
        var_results = {}
        for var in variables:
            expr = h[var].expansion.subs(eps_sym, ev)
            fn   = lambdify(t_sym, expr, 'numpy')
            try:
                vals = np.real(np.array([complex(fn(ti)) for ti in at]))
            except Exception:
                vals = np.real(np.array(fn(at), dtype=complex))
            var_results[var] = vals
        results[ev] = var_results

    if scalar:
        return results[eps_list[0]]
    return results


# ---------------------------------------------------------------------------
# Algebraic System
# ---------------------------------------------------------------------------

def _eval_algebraic_system(h, eps_list, scalar, param_subs=None):
    eps_sym = h.small_param
    variables = list(h.hierarchies.keys())

    result = {}
    for var in variables:
        vh = h.hierarchies[var]
        param_subs = param_subs or {}
        vals = np.array([
            float(vh.expansion.subs(eps_sym, ev).subs(param_subs))
            for ev in eps_list
        ])
        result[var] = float(vals[0]) if scalar else vals

    return result
