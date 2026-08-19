"""
asymptotics.numerics
=====================
Numeric comparison utilities for perturbation expansions.

Provides compare_numeric() which dispatches to the appropriate
numerical solver based on the problem and hierarchy type.
"""

from __future__ import annotations
import numpy as np
from sympy import Symbol, symbols, lambdify, solve, Integer, sympify


# ---------------------------------------------------------------------------
# Dispatcher — called by sol.compare_numeric()
# ---------------------------------------------------------------------------

def compare_numeric(hierarchy, eps, params=None, filename=None, **kwargs):
    r"""
    Compare a perturbation expansion against a numerical reference solution.

    This is the engine behind ``sol.compare_numeric(...)``.  It validates the
    symbolic asymptotic expansion by solving the *original* (unexpanded)
    problem numerically with SciPy and overlaying the two on a single figure.
    The solver is chosen from the hierarchy type:

    ==========================================  =============================
    Hierarchy                                   SciPy reference solver
    ==========================================  =============================
    ODE / Stepwise (IVP)                        :func:`scipy.integrate.solve_ivp`
    ODE (BVP), boundary layer                   :func:`scipy.integrate.solve_bvp`
    Lindstedt / multiple scales (oscillators)   :func:`scipy.integrate.solve_ivp`
    ODE system                                  :func:`scipy.integrate.solve_ivp`
    Algebraic equation / system                 :func:`scipy.optimize.root`
    ==========================================  =============================

    The returned dict is augmented (best-effort) with quantitative error norms
    and the exact solver settings used, so a comparison is reproducible.

    Parameters
    ----------
    hierarchy : perturbation hierarchy
        Any of the hierarchy objects produced by an ``expand_*`` method
        (``ODEHierarchy``, ``LindstedtHierarchy``, ``OrderHierarchy``,
        ``SystemHierarchy``, ...).
    eps : float or iterable of float
        Value(s) of the small parameter.  A scalar is normalised to a
        one-element list internally; a list produces one curve pair per value
        (and one error entry per value).
    params : dict, optional
        Numerical values for any symbolic parameters that remain in the
        equation, e.g. ``{'a': 0.5, 'b': 2.0}``.  Required (with all keys
        present) whenever the original problem carried symbolic parameters,
        otherwise a ``ValueError`` is raised.
    filename : str, optional
        If given, the resulting figure is saved to this path (format inferred
        from the extension, e.g. ``'out.pdf'``/``'out.png'``) with
        ``bbox_inches='tight'``.
    **kwargs
        Forwarded to the per-type solver.  Commonly:

        plot_range : [a, b]
            Domain for plotting / integration (ODE types).  Inferred from the
            problem's conditions when omitted.
        n_points : int
            Number of plotting points (default 300 for ODEs, 500 for
            oscillators, 400 for boundary layers).

    Returns
    -------
    dict
        Keys depend on the problem type.  For ODE, oscillator, stepwise and
        ODE-system hierarchies:

        ``'t'`` : ndarray
            The evaluation grid (independent variable), shape ``(n_points,)``.
        ``'u_pert'`` : dict
            ``{eps_value: ndarray}`` — the perturbation expansion sampled on
            ``'t'``.  For ODE systems each value is itself
            ``{variable: ndarray}``.
        ``'u_numerical'`` : dict
            ``{eps_value: ndarray}`` — the SciPy reference solution on the same
            grid, keyed identically to ``'u_pert'``.
        ``'fig'`` : matplotlib.figure.Figure
            The comparison figure (always present; save it via ``filename`` or
            ``result['fig'].savefig(...)``).
        ``'errors'`` : dict
            ``{eps_value: {...}}`` — error norms of the expansion against the
            reference, per eps value.  For ODE-type problems each entry is the
            :func:`error_norms` dict ``{'L2', 'Linf', 'L2_rel', 'Linf_rel'}``
            (per-variable for systems).  For algebraic problems each entry is
            ``{'abs', 'rel'}`` (per-variable for systems).
        ``'settings'`` : dict
            The SciPy solver, method, and tolerances used for the reference
            (e.g. ``{'solver': 'scipy.integrate.solve_ivp', 'method': 'RK45',
            'rtol': 1e-10, 'atol': 1e-12}``), for reproducible reporting.

        Algebraic equations/systems instead return ``'eps'`` (the sorted eps
        grid used as the x-axis), ``'perturbation'`` and ``'numerical'``
        (scalar arrays, or ``{variable: array}`` for systems), plus ``'fig'``,
        ``'errors'`` and ``'settings'``.  Boundary-layer hierarchies return
        ``'x'``, ``'results'`` (``{eps: {'u_outer', 'u_inner', 'u_expansion',
        'u_numerical'}}``), ``'fig'``, ``'errors'`` and ``'settings'``.

    Raises
    ------
    ValueError
        If the equation has symbolic parameters and ``params`` is missing or
        incomplete.
    TypeError
        If ``hierarchy`` is not a supported hierarchy type.
    NotImplementedError
        For ODE BVPs that contain limit (singular) boundary conditions, which
        SciPy's BVP solver cannot handle reliably.

    Notes
    -----
    The ``'errors'`` and ``'settings'`` keys are attached defensively: if the
    error/settings computation fails for any reason it is silently skipped so
    that the primary comparison (and its figure) is never lost.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')          # non-interactive backend
    >>> import numpy as np
    >>> from asymptotics import ODE, compare_numeric
    >>> eq  = ODE("u' + u + eps*u**2", small_param='eps',
    ...           conditions=["u(0) = 1"])          # doctest: +SKIP
    >>> sol = eq.expand_regular(order=2)            # doctest: +SKIP
    >>> res = compare_numeric(sol, eps=0.1)         # doctest: +SKIP
    >>> sorted(res)                                 # doctest: +SKIP
    ['errors', 'fig', 'settings', 't', 'u_numerical', 'u_pert']
    >>> res['settings']['solver']                   # doctest: +SKIP
    'scipy.integrate.solve_ivp'
    >>> res['errors'][0.1]['L2'] < 1e-3             # e.g. True   # doctest: +SKIP
    True
    """
    # Normalize eps: scalar -> single-element list, list -> list
    if not hasattr(eps, '__iter__'):
        eps_list = [float(eps)]
    else:
        eps_list = [float(e) for e in eps]

    # Check for symbolic parameters
    from sympy import Symbol
    problem  = getattr(hierarchy, '_problem', None)
    prob_params = getattr(problem, 'params', set()) if problem else set()
    if prob_params:
        if params is None:
            param_str = ', '.join(f"'{p}': value" for p in sorted(prob_params))
            raise ValueError(
                f"\n\n  Equation has symbolic parameters: {set(sorted(prob_params))}\n"
                f"  Provide values:\n"
                f"    sol.compare_numeric(eps=.., params={{{param_str}}})\n"
            )
        missing = prob_params - set(params.keys())
        if missing:
            param_str = ', '.join(f"'{p}': value" for p in sorted(prob_params))
            raise ValueError(
                f"\n\n  Missing parameter values: {set(sorted(missing))}\n"
                f"  Required: {set(sorted(prob_params))}\n"
                f"  Provide all values:\n"
                f"    sol.compare_numeric(eps=.., params={{{param_str}}})\n"
            )
    param_subs = {Symbol(k): v for k, v in params.items()} if params else {}
    kwargs['param_subs'] = param_subs
    from asymptotics.methods.regular_ode      import ODEHierarchy
    from asymptotics.methods.lindstedt        import LindstedtHierarchy
    from asymptotics.methods.multiple_scales  import MultScalesHierarchy
    from asymptotics.methods.boundary_layer   import BoundaryLayerHierarchy
    from asymptotics.core.hierarchy           import OrderHierarchy   # algebraic

    from asymptotics.methods.regular_ode_system import ODESystemHierarchy
    from asymptotics.core.system_hierarchy import SystemHierarchy
    from asymptotics.methods.stepwise import StepwiseHierarchy
    if isinstance(hierarchy, StepwiseHierarchy):
        result = _compare_ode(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, SystemHierarchy):
        result = _compare_algebraic_system(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, ODESystemHierarchy):
        result = _compare_ode_system(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, BoundaryLayerHierarchy):
        result = _compare_boundary_layer(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, ODEHierarchy):
        result = _compare_ode(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, (LindstedtHierarchy, MultScalesHierarchy)):
        result = _compare_ode_oscillator(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, OrderHierarchy):
        result = _compare_algebraic(hierarchy, eps_list, **kwargs)
    else:
        raise TypeError(
            f"\n\n  compare_numeric does not support {type(hierarchy).__name__}.\n"
        )

    if filename is not None and 'fig' in result:
        result['fig'].savefig(filename, bbox_inches='tight')

    # Attach quantitative error norms + solver settings (additive, backward-compatible)
    try:
        result['errors']   = _attach_error_norms(result)
        result['settings'] = _solver_settings(hierarchy)
    except Exception:
        # Never let error-reporting break the primary comparison
        pass

    return result


# ---------------------------------------------------------------------------
# Quantitative error norms
# ---------------------------------------------------------------------------

def error_norms(u_ref, u_approx, x=None):
    r"""
    L2 and L-infinity error between an approximation and a reference solution.

    Computes absolute and relative error norms between a reference (numerical)
    solution and an approximation (typically a perturbation expansion) sampled
    at the same points.

    The absolute norms are

    .. math::

        \|u_\text{approx} - u_\text{ref}\|_2, \qquad
        \|u_\text{approx} - u_\text{ref}\|_\infty
        = \max_i \bigl|u_{\text{approx},i} - u_{\text{ref},i}\bigr|,

    and the relative norms divide these by the corresponding norm of the
    reference,

    .. math::

        \frac{\|u_\text{approx} - u_\text{ref}\|_2}{\|u_\text{ref}\|_2},
        \qquad
        \frac{\|u_\text{approx} - u_\text{ref}\|_\infty}
             {\|u_\text{ref}\|_\infty}.

    Parameters
    ----------
    u_ref : array_like
        Reference (numerical) solution values.
    u_approx : array_like
        Approximate (perturbation) solution values, same shape as ``u_ref``.
    x : array_like, optional
        Grid on which the two solutions are sampled.  If given, the L2 norm is
        the grid-independent root-mean-square integral

        .. math::

            \|u_\text{approx} - u_\text{ref}\|_2 =
            \sqrt{\frac{1}{b-a}\int_a^b
                  \bigl(u_\text{approx}-u_\text{ref}\bigr)^2\,dx},

        evaluated by the trapezoidal rule (with :math:`a`, :math:`b` the grid
        endpoints), so the reported value does not depend on the number of
        sample points.  The relative L2 norm uses the same integral form for
        the reference.  If ``x`` is omitted, the discrete RMS
        :math:`\sqrt{\tfrac{1}{N}\sum_i (u_{\text{approx},i}-u_{\text{ref},i})^2}`
        over the samples is used instead.

    Returns
    -------
    dict
        A dict with four float entries:

        ``'L2'`` : float
            Absolute L2 (RMS or grid-independent RMS-integral) error.
        ``'Linf'`` : float
            Absolute maximum (L-infinity) error.
        ``'L2_rel'`` : float
            L2 error normalised by the L2 norm of ``u_ref`` (``NaN`` if that
            norm is zero).
        ``'Linf_rel'`` : float
            L-infinity error normalised by ``max|u_ref|`` (``NaN`` if zero).

    Notes
    -----
    Non-finite (``NaN``/``inf``) samples in either array are masked out before
    the norms are computed.  If nothing finite remains, all four entries are
    ``NaN``.  When ``x`` is supplied the samples are sorted by ``x`` first, so
    an unordered grid is handled correctly; a degenerate grid (single point or
    zero span) falls back to the discrete RMS.

    Examples
    --------
    >>> import numpy as np
    >>> from asymptotics import error_norms
    >>> u_ref    = np.array([1.0, 2.0, 3.0])
    >>> u_approx = np.array([1.1, 2.0, 2.8])
    >>> e = error_norms(u_ref, u_approx)
    >>> round(e['L2'], 6)
    0.129099
    >>> round(e['Linf'], 6)
    0.2
    >>> round(e['L2_rel'], 6)
    0.059761
    >>> round(e['Linf_rel'], 6)
    0.066667

    With an explicit grid the L2 norm becomes the grid-independent RMS
    integral.  For a uniform 0.1 offset on ``x = [0, 1, 2]`` it equals 0.1:

    >>> x  = np.array([0.0, 1.0, 2.0])
    >>> ur = np.array([0.0, 1.0, 2.0])
    >>> ua = ur + 0.1
    >>> round(error_norms(ur, ua, x)['L2'], 6)
    0.1
    """
    u_ref    = np.asarray(u_ref, dtype=float)
    u_approx = np.asarray(u_approx, dtype=float)
    diff     = u_approx - u_ref
    mask     = np.isfinite(diff) & np.isfinite(u_ref)
    diff     = diff[mask]
    ref      = u_ref[mask]
    if diff.size == 0:
        return {'L2': float('nan'), 'Linf': float('nan'),
                'L2_rel': float('nan'), 'Linf_rel': float('nan')}

    if x is not None:
        xa = np.asarray(x, dtype=float)[mask]
        order = np.argsort(xa)
        xa, d_ord, r_ord = xa[order], diff[order], ref[order]
        span = xa[-1] - xa[0]
        _trapz = getattr(np, 'trapezoid', None) or np.trapz  # NumPy 2.x renamed trapz
        if xa.size > 1 and span > 0:
            L2     = np.sqrt(_trapz(d_ord**2, xa) / span)
            L2_ref = np.sqrt(_trapz(r_ord**2, xa) / span)
        else:
            L2     = np.sqrt(np.mean(diff**2))
            L2_ref = np.sqrt(np.mean(ref**2))
    else:
        L2     = np.sqrt(np.mean(diff**2))
        L2_ref = np.sqrt(np.mean(ref**2))

    Linf     = float(np.max(np.abs(diff)))
    Linf_ref = float(np.max(np.abs(ref)))
    return {
        'L2'      : float(L2),
        'Linf'    : Linf,
        'L2_rel'  : float(L2 / L2_ref)     if L2_ref   > 0 else float('nan'),
        'Linf_rel': float(Linf / Linf_ref) if Linf_ref > 0 else float('nan'),
    }


def _attach_error_norms(result):
    """
    Inspect a compare_numeric() result dict and compute error norms for every
    value of the small parameter (and every variable, for systems).  Returns a
    dict keyed by eps value.
    """
    errors = {}

    # ODE / oscillator / stepwise: keys 't', 'u_pert'{eps:arr}, 'u_numerical'{eps:arr}
    if 'u_pert' in result and 'u_numerical' in result and 't' in result:
        t = result['t']
        for ev, up in result['u_pert'].items():
            un = result['u_numerical'][ev]
            if isinstance(up, dict):                      # ODE system: {var: arr}
                errors[ev] = {v: error_norms(un[v], up[v], t) for v in up}
            else:
                errors[ev] = error_norms(un, up, t)
        return errors

    # Boundary layer: keys 'x', 'results'{eps:{'u_expansion','u_numerical'}}
    if 'x' in result and 'results' in result:
        x = result['x']
        for ev, d in result['results'].items():
            errors[ev] = error_norms(d['u_numerical'], d['u_expansion'], x)
        return errors

    # Algebraic (scalar per eps): 'eps', 'perturbation', 'numerical'
    if 'eps' in result and 'perturbation' in result and 'numerical' in result:
        eps_arr = np.asarray(result['eps'], dtype=float)
        pert, num = result['perturbation'], result['numerical']
        if isinstance(pert, dict):                        # algebraic system: {var: arr}
            for i, ev in enumerate(eps_arr):
                errors[float(ev)] = {
                    v: _scalar_error(num[v][i], pert[v][i]) for v in pert
                }
        else:
            pert = np.asarray(pert, float); num = np.asarray(num, float)
            for i, ev in enumerate(eps_arr):
                errors[float(ev)] = _scalar_error(num[i], pert[i])
        return errors

    return errors


def _scalar_error(ref, approx):
    """Absolute and relative error for a single scalar (algebraic problems)."""
    ref = float(ref); approx = float(approx)
    abs_err = abs(approx - ref)
    return {
        'abs': abs_err,
        'rel': abs_err / abs(ref) if ref != 0 else float('nan'),
    }


def _solver_settings(hierarchy):
    """
    Report the SciPy solver settings used for the numerical reference, so that
    the comparison is fully reproducible.  The classification follows the same
    dispatch order as compare_numeric().
    """
    from asymptotics.methods.regular_ode         import ODEHierarchy
    from asymptotics.methods.lindstedt           import LindstedtHierarchy
    from asymptotics.methods.multiple_scales     import MultScalesHierarchy
    from asymptotics.methods.boundary_layer      import BoundaryLayerHierarchy
    from asymptotics.core.hierarchy              import OrderHierarchy
    from asymptotics.core.system_hierarchy       import SystemHierarchy
    from asymptotics.methods.regular_ode_system  import ODESystemHierarchy
    from asymptotics.methods.stepwise            import StepwiseHierarchy

    ivp = {'solver': 'scipy.integrate.solve_ivp', 'method': 'RK45',
           'rtol': 1e-10, 'atol': 1e-12, 'reference_label': 'Numerical reference'}
    root = {'solver': 'scipy.optimize.root', 'method': 'hybr',
            'reference_label': 'Numerical reference'}

    if isinstance(hierarchy, StepwiseHierarchy):
        return ivp
    if isinstance(hierarchy, SystemHierarchy):
        return root
    if isinstance(hierarchy, ODESystemHierarchy):
        return ivp
    if isinstance(hierarchy, BoundaryLayerHierarchy):
        return {'solver': 'scipy.integrate.solve_bvp', 'tol': 1e-6,
                'max_nodes': 10000, 'reference_label': 'Numerical reference'}
    if isinstance(hierarchy, ODEHierarchy):
        return ivp
    if isinstance(hierarchy, (LindstedtHierarchy, MultScalesHierarchy)):
        return ivp
    if isinstance(hierarchy, OrderHierarchy):
        return root
    return {'reference_label': 'Numerical reference'}



# ---------------------------------------------------------------------------
# Algebraic — no plot, just return values
# ---------------------------------------------------------------------------

def _compare_algebraic(h, eps_list, n_points=100, problem=None, **kwargs):
    """
    Compare algebraic perturbation expansion against scipy root-finder.
    eps_list is used as the x-axis: plot x(eps) vs exact root over eps_list.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import root

    eps_sym = h.small_param

    if not hasattr(h, '_original_equation') or not hasattr(h, '_dependent_sym'):
        raise ValueError(
            "\n\n  compare_numeric for algebraic equations requires the updated "
            "regular_algebraic.py.\n"
        )

    f_orig   = h._original_equation
    x_sym    = h._dependent_sym
    dep_name = str(getattr(h, '_dependent_sym', 'x'))

    # Use eps_list as x-axis range
    eps_vals = np.array(sorted(eps_list))

    pert_vals = np.array([
        float(h.expansion.subs(eps_sym, ev).subs(kwargs.get('param_subs', {}))) for ev in eps_vals
    ])

    num_vals = []
    for j, ev in enumerate(eps_vals):
        f_fn = lambdify(x_sym, f_orig.subs(eps_sym, ev).subs(kwargs.get('param_subs', {})), 'numpy')
        try:
            sol = root(f_fn, pert_vals[j])
            num_vals.append(float(sol.x[0]) if sol.success else float('nan'))
        except Exception:
            num_vals.append(float('nan'))
    num_vals = np.array(num_vals)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(eps_vals, num_vals,  'k-',   lw=2,   label='Numerical reference', alpha=0.9)
    ax.plot(eps_vals, pert_vals, 'ro--', lw=1.5, ms=5, markevery=max(1, len(eps_vals)//10),
            label=f'Perturbation (order {len(h)-1})')
    ax.set_xlabel('ε')
    ax.set_ylabel(dep_name)
    ax.set_title(f'Algebraic perturbation  —  order {len(h)-1}')
    ax.legend()
    plt.tight_layout()

    return {
        'eps'         : eps_vals,
        'perturbation': pert_vals,
        'numerical'   : num_vals,
        'fig'         : fig,
    }

def _compare_ode(h, eps_list, plot_range=None, n_points=300, problem=None, **kwargs):
    problem = problem or getattr(h, "_problem", None)
    # Limit conditions cannot be reliably handled by scipy BVP solvers
    if problem is not None:
        limit_conds = [c for c in problem.conditions if getattr(c, 'is_limit', False)]
        if limit_conds:
            lc_strs = [str(lc) for lc in limit_conds]
            raise NotImplementedError(
                f"\n\n  compare_numeric does not support limit boundary conditions:\n"
                + "".join(f"    {s}\n" for s in lc_strs) +
                f"\n  Singular BVPs require specialized numerical treatment\n"
                f"  (e.g. Frobenius-type methods or Mathematica's NDSolve).\n"
                f"\n  You can still use the symbolic expansion:\n"
                f"    sol.show()\n"
                f"    sol.eval(eps=0.1, at=t_vals)\n"
                f"    sol.to_latex()\n"
            )
    """
    Compare ODE perturbation expansion against scipy numerical solution.
    eps_list : list of floats — one curve pair per eps value.
    """
    import matplotlib.pyplot as plt

    eps_sym = h.small_param
    t_sym   = h.independent

    if plot_range is None:
        plot_range = _infer_range(h, problem)

    t_vals = np.linspace(plot_range[0], plot_range[1], n_points)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax    = plt.subplots(figsize=(5, 4))
    u_pert_all = {}
    u_num_all  = {}

    for i, eps_val in enumerate(eps_list):
        color   = colors[i % len(colors)]
        comp_fn = lambdify(t_sym, h.expansion.subs(eps_sym, eps_val).subs(kwargs.get("param_subs", {})), "numpy")
        u_pert  = np.real(np.array([complex(comp_fn(ti)) for ti in t_vals]))

        if h._problem_type == 'ivp':
            u_num = _solve_ode_ivp(h, eps_val, plot_range, t_vals, problem, param_subs=kwargs.get("param_subs", {}))
        else:
            u_num = _solve_ode_bvp(h, eps_val, plot_range, t_vals, problem, param_subs=kwargs.get("param_subs", {}))

        ax.plot(t_vals, u_num,  '-',  color=color, lw=2,   alpha=0.85,
                label=f'Numerical reference  ε={eps_val}')
        ax.plot(t_vals, u_pert, '--', color=color, lw=1.5, ms=4,
                marker='o', markevery=30, alpha=0.9,
                label=f'Perturbation  ε={eps_val}')

        u_pert_all[eps_val] = u_pert
        u_num_all[eps_val]  = u_num

    ax.set_xlabel(str(t_sym))
    ax.set_title(f'{h._method}  —  order {len(h)-1}')
    ncol = 2 if len(eps_list) > 1 else 1
    ax.legend(fontsize=8, ncol=ncol)
    plt.tight_layout()

    return {
        't'           : t_vals,
        'u_pert'      : u_pert_all,
        'u_numerical' : u_num_all,
        'fig'         : fig,
    }


def _solve_ode_ivp(h, eps_val, plot_range, t_vals, problem, param_subs=None):
    """Solve IVP numerically using solve_ivp."""
    from scipy.integrate import solve_ivp as _solve_ivp

    rhs_fn = _build_ode_rhs(h, eps_val, problem, param_subs=param_subs)
    ics    = _get_ics(h, problem, eps_val=eps_val, param_subs=param_subs)

    sol = _solve_ivp(
        rhs_fn, plot_range, ics,
        dense_output=True, rtol=1e-10, atol=1e-12
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.sol(t_vals)[0]


def _solve_ode_bvp(h, eps_val, plot_range, t_vals, problem, param_subs=None):
    """Solve BVP numerically using solve_bvp."""
    from scipy.integrate import solve_bvp as _solve_bvp

    rhs_fn, bc_fn, a_bc, b_bc = _build_bvp_rhs_bc(h, eps_val, problem, plot_range, param_subs=param_subs)

    x_grid = np.linspace(a_bc, b_bc, 50)
    ode_order = getattr(problem, "ode_order", 2)
    y0     = np.zeros((ode_order, len(x_grid)))
    y0[0]  = np.linspace(0, 1, len(x_grid))

    sol = _solve_bvp(rhs_fn, bc_fn, x_grid, y0, tol=1e-8, max_nodes=5000)
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    return sol.sol(t_vals)[0]


# ---------------------------------------------------------------------------
# Oscillator methods (Lindstedt, Multiple scales)
# ---------------------------------------------------------------------------

def _compare_ode_oscillator(h, eps_list, plot_range=None, n_points=500,
                             problem=None, **kwargs):
    """Compare oscillator expansion (Lindstedt or multiple scales).
    eps_list : list of floats — one curve pair per eps value.
    """
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp as _solve_ivp

    eps_sym = h.small_param
    t_sym   = h.independent

    if plot_range is None:
        plot_range = _infer_range(h, problem)

    t_vals = np.linspace(plot_range[0], plot_range[1], n_points)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax    = plt.subplots(figsize=(5, 4))
    u_pert_all = {}
    u_num_all  = {}

    param_subs = kwargs.get('param_subs', {})

    for i, eps_val in enumerate(eps_list):
        color = colors[i % len(colors)]

        if hasattr(h, 'expansion_t'):
            comp_expr = h.expansion_t.subs(eps_sym, eps_val).subs(param_subs)
        else:
            comp_expr = h.expansion.subs(eps_sym, eps_val).subs(param_subs)

        comp_fn = lambdify(t_sym, comp_expr, 'numpy')
        try:
            u_pert = np.real(np.array([complex(comp_fn(ti)) for ti in t_vals]))
        except Exception:
            u_pert = comp_fn(t_vals)

        rhs_fn = _build_ode_rhs(h, eps_val, problem, param_subs=param_subs)
        ics    = _get_ics(h, problem, eps_val=eps_val, param_subs=param_subs)
        sol    = _solve_ivp(rhs_fn, plot_range, ics,
                            dense_output=True, rtol=1e-10, atol=1e-12)
        u_num  = sol.sol(t_vals)[0]

        ax.plot(t_vals, u_num,  '-',  color=color, lw=2,   alpha=0.85,
                label=f'Numerical reference  ε={eps_val}')
        ax.plot(t_vals, u_pert, '--', color=color, lw=1.5, ms=4,
                marker='o', markevery=30, alpha=0.9,
                label=f'Perturbation  ε={eps_val}')

        u_pert_all[eps_val] = u_pert
        u_num_all[eps_val]  = u_num

    ax.set_xlabel(str(t_sym))
    ax.set_title(f'{h._method}  —  order {len(h)-1}')
    ax.legend(fontsize=8, ncol=2 if len(eps_list) > 1 else 1)
    plt.tight_layout()

    return {
        't'           : t_vals,
        'u_pert'      : u_pert_all,
        'u_numerical' : u_num_all,
        'fig'         : fig,
    }


# ---------------------------------------------------------------------------
# Boundary layer
# ---------------------------------------------------------------------------

def _compare_boundary_layer(h, eps_list, n_points=400, problem=None, **kwargs):
    """
    Compare boundary layer expansion against solve_bvp.
    Shows outer, inner, AND expansion. One subplot per eps value.
    """
    import matplotlib.pyplot as plt

    eps_sym    = h.small_param
    x_sym      = h.independent
    layer_side = 'left' if '0' in h.layer_location else 'right'
    x_vals     = np.linspace(0, 1, n_points)

    n_eps = len(eps_list)
    fig, axes = plt.subplots(1, n_eps, figsize=(5*n_eps, 4), squeeze=False)

    results = {ev: {} for ev in eps_list}

    for i, eps_val in enumerate(eps_list):
        ax = axes[0][i]

        def _eval(expr):
            sub = expr.subs(eps_sym, eps_val)
            fn  = lambdify(x_sym, sub, 'numpy')
            vals = fn(x_vals)
            return np.real(np.broadcast_to(
                np.atleast_1d(np.array(vals, dtype=complex)), x_vals.shape
            ))

        u_outer = _eval(h.outer)
        u_inner = _eval(h.inner_xi)
        u_comp  = _eval(h.expansion)

        rhs_fn, bc_fn, _, __ = _build_bvp_rhs_bc(h, eps_val, problem, [0, 1])
        x_num = _layer_aware_grid(eps_val, layer_side, n_points)
        y0    = np.zeros((2, len(x_num)))
        y0[0] = np.real(_eval_at(h.expansion, eps_sym, eps_val, x_sym, x_num))

        from scipy.integrate import solve_bvp as _solve_bvp
        sol = _solve_bvp(rhs_fn, bc_fn, x_num, y0, tol=1e-6, max_nodes=10000)
        if not sol.success:
            x_num = np.linspace(0, 1, 200)
            y0    = np.zeros((2, 200))
            y0[0] = np.real(_eval_at(h.expansion, eps_sym, eps_val, x_sym, x_num))
            sol   = _solve_bvp(rhs_fn, bc_fn, x_num, y0, tol=1e-5, max_nodes=10000)
        u_num = sol.sol(x_vals)[0]

        ax.plot(x_vals, u_num,   'k-',   lw=2,   label='Numerical reference', alpha=0.9)
        ax.plot(x_vals, u_comp,  'ro--', lw=1.5, ms=4, markevery=25, label='Expansion')
        ax.plot(x_vals, u_outer, 'b:',   lw=2,   label='Outer')
        ax.plot(x_vals, u_inner, 'g-.',  lw=1.8, label='Inner', alpha=0.85)

        thickness = 5 * eps_val
        if layer_side == 'left':
            ax.axvspan(0, min(thickness, 0.3), alpha=0.07, color='blue')
        else:
            ax.axvspan(max(1-thickness, 0.7), 1, alpha=0.07, color='blue')

        ax.set_title(f'ε = {eps_val}  (layer: {h.layer_location})')
        ax.set_xlabel(str(x_sym))
        ax.legend(fontsize=8)

        results[eps_val] = {
            'u_outer': u_outer, 'u_inner': u_inner,
            'u_expansion': u_comp, 'u_numerical': u_num,
        }

    plt.suptitle(f'Matched Asymptotic Expansions', fontsize=11, y=1.02)
    plt.tight_layout()

    return {
        'x'      : x_vals,
        'results': results,
        'fig'    : fig,
    }

def _compare_ode_system(h, eps_list, plot_range=None, n_points=300,
                        problem=None, **kwargs):
    """
    Compare coupled ODE system expansion against scipy.
    eps_list : list of floats — one curve pair per eps value per variable.
    """
    import matplotlib.pyplot as plt

    eps_sym   = h.small_param
    t_sym     = h.independent
    variables = h.variables

    if plot_range is None:
        plot_range = _infer_range(h, problem)

    t_vals = np.linspace(plot_range[0], plot_range[1], n_points)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_vars = len(variables)

    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4), squeeze=False)

    u_pert_all = {ev: {} for ev in eps_list}
    u_num_all  = {ev: {} for ev in eps_list}

    # Compute the starting index of each variable in the flat state vector
    # State layout: [u, du, ..., v, dv, ...] — order entries per variable
    state_start = {}
    _idx = 0
    for _var in variables:
        state_start[_var] = _idx
        _idx += problem.ode_orders[_var]

    for i, eps_val in enumerate(eps_list):
        color  = colors[i % len(colors)]
        rhs_fn = _build_system_rhs(problem, eps_val)
        ics    = _get_system_ics(problem)

        from scipy.integrate import solve_ivp as _solve_ivp
        sol = _solve_ivp(rhs_fn, plot_range, ics,
                         dense_output=True, rtol=1e-10, atol=1e-12)

        for j, var in enumerate(variables):
            ax = axes[0][j]
            fn = lambdify(t_sym, h[var].expansion.subs(eps_sym, eps_val), 'numpy')
            try:
                u_pert = np.real(np.array([complex(fn(ti)) for ti in t_vals]))
            except Exception:
                u_pert = np.array([float(fn(ti)) for ti in t_vals])
            u_num = sol.sol(t_vals)[state_start[var]]

            ax.plot(t_vals, u_num,  '-',  color=color, lw=2,   alpha=0.85,
                    label=f'Numerical reference  ε={eps_val}' if j==0 else f'ε={eps_val}')
            ax.plot(t_vals, u_pert, '--', color=color, lw=1.5, ms=4,
                    marker='o', markevery=25, alpha=0.9,
                    label=f'Pert.  ε={eps_val}' if j==0 else None)

            u_pert_all[eps_val][var] = u_pert
            u_num_all[eps_val][var]  = u_num

        for j, var in enumerate(variables):
            axes[0][j].set_title(f'{var}(t)  —  order {len(h[var])-1}')
            axes[0][j].set_xlabel(str(t_sym))
            axes[0][j].legend(fontsize=7)

    plt.suptitle(h._method, fontsize=11, y=1.02)
    plt.tight_layout()

    return {
        't'           : t_vals,
        'u_pert'      : u_pert_all,
        'u_numerical' : u_num_all,
        'fig'         : fig,
    }

def _build_system_rhs(problem, eps_val):
    """Build scipy RHS for the full ODE system."""
    from sympy import solve as _solve, Symbol as _Sym, diff as _diff

    eps_sym  = problem.small_param
    t_sym    = problem.independent
    deps     = problem.dependent_names
    dep_syms = problem._dep_syms

    # For each equation, solve for the highest derivative
    rhs_exprs = []
    for dep in deps:
        f_orig    = problem.equations[dep].subs(eps_sym, eps_val)
        order     = problem.ode_orders[dep]
        deriv_sym = problem._deriv_syms[dep][order]

        # Substitute all dep symbols and lower deriv symbols as plain symbols
        f_sub = f_orig
        solved = _solve(f_sub, deriv_sym)
        if not solved:
            raise RuntimeError(
                f"Could not solve equation for highest derivative of '{dep}'"
            )
        rhs_exprs.append(solved[0])

    # Build lambdified functions
    # State vector: [u, du, v, dv, ...] for 2nd order, [u, v, ...] for 1st order
    all_syms = []
    state_map = {}   # dep -> list of state indices
    idx = 0
    for dep in deps:
        order = problem.ode_orders[dep]
        state_map[dep] = list(range(idx, idx + order))
        for k in range(order):
            all_syms.append(Symbol(dep if k == 0 else f"{'d'*k}{dep}"))
        idx += order

    # Build substitution mapping: dep_sym -> y[i], deriv_sym -> y[j]
    def rhs(t_val, y):
        subs = {t_sym: t_val}
        for dep in deps:
            indices = state_map[dep]
            subs[dep_syms[dep]] = y[indices[0]]
            for k_deriv, dsym in problem._deriv_syms[dep].items():
                if k_deriv < len(indices):
                    subs[dsym] = y[indices[k_deriv]]

        result = []
        for dep in deps:
            order   = problem.ode_orders[dep]
            indices = state_map[dep]
            # Add lower derivatives to state
            for k in range(1, order):
                result.append(float(y[indices[k]]))
            # Add highest derivative
            expr = rhs_exprs[deps.index(dep)]
            result.append(float(expr.subs(subs)))

        return result

    return rhs


def _get_system_ics(problem):
    """Extract initial conditions as a flat state vector."""
    deps = problem.dependent_names
    ics  = []
    for dep in deps:
        order     = problem.ode_orders[dep]
        conds_dep = sorted(problem.conditions.get(dep, []),
                           key=lambda c: c.deriv_order)
        for cond in conds_dep:
            ics.append(float(cond.value))
    return ics


# ---------------------------------------------------------------------------
# Algebraic System
# ---------------------------------------------------------------------------

def _compare_algebraic_system(h, eps_list, problem=None, **kwargs):
    """
    Compare coupled algebraic system expansion against scipy.optimize.root.
    Plots perturbation vs numerical root for each variable over eps range.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import root
    from sympy import lambdify

    if not hasattr(h, '_original_equations') or not hasattr(h, '_dependent_syms'):
        raise ValueError(
            "\n\n  compare_numeric for algebraic systems requires _original_equations "
            "and _dependent_syms on the hierarchy.\n"
            "  Re-run expand_regular() with the current version of asymptotics.\n"
        )

    eps_sym      = h.small_param
    dep_syms     = h._dependent_syms
    orig_eqs     = h._original_equations
    variables    = list(h.hierarchies.keys())
    n_vars       = len(variables)
    param_subs   = kwargs.get('param_subs', {})

    eps_vals = np.array(sorted(eps_list))

    # Build a lambdified residual F(vec, eps_val) -> list of n residuals
    # Apply any symbolic parameter substitutions first
    eqs_subbed = [f.subs(param_subs) for f in orig_eqs]
    residual_fn = lambdify([dep_syms, eps_sym], eqs_subbed, "numpy")

    # Leading-order solution as initial guess (evaluated at eps=0)
    x0 = np.array([
        float(complex(h.hierarchies[v].expansion.subs(eps_sym, 0)
                      .subs(param_subs).evalf()).real)
        for v in variables
    ])

    # Solve numerically at each eps value
    num_results = {v: np.empty(len(eps_vals)) for v in variables}
    for i, ev in enumerate(eps_vals):
        res = root(lambda vec: residual_fn(vec, ev), x0)
        for j, v in enumerate(variables):
            num_results[v][i] = res.x[j]
        x0 = res.x  # warm-start next solve

    # Perturbation values
    pert_results = {}
    order = max(e.order for e in next(iter(h.hierarchies.values())).entries)
    for v in variables:
        vh = h.hierarchies[v]
        pert_results[v] = np.array([
            complex(vh.expansion.subs(eps_sym, ev).subs(param_subs).evalf()).real
            for ev in eps_vals
        ])

    # Plot
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4), squeeze=False)
    for j, v in enumerate(variables):
        ax = axes[0][j]
        ax.plot(eps_vals, num_results[v],  'b-',   lw=2,   label='Numerical (root)')
        ax.plot(eps_vals, pert_results[v], 'ro--', lw=1.5, ms=5,
                markevery=max(1, len(eps_vals) // 10),
                label=f'Perturbation (order {order})')
        ax.set_title(f'{v}(ε)')
        ax.set_xlabel('ε')
        ax.legend(fontsize=8)

    plt.suptitle('Algebraic system — regular perturbation vs numerical', fontsize=11)
    plt.tight_layout()

    return {
        'eps'        : eps_vals,
        'perturbation': pert_results,
        'numerical'  : num_results,
        'fig'        : fig,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ode_rhs(h, eps_val, problem, param_subs=None):
    """
    Build a scipy-compatible RHS for an ODE of any order N.
    State vector: y = [u, u', u'', ..., u^(N-1)]
    Returns f(t, y) with len N.
    """
    if problem is None:
        raise ValueError(
            "\n\n  compare_numeric needs the original problem object.\n"
        )

    f_orig    = problem.equation
    eps_sym   = problem.small_param
    t_sym     = problem._indep_sym
    u_sym     = Symbol(problem._dependent_name)
    ode_order = problem.ode_order
    deriv_syms = problem._deriv_syms  # {1: du, 2: d2u, ...}

    param_subs = param_subs or {}
    f_at_eps   = f_orig.subs(eps_sym, eps_val).subs(param_subs)

    # Solve for the highest derivative
    highest_sym = deriv_syms.get(ode_order)
    rhs_sols = solve(f_at_eps, highest_sym)
    if not rhs_sols:
        raise ValueError(
            f"\n\n  Could not solve equation for highest derivative: {f_at_eps}\n"
        )
    highest_expr = rhs_sols[0]

    # Build lambdify args: [t, u, du, d2u, ..., d(N-1)u]
    lam_args = [t_sym, u_sym] + [deriv_syms[k] for k in range(1, ode_order) if k in deriv_syms]
    highest_fn = lambdify(lam_args, highest_expr, "numpy")

    # State: y[0]=u, y[1]=u', ..., y[N-1]=u^(N-1)
    n_args = len(lam_args)

    def rhs(t, y):
        # dy[k]/dt = y[k+1] for k < N-1
        # dy[N-1]/dt = f(t, y[0], ..., y[N-1])
        # lam_args = [t, u, du, ..., d(N-1)u] — always ode_order+1 args
        args = [t] + list(y[:ode_order])
        return list(y[1:ode_order]) + [float(highest_fn(*args))]

    return rhs


def _build_bvp_rhs_bc(h, eps_val, problem, x_range, param_subs=None):
    """
    Build RHS and BC functions for solve_bvp, any ODE order.

    Limit conditions are approximated as near-point conditions at delta=1e-4.
    Warning is printed when limit conditions are present.
    """
    if problem is None:
        raise ValueError("\n\n  compare_numeric needs the original problem object.\n")

    from asymptotics.core.conditions import LimitCondition as _LC
    import warnings

    param_subs  = param_subs or {}
    # Include eps -> eps_val in substitutions so BCs like "u(0) = 1 + eps" evaluate correctly
    eps_subs = {}
    if hasattr(h, 'small_param') and h.small_param is not None:
        eps_subs = {h.small_param: eps_val}
    all_bc_subs = {**eps_subs, **param_subs}
    delta       = 1e-4
    ode_order   = problem.ode_order
    point_conds = [c for c in problem.conditions if not getattr(c, 'is_limit', False)]
    limit_conds = [c for c in problem.conditions if getattr(c, 'is_limit', False)]

    if limit_conds:
        warnings.warn(
            "\n  Limit boundary conditions are approximated numerically.\n"
            f"  Using delta={delta} as approximate singular boundary.\n"
            "  Results near the singular point may be less accurate.",
            UserWarning, stacklevel=4
        )

    # Build the scalar ODE RHS
    rhs_ode = _build_ode_rhs(h, eps_val, problem, param_subs=param_subs)

    # scipy solve_bvp needs f(x, y) returning (N, M) array
    def rhs_bvp(x, y):
        dudt = np.zeros_like(y)
        for i in range(y.shape[1]):
            r = rhs_ode(x[i], y[:, i])
            for j in range(ode_order):
                dudt[j, i] = r[j]
        return dudt

    # Determine boundary points
    bc_by_point = {}  # {float_point: [(deriv_order, float_value), ...]}
    for c in point_conds:
        pt = float(c.point.evalf())
        val = float(c.value.subs(all_bc_subs) if hasattr(c.value, 'subs') else c.value)
        bc_by_point.setdefault(pt, []).append((c.deriv_order, val))

    pts_sorted = sorted(bc_by_point.keys())

    if len(pts_sorted) == 1 and limit_conds:
        # One standard boundary + limit at singular point
        reg_pt   = pts_sorted[0]
        sing_pts = [float(lc.point.evalf()) for lc in limit_conds]
        sing_pt  = min(sing_pts)

        if sing_pt < reg_pt:
            a, b    = sing_pt + delta, reg_pt
            reg_end = 'b'  # regular conditions at b
        else:
            a, b    = reg_pt, sing_pt - delta
            reg_end = 'a'

        reg_conds = bc_by_point[reg_pt]

        def bc_bvp(ya, yb):
            reg_state = yb if reg_end == 'b' else ya
            sing_state = ya if reg_end == 'b' else yb
            residuals = []
            for order, val in sorted(reg_conds, key=lambda x: x[0]):
                residuals.append(reg_state[order] - val)
            # Limit condition: approximate as 0 at delta
            # Use deriv_order from limit condition expression
            # Default: u itself should be 0 at singular boundary
            for lc in limit_conds:
                # Approximate: lim condition → value at delta ≈ 0
                residuals.append(sing_state[0] - float(lc.value))
            return residuals

    elif len(pts_sorted) == 2:
        a_pt, b_pt   = pts_sorted[0], pts_sorted[1]
        a, b         = a_pt, b_pt
        a_conds      = bc_by_point[a_pt]
        b_conds      = bc_by_point[b_pt]

        def bc_bvp(ya, yb):
            residuals = []
            for order, val in sorted(a_conds, key=lambda x: x[0]):
                residuals.append(ya[order] - val)
            for order, val in sorted(b_conds, key=lambda x: x[0]):
                residuals.append(yb[order] - val)
            return residuals

    else:
        raise ValueError(
            "\n\n  Cannot determine BVP boundary points from conditions.\n"
        )

    return rhs_bvp, bc_bvp, a, b


def _get_ics(h, problem, eps_val=None, param_subs=None):
    """Extract initial conditions as a list [u0, u'0, ...].
    Substitutes the eps value and any symbolic parameter values.
    """
    if problem is None:
        raise ValueError(
            "\n\n  compare_numeric needs the original problem object.\n"
            "  Call: sol.compare_numeric(eps=0.1, problem=eq)\n"
        )
    param_subs = param_subs or {}
    limit_conds = [c for c in problem.conditions if getattr(c, 'is_limit', False)]
    if limit_conds:
        import warnings
        warnings.warn(
            "\n  Limit boundary conditions cannot be enforced directly in scipy solvers.\n"
            "  They are skipped in compare_numeric. The numerical solution may differ\n"
            "  from the perturbation solution near singular points.",
            UserWarning, stacklevel=4
        )
    conds = sorted([c for c in problem.conditions if not getattr(c, 'is_limit', False)], key=lambda c: c.deriv_order)
    # Build a combined substitution that includes eps -> eps_val
    eps_subs = {}
    if eps_val is not None and hasattr(h, 'small_param') and h.small_param is not None:
        eps_subs = {h.small_param: eps_val}
    all_subs = {**eps_subs, **param_subs}
    ics = []
    for c in conds:
        if float(c.point.evalf()) == float(min(c.point for c in conds).evalf()):
            val = c.value.subs(all_subs) if hasattr(c.value, 'subs') else c.value
            try:
                ics.append(float(val))
            except Exception:
                raise ValueError(
                    f"\n\n  Could not evaluate initial condition value: {c.value}\n"
                    f"  After substitution (eps={eps_val}, params={param_subs}): {val}\n"
                    f"  Provide extra values: sol.compare_numeric(eps=..., params={{...}})\n"
                )
    return ics


def _infer_range(h, problem):
    """
    Infer the plotting domain from the problem's conditions.
    For IVPs: [t0, t0 + 10] where t0 is the IC point.
    For BVPs: [a, b] from the two boundary points.
    For oscillators: 6 periods.
    Falls back to [0, 10] if nothing can be determined.
    """
    # Try to read boundary points from problem conditions
    if problem is not None and hasattr(problem, 'conditions'):
        conds = problem.conditions
        if isinstance(conds, list) and len(conds) >= 1:
            try:
                pts = sorted(set(float(c.point.evalf()) for c in conds))
                if len(pts) == 2:
                    # BVP: use the two boundary points
                    return [pts[0], pts[1]]
                elif len(pts) == 1:
                    # IVP: start at IC point, plot for 10 units
                    return [pts[0], pts[0] + 10]
            except Exception:
                pass
        elif isinstance(conds, dict):
            # ODESystem: conditions is a dict of {var: [ParsedCondition]}
            try:
                all_pts = []
                for var_conds in conds.values():
                    all_pts.extend(float(c.point.evalf()) for c in var_conds)
                pts = sorted(set(all_pts))
                if len(pts) == 1:
                    return [pts[0], pts[0] + 10]
                elif len(pts) == 2:
                    return [pts[0], pts[1]]
            except Exception:
                pass

    # Oscillator fallback: 6 periods
    if hasattr(h, 'omega_0'):
        try:
            omega0 = float(h.omega_0)
            return [0, 6 * np.pi / omega0]
        except Exception:
            pass

    # Final fallback
    return [0, 10]


def _layer_aware_grid(eps_val, layer_side, n_points):
    """Build a grid denser near the boundary layer."""
    thickness = max(10 * eps_val, 0.05)
    n_dense   = n_points // 2
    n_coarse  = n_points - n_dense

    if layer_side == 'left':
        x1 = np.linspace(0, min(thickness, 0.4), n_dense)
        x2 = np.linspace(min(thickness, 0.4), 1, n_coarse)
    else:
        x1 = np.linspace(0, max(1 - thickness, 0.6), n_coarse)
        x2 = np.linspace(max(1 - thickness, 0.6), 1, n_dense)

    return np.unique(np.concatenate([x1, x2]))


def _eval_at(expr, eps_sym, eps_val, x_sym, x_arr):
    """Evaluate a SymPy expression at an array of x values."""
    sub = expr.subs(eps_sym, eps_val)
    fn  = lambdify(x_sym, sub, 'numpy')
    vals = fn(x_arr)
    return np.broadcast_to(np.atleast_1d(np.array(vals, dtype=complex)),
                            x_arr.shape)
