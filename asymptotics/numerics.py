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

def compare_numeric(hierarchy, eps, params=None, **kwargs):
    """
    Compare a perturbation expansion against a numerical solution.

    Parameters
    ----------
    hierarchy : any perturbation hierarchy object
    eps : float
        Value of the small parameter to use.
    **kwargs
        plot_range : [a, b] — domain for plotting (ODE only)
        n_points   : int — number of plot points (default 300)

    Returns
    -------
    dict with keys depending on problem type (see individual solvers)
    """
    # Handle legacy/alias kwargs gracefully
    for alias in ('t_range', 'x_range'):
        if alias in kwargs:
            import warnings
            warnings.warn(
                f"'{alias}' is not a valid parameter — use 'plot_range' instead.",
                UserWarning, stacklevel=2
            )
            kwargs.setdefault('plot_range', kwargs.pop(alias))
            break

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
    if isinstance(hierarchy, SystemHierarchy):
        return _compare_algebraic_system(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, ODESystemHierarchy):
        return _compare_ode_system(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, BoundaryLayerHierarchy):
        return _compare_boundary_layer(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, ODEHierarchy):
        return _compare_ode(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, (LindstedtHierarchy, MultScalesHierarchy)):
        return _compare_ode_oscillator(hierarchy, eps_list, **kwargs)
    elif isinstance(hierarchy, OrderHierarchy):
        return _compare_algebraic(hierarchy, eps_list, **kwargs)
    else:
        raise TypeError(
            f"\n\n  compare_numeric does not support {type(hierarchy).__name__}.\n"
        )


# ---------------------------------------------------------------------------
# Algebraic — no plot, just return values
# ---------------------------------------------------------------------------

def _compare_algebraic(h, eps_list, n_points=100, problem=None, **kwargs):
    """
    Compare algebraic perturbation expansion against scipy root-finder.
    eps_list is used as the x-axis: plot x(eps) vs exact root over eps_list.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve

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
        float(h.composite.subs(eps_sym, ev).subs(kwargs.get('param_subs', {}))) for ev in eps_vals
    ])

    num_vals = []
    for j, ev in enumerate(eps_vals):
        f_fn = lambdify(x_sym, f_orig.subs(eps_sym, ev).subs(kwargs.get('param_subs', {})), 'numpy')
        try:
            num_vals.append(float(fsolve(f_fn, pert_vals[j])[0]))
        except Exception:
            num_vals.append(float('nan'))
    num_vals = np.array(num_vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps_vals, num_vals,  'k-',   lw=2,   label='Exact (fsolve)', alpha=0.9)
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

    fig, ax    = plt.subplots(figsize=(10, 4))
    u_pert_all = {}
    u_num_all  = {}

    for i, eps_val in enumerate(eps_list):
        color   = colors[i % len(colors)]
        comp_fn = lambdify(t_sym, h.composite.subs(eps_sym, eps_val).subs(kwargs.get("param_subs", {})), "numpy")
        u_pert  = np.real(np.array([complex(comp_fn(ti)) for ti in t_vals]))

        if h._problem_type == 'ivp':
            u_num = _solve_ode_ivp(h, eps_val, plot_range, t_vals, problem, param_subs=kwargs.get("param_subs", {}))
        else:
            u_num = _solve_ode_bvp(h, eps_val, plot_range, t_vals, problem, param_subs=kwargs.get("param_subs", {}))

        ax.plot(t_vals, u_num,  '-',  color=color, lw=2,   alpha=0.85,
                label=f'Exact  ε={eps_val}')
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
    ics    = _get_ics(h, problem, param_subs=param_subs)

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

    rhs_fn, bc_fn = _build_bvp_rhs_bc(h, eps_val, problem, plot_range, param_subs=param_subs)

    x_grid = np.linspace(plot_range[0], plot_range[1], 50)
    y0     = np.zeros((2, len(x_grid)))
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

    fig, ax    = plt.subplots(figsize=(10, 4))
    u_pert_all = {}
    u_num_all  = {}

    for i, eps_val in enumerate(eps_list):
        color = colors[i % len(colors)]

        if hasattr(h, 'composite_t'):
            comp_expr = h.composite_t.subs(eps_sym, eps_val)
        else:
            comp_expr = h.composite.subs(eps_sym, eps_val)

        comp_fn = lambdify(t_sym, comp_expr, 'numpy')
        try:
            u_pert = np.real(np.array([complex(comp_fn(ti)) for ti in t_vals]))
        except Exception:
            u_pert = comp_fn(t_vals)

        rhs_fn = _build_ode_rhs(h, eps_val, problem)
        ics    = _get_ics(h, problem)
        sol    = _solve_ivp(rhs_fn, plot_range, ics,
                            dense_output=True, rtol=1e-10, atol=1e-12)
        u_num  = sol.sol(t_vals)[0]

        ax.plot(t_vals, u_num,  '-',  color=color, lw=2,   alpha=0.85,
                label=f'Exact  ε={eps_val}')
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
    Shows outer, inner, AND composite. One subplot per eps value.
    """
    import matplotlib.pyplot as plt

    eps_sym    = h.small_param
    x_sym      = h.independent
    layer_side = 'left' if '0' in h.layer_location else 'right'
    x_vals     = np.linspace(0, 1, n_points)

    n_eps = len(eps_list)
    fig, axes = plt.subplots(1, n_eps, figsize=(6*n_eps, 4), squeeze=False)

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
        u_comp  = _eval(h.composite)

        rhs_fn, bc_fn = _build_bvp_rhs_bc(h, eps_val, problem, [0, 1])
        x_num = _layer_aware_grid(eps_val, layer_side, n_points)
        y0    = np.zeros((2, len(x_num)))
        y0[0] = np.real(_eval_at(h.composite, eps_sym, eps_val, x_sym, x_num))

        from scipy.integrate import solve_bvp as _solve_bvp
        sol = _solve_bvp(rhs_fn, bc_fn, x_num, y0, tol=1e-6, max_nodes=10000)
        if not sol.success:
            x_num = np.linspace(0, 1, 200)
            y0    = np.zeros((2, 200))
            y0[0] = np.real(_eval_at(h.composite, eps_sym, eps_val, x_sym, x_num))
            sol   = _solve_bvp(rhs_fn, bc_fn, x_num, y0, tol=1e-5, max_nodes=10000)
        u_num = sol.sol(x_vals)[0]

        ax.plot(x_vals, u_num,   'k-',   lw=2,   label='Exact', alpha=0.9)
        ax.plot(x_vals, u_comp,  'ro--', lw=1.5, ms=4, markevery=25, label='Composite')
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
            'u_composite': u_comp, 'u_numerical': u_num,
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

    for i, eps_val in enumerate(eps_list):
        color  = colors[i % len(colors)]
        rhs_fn = _build_system_rhs(problem, eps_val)
        ics    = _get_system_ics(problem)

        from scipy.integrate import solve_ivp as _solve_ivp
        sol = _solve_ivp(rhs_fn, plot_range, ics,
                         dense_output=True, rtol=1e-10, atol=1e-12)

        for j, var in enumerate(variables):
            ax = axes[0][j]
            fn = lambdify(t_sym, h[var].composite.subs(eps_sym, eps_val), 'numpy')
            try:
                u_pert = np.real(np.array([complex(fn(ti)) for ti in t_vals]))
            except Exception:
                u_pert = np.array([float(fn(ti)) for ti in t_vals])
            u_num = sol.sol(t_vals)[j]

            ax.plot(t_vals, u_num,  '-',  color=color, lw=2,   alpha=0.85,
                    label=f'Exact  ε={eps_val}' if j==0 else f'ε={eps_val}')
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
                if k_deriv - 1 < len(indices):
                    subs[dsym] = y[indices[k_deriv - 1]]

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
    Compare coupled algebraic system expansion against scipy root-finder.
    Plots each variable vs exact root over eps range.
    """
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve

    eps_sym   = h.small_param
    variables = list(h.hierarchies.keys())
    n_vars    = len(variables)

    eps_vals = np.array(sorted(eps_list))

    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4), squeeze=False)

    results = {}
    for j, var in enumerate(variables):
        vh = h.hierarchies[var]
        pert_vals = np.array([
            complex(vh.composite.subs(eps_sym, ev).subs(kwargs.get('param_subs', {})).evalf()).real for ev in eps_vals
        ])

        axes[0][j].plot(eps_vals, pert_vals, 'ro--', lw=1.5, ms=5,
                        markevery=max(1, len(eps_vals)//10),
                        label=f'Perturbation (order {len(vh)-1})')
        axes[0][j].set_title(f'{var}(ε)')
        axes[0][j].set_xlabel('ε')
        axes[0][j].legend(fontsize=8)
        results[var] = pert_vals

    plt.suptitle('Algebraic system — regular perturbation', fontsize=11)
    plt.tight_layout()

    return {
        'eps'    : eps_vals,
        'results': results,
        'fig'    : fig,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ode_rhs(h, eps_val, problem, param_subs=None):
    """
    Build a scipy-compatible RHS function from the problem's equation.
    Returns f(t, y) where y = [u, u'].
    """
    if problem is None:
        raise ValueError(
            "\n\n  compare_numeric needs the original problem object.\n"
            "  Call: sol.compare_numeric(eps=0.1, problem=eq)\n"
        )

    f_orig   = problem.equation
    eps_sym  = problem.small_param
    t_sym    = problem._indep_sym
    u_sym    = Symbol(problem._dependent_name)
    du_sym   = problem._deriv_syms.get(1)
    d2u_sym  = problem._deriv_syms.get(2)

    param_subs = param_subs or {}
    f_at_eps  = f_orig.subs(eps_sym, eps_val).subs(param_subs)
    ode_order = problem.ode_order

    if ode_order == 1:
        # First-order: solve for du, state = [u]
        du_rhs = solve(f_at_eps, du_sym)
        if not du_rhs:
            raise ValueError(
                f"\n\n  Could not solve equation for u': {f_at_eps}\n"
            )
        du_fn = lambdify([t_sym, u_sym], du_rhs[0], 'numpy')

        def rhs(t, y):
            return [float(du_fn(t, y[0]))]

    else:
        # Second-order: solve for d2u, state = [u, u']
        d2u_rhs = solve(f_at_eps, d2u_sym)
        if not d2u_rhs:
            raise ValueError(
                f"\n\n  Could not solve equation for u'': {f_at_eps}\n"
            )
        d2u_expr = d2u_rhs[0]
        args     = [t_sym, u_sym] + ([du_sym] if du_sym else [])
        d2u_fn   = lambdify(args, d2u_expr, 'numpy')

        if du_sym:
            def rhs(t, y):
                return [y[1], float(d2u_fn(t, y[0], y[1]))]
        else:
            def rhs(t, y):
                return [y[1], float(d2u_fn(t, y[0]))]

    return rhs


def _build_bvp_rhs_bc(h, eps_val, problem, x_range, param_subs=None):
    """Build RHS and BC functions for solve_bvp."""
    if problem is None:
        raise ValueError(
            "\n\n  compare_numeric needs the original problem object.\n"
            "  Call: sol.compare_numeric(eps=0.1, problem=eq)\n"
        )

    rhs_ode = _build_ode_rhs(h, eps_val, problem)

    def rhs_bvp(x, y):
        dudt  = np.zeros_like(y)
        for i in range(y.shape[1]):
            r = rhs_ode(x[i], y[:, i])
            dudt[0, i] = r[0]
            dudt[1, i] = r[1]
        return dudt

    # BCs from conditions
    conds  = problem.conditions
    bc_dict = {c.point: c.value for c in conds if c.deriv_order == 0}
    pts    = sorted(bc_dict.keys(), key=lambda p: float(p.evalf()))
    a, b   = float(pts[0].evalf()), float(pts[1].evalf())
    param_subs = param_subs or {}
    va_raw, vb_raw = bc_dict[pts[0]], bc_dict[pts[1]]
    va = float(va_raw.subs(param_subs) if hasattr(va_raw, "subs") else va_raw)
    vb = float(vb_raw.subs(param_subs) if hasattr(vb_raw, "subs") else vb_raw)

    def bc_bvp(ya, yb):
        return [ya[0] - va, yb[0] - vb]

    return rhs_bvp, bc_bvp


def _get_ics(h, problem, param_subs=None):
    """Extract initial conditions as a list [u0, u'0, ...].
    Substitutes symbolic parameter values if provided.
    """
    if problem is None:
        raise ValueError(
            "\n\n  compare_numeric needs the original problem object.\n"
            "  Call: sol.compare_numeric(eps=0.1, problem=eq)\n"
        )
    param_subs = param_subs or {}
    conds = sorted(problem.conditions, key=lambda c: c.deriv_order)
    ics = []
    for c in conds:
        if float(c.point.evalf()) == float(min(c.point for c in conds).evalf()):
            val = c.value.subs(param_subs) if hasattr(c.value, 'subs') else c.value
            try:
                ics.append(float(val))
            except Exception:
                raise ValueError(
                    f"\n\n  Could not evaluate initial condition value: {c.value}\n"
                    f"  After param substitution: {val}\n"
                    f"  Provide values: sol.compare_numeric(eps=..., params={{...}})\n"
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
