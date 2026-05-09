"""
asymptotics.latex_export
=====================
Export perturbation expansion results as LaTeX source.

Usage
-----
>>> sol = eq.expand_lindstedt(order=2)
>>> print(sol.to_latex())                    # print to console
>>> sol.to_latex(filename="duffing.tex")     # save to file

Output format
-------------
Ready-to-paste LaTeX using align environments.
The small parameter is always rendered as \\varepsilon.
"""

from __future__ import annotations
from sympy import latex, Symbol


def _eps_latex(expr, eps_sym):
    """Substitute user's eps symbol -> varepsilon before latex()."""
    if str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, Symbol('varepsilon'))
    return latex(expr)


def _order_remainder(n):
    """Return O(eps^n) string."""
    if n == 1:
        return r'\mathcal{O}(\varepsilon)'
    return r'\mathcal{O}(\varepsilon^{' + str(n) + r'})'


def _write(filename, content):
    """Write content to file or print to console."""
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(content)
        print(f"LaTeX written to: {filename}")
    else:
        print(content)
    return content


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def to_latex(hierarchy, environment='align', show_orders=False,
             filename=None):
    # Guard against string booleans e.g. show_orders='False'
    if isinstance(show_orders, str):
        show_orders = show_orders.lower() not in ('false', '0', 'no', '')
    """
    Export a perturbation hierarchy as LaTeX source.

    Parameters
    ----------
    hierarchy : any perturbation hierarchy
    environment : str
        LaTeX math environment — 'align', 'equation', or 'gather'.
        Default 'align'.
    show_orders : bool
        If True, show each order u_k separately in addition to
        the composite. Default False.
    filename : str, optional
        If given, write output to this file. Otherwise print to console.

    Returns
    -------
    str — the LaTeX source string
    """
    from asymptotics.methods.regular_ode      import ODEHierarchy
    from asymptotics.methods.lindstedt        import LindstedtHierarchy
    from asymptotics.methods.multiple_scales  import MultScalesHierarchy
    from asymptotics.methods.boundary_layer   import BoundaryLayerHierarchy
    from asymptotics.methods.regular_ode_system import ODESystemHierarchy
    from asymptotics.core.hierarchy           import OrderHierarchy

    if isinstance(hierarchy, LindstedtHierarchy):
        content = _latex_lindstedt(hierarchy, environment, show_orders)
    elif isinstance(hierarchy, MultScalesHierarchy):
        content = _latex_multiple_scales(hierarchy, environment, show_orders)
    elif isinstance(hierarchy, BoundaryLayerHierarchy):
        content = _latex_boundary_layer(hierarchy, environment)
    elif isinstance(hierarchy, ODESystemHierarchy):
        content = _latex_ode_system(hierarchy, environment, show_orders)
    elif isinstance(hierarchy, ODEHierarchy):
        content = _latex_ode(hierarchy, environment, show_orders)
    elif isinstance(hierarchy, OrderHierarchy):
        content = _latex_algebraic(hierarchy, environment, show_orders)
    else:
        raise TypeError(
            f"\n\n  to_latex() does not support {type(hierarchy).__name__}.\n"
        )

    return _write(filename, content)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _wrap(body, environment):
    """Wrap LaTeX body in the chosen math environment."""
    if environment == 'equation':
        # Single equation — use just the last line without alignment
        return f"\\begin{{equation}}\n{body}\n\\end{{equation}}"
    elif environment == 'gather':
        return f"\\begin{{gather}}\n{body}\n\\end{{gather}}"
    else:  # align (default)
        return f"\\begin{{align}}\n{body}\n\\end{{align}}"


def _comment(text):
    return f"% {text}\n"


# ---------------------------------------------------------------------------
# Algebraic
# ---------------------------------------------------------------------------

def _latex_algebraic(h, environment, show_orders):
    eps = h.small_param
    # Use original dependent name if stored, else fall back to symbol
    dep = str(getattr(h, '_dependent_sym', h.entries[0].symbol))
    N   = len(h.entries) - 1
    lines = []

    lines.append(_comment(f"Regular perturbation — algebraic"))

    if show_orders:
        lines.append(_comment("Order-by-order solutions"))
        lines.append(_wrap(
            " \\\\\n".join(
                f"  {dep}_{{{e.order}}} &= {_eps_latex(e.solution, eps)}"
                for e in h.entries
            ),
            environment
        ))
        lines.append("\n")

    lines.append(_comment("Composite expansion"))
    comp = _eps_latex(h.composite, eps)
    remainder = _order_remainder(N + 1)
    lines.append(_wrap(
        f"  {dep}(\\varepsilon) &= {comp} + {remainder}",
        environment
    ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------

def _latex_ode(h, environment, show_orders):
    eps  = h.small_param
    t    = h.independent
    N    = len(h.entries) - 1
    dep  = str(h.entries[0].symbol.func).rsplit('_', 1)[0]
    lines = []

    ptype = h._problem_type.upper()
    lines.append(_comment(f"Regular perturbation — ODE ({ptype})"))

    if show_orders:
        lines.append(_comment("Particular solutions at each order"))
        order_lines = []
        for e in h.entries:
            sol_str = _eps_latex(e.particular_solution, eps)
            order_lines.append(
                f"  {dep}_{{{e.order}}}({t}) &= {sol_str}"
            )
        lines.append(_wrap(" \\\\\n".join(order_lines), environment))
        lines.append("\n")

    lines.append(_comment("Composite expansion"))
    comp = _eps_latex(h.composite, eps)
    remainder = _order_remainder(N + 1)
    lines.append(_wrap(
        f"  {dep}({t},\\varepsilon) &= {comp} + {remainder}",
        environment
    ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lindstedt
# ---------------------------------------------------------------------------

def _latex_lindstedt(h, environment, show_orders):
    eps  = h.small_param
    t    = h.independent
    tau  = h.tau
    N    = len(h.entries) - 1
    dep  = str(h.entries[0].symbol.func).rsplit('_', 1)[0]
    lines = []

    lines.append(_comment("Lindstedt-Poincare method"))
    lines.append(_comment(f"Strained time: tau = omega(eps) * {t}"))
    lines.append("\n")

    # Frequency expansion
    lines.append(_comment("Frequency expansion"))
    omega_str = _eps_latex(h.omega_expansion, eps)
    remainder = _order_remainder(N + 1)
    lines.append(_wrap(
        f"  \\omega(\\varepsilon) &= {omega_str} + {remainder}",
        environment
    ))
    lines.append("\n")

    # Order-by-order solutions
    if show_orders:
        lines.append(_comment("Solutions at each order (in strained time tau)"))
        order_lines = []
        for e in h.entries:
            sol_str = _eps_latex(e.particular_solution, eps)
            order_lines.append(
                f"  {dep}_{{{e.order}}}(\\tau) &= {sol_str}"
            )
            if e.order > 0 and e.omega_k_val is not None:
                omega_str_k = _eps_latex(e.omega_k_val, eps)
                order_lines.append(
                    f"  \\omega_{{{e.order}}} &= {omega_str_k}"
                )
        lines.append(_wrap(" \\\\\n".join(order_lines), environment))
        lines.append("\n")

    # Composite in tau
    lines.append(_comment("Composite solution (in strained time tau)"))
    comp_tau = _eps_latex(h.composite, eps)
    lines.append(_wrap(
        f"  {dep}(\\tau,\\varepsilon) &= {comp_tau} + {remainder}",
        environment
    ))
    lines.append("\n")

    # Composite in t
    lines.append(_comment(f"Composite solution (in physical time {t})"))
    comp_t = _eps_latex(h.composite_t, eps)
    lines.append(_wrap(
        f"  {dep}({t},\\varepsilon) &= {comp_t} + {remainder}",
        environment
    ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multiple scales
# ---------------------------------------------------------------------------

def _latex_multiple_scales(h, environment, show_orders):
    eps  = h.small_param
    t    = h.independent
    T0   = h.T0
    T1   = h.T1
    N    = len(h.entries) - 1
    dep  = str(h.entries[0].symbol.func).split('_')[0]
    lines = []

    lines.append(_comment("Method of multiple scales"))
    lines.append(_comment(f"T_0 = {t} (fast),  T_1 = eps*{t} (slow)"))
    lines.append("\n")

    # Amplitude equations
    lines.append(_comment("Amplitude functions (from solvability conditions)"))
    A_str = _eps_latex(h.amplitude_A, eps)
    B_str = _eps_latex(h.amplitude_B, eps)
    lines.append(_wrap(
        f"  A(T_1) &= {A_str} \\\\\n"
        f"  B(T_1) &= {B_str}",
        environment
    ))
    lines.append("\n")

    # Composite in t
    lines.append(_comment("Composite solution"))
    comp_t = _eps_latex(h.composite_t, eps)
    remainder = _order_remainder(N + 1)
    lines.append(_wrap(
        f"  {dep}({t},\\varepsilon) &= {comp_t} + {remainder}",
        environment
    ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Boundary layer
# ---------------------------------------------------------------------------

def _latex_boundary_layer(h, environment):
    eps  = h.small_param
    x    = h.independent
    xi   = h.layer_var
    lines = []

    lines.append(_comment("Matched asymptotic expansions"))
    lines.append(_comment(f"Layer location: {h.layer_location}"))
    lines.append(_comment(f"Stretched coordinate: xi = (x - layer) / eps"))
    lines.append("\n")

    # Outer
    lines.append(_comment("Outer solution"))
    outer_str = _eps_latex(h.outer, eps)
    lines.append(_wrap(
        f"  u^{{\\text{{out}}}}({x}) &= {outer_str}",
        environment
    ))
    lines.append("\n")

    # Inner
    lines.append(_comment("Inner solution"))
    inner_str = _eps_latex(h.inner, eps)
    lines.append(_wrap(
        f"  U(\\xi) &= {inner_str}",
        environment
    ))
    lines.append("\n")

    # Matching value
    match_str = _eps_latex(h.match, eps)
    lines.append(_comment(f"Matching value: {match_str}"))
    lines.append("\n")

    # Composite
    lines.append(_comment("Composite solution (Van Dyke rule)"))
    comp_str = _eps_latex(h.composite, eps)
    lines.append(_wrap(
        f"  u^{{\\text{{comp}}}}({x},\\varepsilon) &= {comp_str}",
        environment
    ))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ODE System
# ---------------------------------------------------------------------------

def _latex_ode_system(h, environment, show_orders):
    eps  = h.small_param
    N    = len(h.hierarchies[h.variables[0]]) - 1
    lines = []

    lines.append(_comment("Regular perturbation — coupled ODE system"))
    lines.append("\n")

    for var in h.variables:
        vh = h.hierarchies[var]

        if show_orders:
            lines.append(_comment(f"Order-by-order: {var}"))
            order_lines = []
            for e in vh.entries:
                sol_str = _eps_latex(e.particular_solution, eps)
                order_lines.append(
                    f"  {var}_{{{e.order}}} &= {sol_str}"
                )
            lines.append(_wrap(" \\\\\n".join(order_lines), environment))
            lines.append("\n")

        lines.append(_comment(f"Composite: {var}"))
        comp = _eps_latex(vh.composite, eps)
        remainder = _order_remainder(N + 1)
        lines.append(_wrap(
            f"  {var}(t,\\varepsilon) &= {comp} + {remainder}",
            environment
        ))
        lines.append("\n")

    return "\n".join(lines)
