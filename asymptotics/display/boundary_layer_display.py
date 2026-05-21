"""
asymptotics.display.boundary_layer_display
========================================
Rich display for BoundaryLayerHierarchy — shows outer, inner, expansion.
"""

from sympy import latex, Symbol, Eq


def _lx(expr, eps_sym):
    """latex with eps -> varepsilon substitution."""
    from sympy import Symbol as _S
    if eps_sym is not None and str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, _S('varepsilon'))
    return latex(expr)


def show_boundary_layer(h, mode: str = "auto") -> None:
    try:
        from IPython.display import display, Math, HTML
        _jupyter = True
    except ImportError:
        _jupyter = False

    if mode == "text" or (mode == "auto" and not _jupyter):
        _show_text(h)
        return

    from IPython.display import display, Math, HTML
    _show_jupyter(h, display, Math, HTML)


def _show_jupyter(h, display, Math, HTML):
    eps = h.small_param
    x   = h.independent
    xi  = h.layer_var
    dep = str(h.independent)   # reuse as placeholder — actually get from problem
    # Extract dep name from outer expression
    dep = 'u'   # default

    # Title
    display(HTML(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"</div>"
    ))

    # Problem
    display(Math(
        r"\textbf{Problem:} \quad " + _lx(h.outer_ode.lhs, eps)
        + r" = " + _lx(h.outer_ode.rhs, eps)
        + r" \quad \text{(reduced outer ODE)}"
    ))

    display(HTML(
        f"<div style='font-size:0.9em;color:#555;margin:4px 0 12px 0;'>"
        f"Layer location: <b>{h.layer_location}</b> &nbsp;|&nbsp; "
        f"Stretched coordinate: ξ = {_lx(xi, eps)}"
        f"</div>"
    ))

    # ── Outer solution ──
    display(HTML(
        "<div style='margin-top:8px;font-weight:600;border-left:3px solid #1D9E75;"
        "padding-left:8px;'>Outer Solution</div>"
    ))
    display(HTML(
        f"<div style='font-size:0.85em;color:#666;margin:2px 0 4px 12px;'>"
        f"Reduced ODE (ε=0): {h.outer_bc}</div>"
    ))
    display(Math(
        r"\boxed{u^{\text{out}}(" + _lx(x, eps) + r") = "
        + _lx(h.outer, eps) + r"}"
    ))

    # ── Inner solution ──
    display(HTML(
        "<div style='margin-top:12px;font-weight:600;border-left:3px solid #378ADD;"
        "padding-left:8px;'>Inner Solution</div>"
    ))
    display(HTML(
        f"<div style='font-size:0.85em;color:#666;margin:2px 0 4px 12px;'>"
        f"Inner ODE in ξ: &nbsp; {h.inner_bc} &nbsp;|&nbsp; "
        f"Matching: U(∞) = {_lx(h.match, eps)}</div>"
    ))
    display(Math(
        r"U(\xi) = " + _lx(h.inner, eps)
    ))
    display(Math(
        r"u^{\text{in}}(" + _lx(x, eps) + r") = U\!\left("
        + r"\frac{" + _lx(h.layer_var, eps) + r"\text{-substituted}}{1}"
        + r"\right) = " + _lx(h.inner_xi, eps)
    ))

    # ── Matching value ──
    display(HTML(
        "<div style='margin-top:12px;font-weight:600;border-left:3px solid #F0A500;"
        "padding-left:8px;'>Matching Condition</div>"
    ))
    display(Math(
        r"\lim_{\xi\to\infty} U(\xi) = u^{\text{out}}(\text{layer}) = "
        + _lx(h.match, eps)
    ))

    # ── Expansion ──
    display(HTML(
        "<div style='margin-top:12px;font-weight:600;border-left:3px solid #7F77DD;"
        "padding-left:8px;'>Expansion Solution</div>"
    ))
    display(HTML(
        "<div style='font-size:0.85em;color:#666;margin:2px 0 4px 12px;'>"
        "u<sup>comp</sup> = u<sup>out</sup> + u<sup>in</sup> − u<sup>match</sup></div>"
    ))
    display(Math(
        r"\boxed{u^{\text{comp}}(" + _lx(x, eps) + r") = "
        + _lx(h.expansion, eps) + r"}"
    ))


def _show_text(h):
    eps = h.small_param
    width = 64

    print("=" * width)
    print(f"  {h._method}")
    print("=" * width)
    print(f"\n  Layer location : {h.layer_location}")
    print(f"  Stretched coord: ξ = (x - layer)/ε")

    print(f"\n── Outer solution ─────────────────────────────────")
    print(f"  ODE : {h.outer_ode}")
    print(f"  BC  : {h.outer_bc}")
    print(f"  u_out(x) = {h.outer}")

    print(f"\n── Inner solution ─────────────────────────────────")
    print(f"  ODE : {h.inner_ode}")
    print(f"  BC  : {h.inner_bc}")
    print(f"  U(ξ) = {h.inner}")
    print(f"  u_in(x) = {h.inner_xi}")

    print(f"\n── Matching ────────────────────────────────────────")
    print(f"  lim_{{ξ→∞}} U(ξ) = u_out(layer) = {h.match}")

    print(f"\n── Expansion ───────────────────────────────────────")
    print(f"  u_comp(x) = u_out + u_in - match")
    print(f"            = {h.expansion}")
    print("=" * width)
