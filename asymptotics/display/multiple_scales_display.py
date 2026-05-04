"""
asymptotics.display.multiple_scales_display
========================================
Rich display for MultScalesHierarchy objects.
"""

from sympy import latex, Eq, Symbol, Function, diff

def _latex(expr, eps_sym=None):
    """Render expr as LaTeX, substituting eps symbol -> varepsilon for display."""
    if eps_sym is not None and str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, Symbol('varepsilon'))
    return latex(expr)


def show_multiple_scales(h, orders=None, mode: str = "auto") -> None:
    try:
        from IPython.display import display, Math, HTML
        _jupyter = True
    except ImportError:
        _jupyter = False

    if mode == "text" or (mode == "auto" and not _jupyter):
        _show_text(h, orders)
        return

    from IPython.display import display, Math, HTML
    _show_jupyter(h, orders, display, Math, HTML)


def _show_jupyter(h, orders, display, Math, HTML):
    entries  = h.entries if orders is None else [h[k] for k in orders]
    T0, T1   = h.T0, h.T1
    eps      = h.small_param
    omega0   = h.omega_0

    fname = str(entries[0].symbol.func)
    dep   = fname.split('_')[0]

    # Title
    display(HTML(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"</div>"
    ))

    # Timescales
    display(Math(
        r"\textbf{Timescales:} \quad "
        r"T_0 = t \quad \text{(fast)}, \qquad "
        r"T_1 = \varepsilon t \quad \text{(slow)}"
    ))

    # Ansatz
    def _eps_prefix(k):
        if k == 0:   return ""
        elif k == 1: return r"\varepsilon\,"
        else:        return r"\varepsilon^{%d}\," % k

    terms = " + ".join(
        _eps_prefix(k) + f"{dep}_{{{k}}}(T_0, T_1)"
        for k in range(len(entries))
    )
    display(Math(
        r"\textbf{Ansatz:} \quad "
        + dep + r"(t,\varepsilon) = " + terms + r" + \cdots"
    ))

    # Derivative expansion
    display(Math(
        r"\frac{d}{dt} = \partial_{T_0} + \varepsilon\,\partial_{T_1}, \qquad"
        r"\frac{d^2}{dt^2} = \partial_{T_0}^2 + "
        r"2\varepsilon\,\partial_{T_0 T_1} + \mathcal{O}(\varepsilon^2)"
    ))

    # O(1) solution
    display(HTML(
        "<div style='margin-top:10px;font-weight:500;border-left:2px solid #7F77DD;"
        "padding-left:8px;'>Order ε⁰</div>"
    ))
    display(Math(
        r"\mathcal{O}(1) \text{ solution:} \quad "
        + f"{dep}_0 = A(T_1)\\cos({latex(omega0)} T_0)"
        + f" + B(T_1)\\sin({latex(omega0)} T_0)"
    ))

    # Higher orders
    for entry in entries[1:]:
        k = entry.order
        eps_label = r"\varepsilon^{%d}" % k if k > 1 else r"\varepsilon"

        display(HTML(
            f"<div style='margin-top:10px;font-weight:500;border-left:2px solid #7F77DD;"
            f"padding-left:8px;'>Order ε{'^'+str(k) if k>1 else ''}</div>"
        ))

        # PDE
        display(Math(
            r"\mathcal{O}(" + eps_label + r") \text{ PDE:} \quad "
            + _latex(entry.pde.lhs, h.small_param) + " = " + _latex(entry.pde.rhs, h.small_param)
        ))

        # Secular terms
        if entry.secular_cos is not None or entry.secular_sin is not None:
            sec_str = r"\text{Secular terms:} \quad"
            if entry.secular_cos is not None and entry.secular_cos != 0:
                sec_str += (r"\left[" + _latex(entry.secular_cos, h.small_param) +
                           r"\right]\cos(" + latex(omega0) + r"T_0) = 0")
            if entry.secular_sin is not None and entry.secular_sin != 0:
                if entry.secular_cos is not None and entry.secular_cos != 0:
                    sec_str += r",\quad"
                sec_str += (r"\left[" + _latex(entry.secular_sin, h.small_param) +
                           r"\right]\sin(" + latex(omega0) + r"T_0) = 0")
            display(Math(sec_str))

        # Solvability conditions
        if entry.solvability_A is not None:
            display(Math(
                r"\text{Solvability:} \quad "
                + _latex(entry.solvability_A, h.small_param)
                + (r", \qquad " + _latex(entry.solvability_B, h.small_param)
                   if entry.solvability_B is not None else "")
            ))

        # Particular solution for uk
        if entry.particular_solution != 0:
            display(Math(
                r"\text{Solution:} \quad "
                + _latex(entry.symbol, h.small_param) + " = "
                + _latex(entry.particular_solution, h.small_param)
            ))

    # Amplitude functions
    display(HTML(
        "<div style='margin-top:12px;font-weight:600;'>Amplitude equations (solved):</div>"
    ))
    display(Math(
        r"A(T_1) = " + _latex(h.amplitude_A, h.small_param)
        + r", \qquad B(T_1) = " + _latex(h.amplitude_B, h.small_param)
    ))

    # Composite in t
    display(HTML(
        "<div style='margin-top:8px;font-weight:600;'>Composite expansion:</div>"
    ))
    dep_sym = Symbol(dep)
    max_order = max(e.order for e in h.entries)
    remainder = r"+ \,\mathcal{O}\!\left(\varepsilon^{%d}\right)" % (max_order + 1)
    display(Math(
        r"\boxed{" + dep + r"(t,\varepsilon) = "
        + _latex(h.composite_t, h.small_param) + " " + remainder + r"}"
    ))


def _show_text(h, orders):
    entries = h.entries if orders is None else [h[k] for k in orders]
    width   = 64
    sup     = ['⁰','¹','²','³','⁴','⁵']

    fname = str(entries[0].symbol.func)
    dep   = fname.split('_')[0]

    print("=" * width)
    print(f"  {h._method}")
    print("=" * width)
    print(f"\n  Timescales: T₀ = t (fast),  T₁ = ε·t (slow)")
    print(f"  ω₀ = {h.omega_0}")
    print(f"\n  Ansatz: {dep}(t,ε) = {dep}₀(T₀,T₁) + ε·{dep}₁(T₀,T₁) + …")
    print(f"\n  O(1) solution:")
    print(f"    {dep}₀ = A(T₁)·cos({h.omega_0}·T₀) + B(T₁)·sin({h.omega_0}·T₀)")

    for entry in entries[1:]:
        k = entry.order
        print(f"\n  O(ε{sup[k] if k < len(sup) else '^'+str(k)})")
        if entry.secular_cos is not None:
            print(f"    Secular cos: [{entry.secular_cos}] = 0")
        if entry.secular_sin is not None:
            print(f"    Secular sin: [{entry.secular_sin}] = 0")
        if entry.solvability_A is not None:
            print(f"    Solvability: {entry.solvability_A}")
        if entry.solvability_B is not None:
            print(f"               {entry.solvability_B}")
        if entry.particular_solution != 0:
            print(f"    {dep}_{k} = {entry.particular_solution}")

    print(f"\n  Amplitude (solved):")
    print(f"    A(T₁) = {h.amplitude_A}")
    print(f"    B(T₁) = {h.amplitude_B}")
    print(f"\n  Composite:")
    print(f"    {dep}(t,ε) = {h.composite_t}")
    print("=" * width)
