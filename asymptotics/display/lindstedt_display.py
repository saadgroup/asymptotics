"""
asymptotics.display.lindstedt_display
===================================
Rich display for LindstedtHierarchy objects.
"""

from sympy import latex, Integer, Eq

def _latex_eps(expr, eps_sym):
    """Render as LaTeX, substituting eps symbol -> varepsilon for display."""
    from sympy import Symbol as _Sym
    if eps_sym is not None and str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, _Sym('varepsilon'))
    return latex(expr)


def show_lindstedt(h, orders=None, mode: str = "auto") -> None:
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
    entries = h.entries if orders is None else [h[k] for k in orders]
    eps     = h.small_param
    tau     = h.tau
    t       = h.independent

    # Extract base variable name
    fname = str(entries[0].symbol.func)
    dep   = fname.rsplit('_', 1)[0] if '_' in fname else fname

    # Title
    display(HTML(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"</div>"
    ))

    # Strained time and ansatz
    display(Math(
        r"\textbf{Strained time:} \quad \tau = \omega(\varepsilon)\,t"
    ))

    def _eps_prefix(k):
        if k == 0:   return ""
        elif k == 1: return r"\varepsilon\,"
        else:        return r"\varepsilon^{%d}\," % k

    terms = " + ".join(
        _eps_prefix(k) + f"{dep}_{{{k}}}(\\tau)"
        for k in range(len(entries))
    )
    display(Math(
        r"\textbf{Ansatz:} \quad "
        + dep + r"(\tau, \varepsilon) = " + terms + r" + \cdots"
    ))

    omega_terms = latex(h.omega_0)
    for k in range(1, len(entries)):
        ok_sym = h.entries[k].omega_k_sym
        ok_val = h.omega_values.get(ok_sym, Integer(0))
        if ok_val != 0:
            omega_terms += r" + " + _eps_prefix(k) + r"\," + latex(ok_val)
    omega_terms += r" + \cdots"
    display(Math(
        r"\textbf{Frequency:} \quad \omega(\varepsilon) = " + omega_terms
    ))

    # Order blocks
    for entry in entries:
        k         = entry.order
        eps_label = (
            r"\varepsilon^{" + str(k) + "}" if k > 1
            else (r"\varepsilon" if k == 1 else r"\varepsilon^{0}")
        )

        sup = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹']
        order_str = 'ε' + (sup[k] if k < len(sup) else str(k))
        display(HTML(
            f"<div style='margin-top:10px;font-weight:500;border-left:2px solid #7F77DD;"
            f"padding-left:8px;'>Order {order_str}</div>"
        ))

        # ODE
        ode_lhs = latex(entry.ode.lhs)
        display(Math(
            r"\mathcal{O}(" + eps_label + r") \text{ ODE:} \quad " + ode_lhs + " = 0"
        ))

        # Secularity condition
        if entry.secularity_condition is not None and k > 0:
            display(Math(
                r"\text{Secularity condition:} \quad "
                + latex(entry.secularity_condition)
                + r"\quad \Longrightarrow \quad "
                + latex(Eq(entry.omega_k_sym, entry.omega_k_val))
            ))

        # Particular solution
        display(Math(
            r"\text{Solution:} \quad "
            + latex(entry.symbol) + " = " + latex(entry.particular_solution)
        ))

    # Final results
    display(HTML("<div style='margin-top:12px;font-weight:600;'>Results:</div>"))

    # Frequency
    display(Math(
        r"\boxed{\omega(\varepsilon) = " + latex(h.omega_expansion)
        + r" + \mathcal{O}\!\left(\varepsilon^{%d}\right)}" % (len(entries))
    ))

    # Expansion in tau
    max_order = max(e.order for e in h.entries)
    pieces    = []
    for entry in sorted(h.entries, key=lambda e: e.order):
        k   = entry.order
        val = entry.particular_solution
        if val == 0:
            continue
        val_latex = latex(val)
        if k == 0:
            term = val_latex
        elif k == 1:
            term = r"\varepsilon \left(" + val_latex + r"\right)"
        else:
            term = r"\varepsilon^{%d} \left(" % k + val_latex + r"\right)"
        pieces.append(term)

    rhs       = " + ".join(pieces) if pieces else "0"
    remainder = r"+ \,\mathcal{O}\!\left(\varepsilon^{%d}\right)" % (max_order + 1)

    display(Math(
        r"\boxed{" + dep + r"(\tau,\varepsilon) = " + rhs + " " + remainder + r"}"
    ))

    display(HTML(
        f"<div style='font-size:0.85em;color:#666;margin-top:4px;'>"
        f"where τ = ω(ε)·t</div>"
    ))


def _show_text(h, orders):
    entries = h.entries if orders is None else [h[k] for k in orders]
    width   = 64
    sup     = ['⁰', '¹', '²', '³', '⁴', '⁵']

    fname = str(entries[0].symbol.func)
    dep   = fname.rsplit('_', 1)[0] if '_' in fname else fname

    print("=" * width)
    print(f"  {h._method}")
    print("=" * width)
    print(f"\n  Strained time: τ = ω(ε)·t")
    print(f"  ω₀ = {h.omega_0}")

    sub = ['₀', '₁', '₂', '₃', '₄', '₅']
    terms = " + ".join(
        ("ε" + (sup[k] if k > 1 else "") + "·" if k > 0 else "") + f"{dep}{sub[k]}(τ)"
        for k in range(len(entries))
    )
    print(f"  Ansatz: {dep}(τ,ε) = {terms} + …\n")

    for entry in entries:
        k = entry.order
        print(f"  O(ε{sup[k] if k < len(sup) else '^'+str(k)})")
        print(f"    ODE      : {entry.ode}")
        if entry.secularity_condition is not None and k > 0:
            print(f"    Secularity: {entry.secularity_condition}")
            print(f"    => {entry.omega_k_sym} = {entry.omega_k_val}")
        print(f"    Solution : {entry.symbol} = {entry.particular_solution}")
        print()

    print("-" * width)
    print(f"  ω(ε) = {h.omega_expansion} + O(ε{sup[len(entries)] if len(entries)<len(sup) else '...'})")
    print(f"  {dep}(τ,ε) = {h.expansion}")
    print(f"  where τ = ω(ε)·t")
    print("=" * width)
