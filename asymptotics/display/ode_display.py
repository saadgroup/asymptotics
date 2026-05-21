"""
asymptotics.display.ode_display
============================
Rich display for ODEHierarchy objects.
"""

from sympy import latex, Eq, Symbol, diff

def _latex_eps(expr, eps_sym):
    """Render as LaTeX, substituting eps symbol -> varepsilon for display."""
    from sympy import Symbol as _Sym
    if eps_sym is not None and str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, _Sym('varepsilon'))
    return latex(expr)


def show_ode(h, orders=None, mode: str = "auto") -> None:
    """Render an ODEHierarchy with LaTeX in Jupyter or plain text."""
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
    t       = h.independent
    eps     = h.small_param

    # Title
    display(HTML(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"</div>"
    ))

    # Ansatz
    # Extract base variable name from u_0 -> u
    _fname = str(entries[0].symbol.func)   # e.g. "u_0"
    dep    = _fname.rsplit('_', 1)[0] if '_' in _fname else _fname

    def _eps_prefix(k):
        if k == 0:   return ""
        elif k == 1: return r"\varepsilon\,"
        else:        return r"\varepsilon^{%d}\," % k

    terms = " + ".join(
        _eps_prefix(k) + f"{dep}_{{{k}}}(t)"
        for k in range(len(entries))
    )
    display(Math(
        r"\textbf{Ansatz:} \quad "
        + dep + r"(t,\varepsilon) = " + terms + r" + \cdots"
    ))

    # Order blocks
    for entry in entries:
        k         = entry.order
        eps_label = (
            r"\varepsilon^{" + str(k) + "}" if k > 1
            else (r"\varepsilon" if k == 1 else r"\varepsilon^{0}")
        )

        # ODE at this order
        ode_lhs = latex(entry.ode.lhs)
        display(Math(
            r"\mathcal{O}(" + eps_label + r") \text{ ODE:} \quad "
            + ode_lhs + " = 0"
        ))

        # General solution
        display(Math(
            r"\text{General:} \quad "
            + latex(entry.symbol) + " = " + latex(entry.general_solution)
        ))

        # Particular solution (constants applied)
        display(Math(
            r"\text{Particular:} \quad "
            + latex(entry.symbol) + " = " + latex(entry.particular_solution)
        ))

        # Secular term warning
        if entry.secular:
            display(HTML(
                f"<div style='margin-left:1em;padding:4px 10px;"
                f"background:#fff3cd;border-left:3px solid #ffc107;"
                f"font-size:0.85em;margin-top:4px;'>"
                f"⚠️ <strong>Secular term detected</strong> at order ε<sup>{k}</sup> "
                f"— solution grows unboundedly in t. "
                f"Consider Lindstedt–Poincaré or multiple scales.</div>"
            ))

    # Expansion
    max_order = max(e.order for e in h.entries)
    pieces = []
    for entry in sorted(h.entries, key=lambda e: e.order):
        k   = entry.order
        val = entry.particular_solution
        if val == 0:
            continue
        try:
            is_neg = False  # ODEs rarely simplify to a known-sign expression
        except Exception:
            is_neg = False

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
    display(Math(r"\boxed{" + dep + r"(t,\varepsilon) = " + rhs + " " + remainder + r"}"))


def _show_text(h, orders):
    entries = h.entries if orders is None else [h[k] for k in orders]
    width   = 60
    sup     = ['⁰','¹','²','³','⁴','⁵']

    print("=" * width)
    print(f"  {h._method}")
    print("=" * width)

    # Extract base variable name from u_0(t) -> u
    _fname = str(entries[0].symbol.func)
    dep    = _fname.rsplit("_", 1)[0] if "_" in _fname else _fname

    # Ansatz
    sub = ['₀','₁','₂','₃','₄','₅']
    terms = " + ".join(
        ("ε" + (sup[k] if k > 1 else "") + "·" if k > 0 else "") + f"{dep}{sub[k]}(t)"
        for k in range(len(entries))
    )
    print(f"\nAnsatz: {dep}(t,ε) = {terms} + …\n")

    for entry in entries:
        k = entry.order
        label = f"O(ε{sup[k] if k < len(sup) else '^'+str(k)})"
        print(f"  {label}")
        from sympy import pretty
        print(f"    ODE      : {pretty(entry.ode)}")
        print(f"    General  : {pretty(entry.symbol)} = {pretty(entry.general_solution)}")
        print(f"    Particular: {pretty(entry.symbol)} = {pretty(entry.particular_solution)}")
        if entry.secular:
            print(f"    ⚠️  SECULAR TERM detected — solution grows in t!")
        print()

    print("-" * width)
    print(f"  Expansion: {dep}(t,ε) = {h.expansion}")
    print("=" * width)
