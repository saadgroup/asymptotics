"""
asymptotics.display.ode_system_display
====================================
Rich display for ODESystemHierarchy.
"""

from sympy import latex, Symbol


def _lx(expr, eps_sym):
    from sympy import Symbol as _S
    if eps_sym is not None and str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, _S('varepsilon'))
    return latex(expr)


def show_ode_system(h, mode: str = "auto") -> None:
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
    t   = h.independent
    N   = len(h.hierarchies[h.variables[0]]) - 1

    # Title
    display(HTML(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"</div>"
    ))

    # Ansatz for each variable
    def _eps_prefix(k):
        if k == 0:   return ""
        elif k == 1: return r"\varepsilon\,"
        else:        return r"\varepsilon^{%d}\," % k

    for var in h.variables:
        terms = " + ".join(
            _eps_prefix(k) + f"{var}_{{{k}}}(t)"
            for k in range(N + 1)
        )
        display(Math(
            r"\textbf{Ansatz:} \quad "
            + var + r"(t,\varepsilon) = " + terms + r" + \cdots"
        ))

    # Order blocks
    for k in range(N + 1):
        eps_label = (
            r"\varepsilon^{" + str(k) + "}" if k > 1
            else (r"\varepsilon" if k == 1 else r"\varepsilon^{0}")
        )

        sup = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹']
        order_str = 'ε' + (sup[k] if k < len(sup) else str(k))
        display(HTML(
            f"<div style='margin-top:10px;font-weight:500;"
            f"border-left:2px solid #7F77DD;padding-left:8px;'>"
            f"Order {order_str}</div>"
        ))

        for var in h.variables:
            entry = h.hierarchies[var][k]
            display(Math(
                r"\mathcal{O}(" + eps_label + r") \text{ for } " + var + r": \quad "
                + _lx(entry.ode.lhs, eps) + " = 0"
            ))
            display(Math(
                r"\quad " + _lx(entry.symbol, eps)
                + " = " + _lx(entry.particular_solution, eps)
            ))

    # Expansions — build term-by-term low-to-high to match show_ode ordering
    display(HTML(
        "<div style='margin-top:12px;font-weight:600;'>Expansion expansions:</div>"
    ))
    for var in h.variables:
        pieces = []
        for k in range(N + 1):
            entry = h.hierarchies[var][k]
            val   = entry.particular_solution
            if val == 0:
                continue

            is_neg    = val.could_extract_minus_sign()
            abs_val   = -val if is_neg else val
            abs_latex = _lx(abs_val, eps)

            if k == 0:
                term = abs_latex
            elif k == 1:
                if abs_val.is_Add:
                    term = r"\varepsilon \left(" + abs_latex + r"\right)"
                else:
                    term = r"\varepsilon\," + abs_latex
            else:
                if abs_val.is_Add:
                    term = r"\varepsilon^{%d} \left(" % k + abs_latex + r"\right)"
                else:
                    term = r"\varepsilon^{%d}\," % k + abs_latex
            pieces.append(("-" if is_neg else "+", term))

        rhs_parts = []
        for i, (sign, term) in enumerate(pieces):
            rhs_parts.append(("-" + term if sign == "-" else term) if i == 0
                             else sign + " " + term)
        rhs       = " ".join(rhs_parts) if rhs_parts else "0"
        remainder = r"+ \,\mathcal{O}\!\left(\varepsilon^{%d}\right)" % (N + 1)
        display(Math(
            r"\boxed{" + var + r"(t,\varepsilon) = " + rhs + " " + remainder + r"}"
        ))


def _show_text(h):
    eps = h.small_param
    N   = len(h.hierarchies[h.variables[0]]) - 1
    sup = ['⁰', '¹', '²', '³', '⁴', '⁵']
    sub = ['₀', '₁', '₂', '₃', '₄', '₅']
    width = 64

    print("=" * width)
    print(f"  {h._method}")
    print("=" * width)

    for var in h.variables:
        terms = " + ".join(
            ("ε" + (sup[k] if k > 1 else "") + "·" if k > 0 else "")
            + f"{var}{sub[k]}(t)"
            for k in range(N + 1)
        )
        print(f"  Ansatz: {var}(t,ε) = {terms} + …")

    print()
    for k in range(N + 1):
        label = f"O(ε{sup[k] if k < len(sup) else '^'+str(k)})"
        print(f"  {label}")
        for var in h.variables:
            entry = h.hierarchies[var][k]
            print(f"    {var}{sub[k]} : {entry.particular_solution}")
        print()

    print("-" * width)
    print("  Expansions:")
    for var in h.variables:
        pieces = []
        for k in range(N + 1):
            entry = h.hierarchies[var][k]
            val   = entry.particular_solution
            if val == 0:
                continue
            val_str = str(val)
            if k == 0:
                term = val_str
            elif k == 1:
                term = f"ε·({val_str})"
            else:
                term = f"ε{sup[k] if k < len(sup) else '^'+str(k)}·({val_str})"
            pieces.append(term)
        rhs = " + ".join(pieces) if pieces else "0"
        print(f"    {var}(t,ε) = {rhs} + …")
    print("=" * width)
