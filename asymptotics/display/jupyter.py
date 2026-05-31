"""
asymptotics.display.jupyter
=======================
Rich display for OrderHierarchy objects in Jupyter notebooks.
Renders equations as LaTeX via IPython's display system.

Usage
-----
from asymptotics.display import show
show(hierarchy)               # full rendered output
show(hierarchy, orders=[0,1]) # only specific orders
"""

from __future__ import annotations
from typing import Optional, List

from sympy import latex, Eq, Symbol, Integer, pretty


def _latex_eps(expr, eps_sym):
    """Render as LaTeX, substituting eps symbol -> varepsilon for display."""
    from sympy import Symbol as _Sym
    if eps_sym is not None and str(eps_sym) not in ('epsilon', 'varepsilon'):
        expr = expr.subs(eps_sym, _Sym('varepsilon'))
    return latex(expr)

from asymptotics.core.hierarchy import OrderHierarchy, OrderEntry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _order_superscript(k: int) -> str:
    sups = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']
    return ''.join(sups[int(d)] for d in str(k))


def _latex_order_block(entry: OrderEntry, dep_sym: str = "x") -> str:
    """Return a LaTeX string for one order block."""
    k = entry.order
    eps_power = r"\varepsilon^{%d}" % k if k > 1 else (r"\varepsilon" if k == 1 else r"1")

    eq_latex  = _latex_eps(entry.equation, h.small_param)
    sol_latex = _latex_eps(Eq(entry.symbol, entry.solution), h.small_param)

    note_line = ""
    if entry.note:
        note_line = r"\\ \quad \scriptsize{\text{" + entry.note.replace("_", r"\_") + r"}}"

    return (
        r"\mathcal{O}(" + eps_power + r")\text{ equation:} & \quad "
        + eq_latex + r"\\" 
        + r"\mathcal{O}(" + eps_power + r")\text{ solution:} & \quad "
        + sol_latex
        + note_line
    )


def _build_full_latex(h: OrderHierarchy, orders: Optional[List[int]] = None) -> str:
    """Build the complete LaTeX block for a hierarchy."""
    entries = h.entries if orders is None else [h[k] for k in orders]

    # Header
    lines = []
    lines.append(r"\begin{array}{ll}")

    for i, entry in enumerate(entries):
        lines.append(_latex_order_block(entry))
        if i < len(entries) - 1:
            lines.append(r"\\[6pt]")

    lines.append(r"\end{array}")

    order_block = "\n".join(lines)

    # Expansion
    from sympy import Symbol as Sym
    dep = entries[0].symbol  # e.g. x_0 — extract base name
    base = str(dep)[:-2] if str(dep).endswith('_0') else str(dep)
    comp_lhs = Sym(base)

    expansion_latex = latex(Eq(comp_lhs, h.expansion))

    full = (
        r"\textbf{Perturbation Hierarchy}"
        + r"\quad \small{[" + h._method.replace("—", r"\text{---}") + r"]}"
        + r"\\ \small{f(" + base + r",\,\varepsilon) = 0}"
        + r"\\[10pt]"
        + r"\textbf{Order equations \& solutions:}"
        + r"\\[4pt]"
        + order_block
        + r"\\[10pt]"
        + r"\textbf{Expansion expansion:}"
        + r"\\[4pt]"
        + r"\boxed{" + expansion_latex + r"}"
    )

    return full


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show(
    h: OrderHierarchy,
    orders: Optional[List[int]] = None,
    mode: str = "auto",
) -> None:
    """
    Render an OrderHierarchy with beautiful math formatting.

    Parameters
    ----------
    h : OrderHierarchy
    orders : list of int, optional
        Which orders to display. Default: all.
    mode : str
        'jupyter' — render LaTeX via IPython display (default in notebooks)
        'text'    — plain pretty-print (always works)
        'auto'    — try Jupyter, fall back to text
    """
    if mode == "text":
        _show_text(h, orders)
        return

    try:
        from IPython.display import display, Math, HTML
        _show_jupyter(h, orders, display, Math, HTML)
    except ImportError:
        if mode == "jupyter":
            raise RuntimeError("IPython not available. Use mode='text'.")
        _show_text(h, orders)


def _show_jupyter(h, orders, display, Math, HTML):
    """Render using IPython rich display."""
    entries = h.entries if orders is None else [h[k] for k in orders]

    # Title + problem statement as HTML
    method_str = h._method
    display(HTML(
        f"<div style='margin-bottom:4px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{method_str}</span>"
        f"</div>"
    ))

    # Ansatz
    base = _get_base_name(entries[0].symbol)
    gauge_seq = getattr(h, '_gauge', None)

    def _gauge_prefix_latex(k):
        if gauge_seq is not None:
            from asymptotics.gauge import gauge_term_latex
            g = gauge_term_latex(gauge_seq[k], h.small_param)
            return "" if g == "1" else g + r"\,"
        if k == 0:   return ""
        elif k == 1: return r"\varepsilon\,"
        else:        return r"\varepsilon^{%d}\," % k

    eps_terms = " + ".join(
        _gauge_prefix_latex(k) + f"{base}_{{{k}}}"
        for k in range(len(entries))
    )
    display(Math(
        r"\textbf{Ansatz:}\quad "
        + base + r"(\varepsilon) = "
        + eps_terms
        + r" + \cdots"
    ))

    # Each order
    for entry in entries:
        k = entry.order
        if gauge_seq is not None:
            from asymptotics.gauge import gauge_term_latex
            eps_label = gauge_term_latex(gauge_seq[k], h.small_param)
            if eps_label == "1":
                eps_label = r"\varepsilon^{0}"
        else:
            eps_label = (
                r"\varepsilon^{" + str(k) + "}"  if k > 1
                else (r"\varepsilon"              if k == 1
                else  r"\varepsilon^{0}")
            )
        eq_latex  = _latex_eps(entry.equation, h.small_param)
        sol_latex = _latex_eps(Eq(entry.symbol, entry.solution), h.small_param)

        block = (
            r"\mathcal{O}(" + eps_label + r") : \quad "
            + eq_latex
            + r" \qquad \Longrightarrow \qquad "
            + sol_latex
        )
        display(Math(block))

        if entry.note:
            display(HTML(
                f"<div style='margin-left:2em;margin-top:-6px;"
                f"font-size:0.82em;color:#666;'>"
                f"ℹ️ {entry.note}</div>"
            ))

    # Expansion in a box — terms ordered low to high, with O() remainder
    comp_lhs = Symbol(base)
    max_order = max(e.order for e in h.entries)
    next_order = max_order + 1

    # Build term-by-term in ascending order of eps
    # Handle signs explicitly so negatives render as "- coeff" not "+ -coeff"
    from sympy import Abs as _Abs, Integer as _Int, Rational as _Rat, Number as _Num
    pieces = []   # list of (sign_str, term_latex)

    for entry in sorted(h.entries, key=lambda e: e.order):
        k   = entry.order
        val = entry.solution
        if val == 0:
            continue

        # Determine sign and absolute value
        is_neg = val.could_extract_minus_sign()

        abs_val    = -val if is_neg else val
        abs_latex  = _latex_eps(abs_val, h.small_param)
        if abs_val.is_Add:
            abs_latex = r"\left(" + abs_latex + r"\right)"

        if k == 0:
            term = abs_latex
        elif k == 1:
            term = r"\varepsilon\," + abs_latex
        else:
            term = r"\varepsilon^{%d}\," % k + abs_latex

        pieces.append(("-" if is_neg else "+", term))

    # Assemble: first term has no leading +
    rhs_parts = []
    for i, (sign, term) in enumerate(pieces):
        if i == 0:
            rhs_parts.append(("-" + term) if sign == "-" else term)
        else:
            rhs_parts.append(sign + " " + term)

    rhs = " ".join(rhs_parts) if rhs_parts else "0"
    remainder = r"+ \,\mathcal{O}\!\left(\varepsilon^{%d}\right)" % next_order

    display(Math(
        r"\boxed{" + _latex_eps(comp_lhs, h.small_param) + " = " + rhs + " " + remainder + r"}"
    ))


def _show_text(h, orders):
    """Plain-text fallback using sympy.pretty."""
    entries = h.entries if orders is None else [h[k] for k in orders]
    width = 56
    print("=" * width)
    print(f"  {h._method}")
    print("=" * width)

    base = _get_base_name(entries[0].symbol)
    gauge_seq = getattr(h, '_gauge', None)

    def _g_unicode(k):
        if gauge_seq is None:
            prefixes = ['', 'ε·', 'ε²·', 'ε³·', 'ε⁴·', 'ε⁵·']
            return prefixes[k] if k < len(prefixes) else f'ε^{k}·'
        from asymptotics.gauge import gauge_term_unicode
        g = gauge_term_unicode(gauge_seq[k], h.small_param)
        return '' if g == '1' else g + '·'

    def _g_label(k):
        if gauge_seq is None:
            sup = ['⁰','¹','²','³','⁴','⁵']
            return f"O(ε{sup[k] if k < len(sup) else '^'+str(k)})"
        from asymptotics.gauge import gauge_term_unicode
        g = gauge_term_unicode(gauge_seq[k], h.small_param)
        return f'O({g})'

    ansatz_terms = ' + '.join(_g_unicode(k) + f'{base}_{k}' for k in range(len(entries)))
    print(f'\nAnsatz: {base}(ε) = {ansatz_terms} + …\n')

    for entry in entries:
        k = entry.order
        label = _g_label(k)
        print(f"  {label}  equation : {pretty(entry.equation)}")
        print(f"  {label}  solution : {entry.symbol} = {pretty(entry.solution)}")
        if entry.note:
            print(f"           note     : {entry.note}")
        print()

    print("-" * width)
    comp_sym = Symbol(base)
    print(f"  Expansion:  {comp_sym} = {pretty(h.expansion)}")
    print("=" * width)


def _get_base_name(sym: Symbol) -> str:
    """Extract base variable name from x_0 -> x."""
    s = str(sym)
    if '_' in s:
        return s.rsplit('_', 1)[0]
    return s


def _show_jupyter_body(h, orders=None):
    """
    Show just the ansatz, order blocks, and expansion for a hierarchy —
    without the title header. Used by SystemHierarchy.
    """
    try:
        from IPython.display import display, Math, HTML
    except ImportError:
        _show_text(h, orders)
        return

    entries = h.entries if orders is None else [h[k] for k in orders]
    base    = _get_base_name(entries[0].symbol)

    def _eps_prefix(k):
        if k == 0:   return ""
        elif k == 1: return r"\varepsilon\,"
        else:        return r"\varepsilon^{%d}\," % k

    # Ansatz
    eps_terms = " + ".join(
        _eps_prefix(k) + f"{base}_{{{k}}}"
        for k in range(len(entries))
    )
    display(Math(
        r"\textbf{Ansatz:}\quad "
        + base + r"(\varepsilon) = "
        + eps_terms + r" + \cdots"
    ))

    # Order blocks
    for entry in entries:
        k         = entry.order
        eps_label = (
            r"\varepsilon^{" + str(k) + "}" if k > 1
            else (r"\varepsilon" if k == 1 else r"\varepsilon^{0}")
        )
        block = (
            r"\mathcal{O}(" + eps_label + r") : \quad "
            + _latex_eps(entry.equation, h.small_param)
            + r" \qquad \Longrightarrow \qquad "
            + _latex_eps(Eq(entry.symbol, entry.solution), h.small_param)
        )
        display(Math(block))
        if entry.note:
            display(HTML(
                f"<div style='margin-left:2em;margin-top:-6px;"
                f"font-size:0.82em;color:#666;'>ℹ️ {entry.note}</div>"
            ))

    # Expansion
    comp_lhs  = Symbol(base)
    max_order = max(e.order for e in h.entries)
    pieces    = []
    for entry in sorted(h.entries, key=lambda e: e.order):
        k   = entry.order
        val = entry.solution
        if val == 0:
            continue
        is_neg = val.could_extract_minus_sign()
        abs_val   = -val if is_neg else val
        abs_latex = _latex_eps(abs_val, h.small_param)
        if abs_val.is_Add:
            abs_latex = r"\left(" + abs_latex + r"\right)"
        if k == 0:
            term = abs_latex
        elif k == 1:
            term = r"\varepsilon\," + abs_latex
        else:
            term = r"\varepsilon^{%d}\," % k + abs_latex
        pieces.append(("-" if is_neg else "+", term))

    rhs_parts = []
    for i, (sign, term) in enumerate(pieces):
        rhs_parts.append(("-" + term if sign == "-" else term) if i == 0
                         else sign + " " + term)

    rhs       = " ".join(rhs_parts) if rhs_parts else "0"
    remainder = r"+ \,\mathcal{O}\!\left(\varepsilon^{%d}\right)" % (max_order + 1)
    display(Math(r"\boxed{" + _latex_eps(comp_lhs, h.small_param) + " = " + rhs + " " + remainder + r"}"))
