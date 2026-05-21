"""
asymptotics.core.system_hierarchy
==============================
Container for the perturbation hierarchy of a coupled algebraic system.

show() renders the full coupled system at each order:

  O(ε¹):  2x₁ + y₀ = 0
           x₀ + 2y₁ = 0
           ⟹  x₁ = -1/2,  y₁ = -1/2
"""

from __future__ import annotations
from typing import Dict, List, Optional, Iterator, Tuple
from sympy import Symbol, Expr, latex, Eq, pretty


class SystemHierarchy:
    """
    Perturbation hierarchy for a system of coupled algebraic equations.

    Stores one OrderHierarchy per dependent variable (for sol["x"] access)
    plus the full coupled system at each order (for show()).

    Attributes
    ----------
    variables : list of str
    hierarchies : dict  — variable name -> OrderHierarchy
    coupled_orders : dict — order k -> {equations, unknowns, solutions}
    small_param : Symbol
    """

    def __init__(self):
        self.variables      : List[str]               = []
        self.hierarchies    : Dict                    = {}
        self.coupled_orders : Dict                    = {}
        self.small_param    : Optional[Symbol]        = None
        self._method        : str                     = ""

    # ------------------------------------------------------------------
    # Dict-like access
    # ------------------------------------------------------------------

    def __getitem__(self, var: str):
        if var not in self.hierarchies:
            raise KeyError(
                f"Variable '{var}' not in system. "
                f"Available: {list(self.hierarchies.keys())}"
            )
        return self.hierarchies[var]

    def __iter__(self) -> Iterator[str]:
        return iter(self.variables)

    def items(self) -> Iterator[Tuple]:
        return ((v, self.hierarchies[v]) for v in self.variables)

    def __len__(self):
        return len(self.variables)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def eval(self, eps, **kwargs):
        """
        Evaluate the algebraic system expansion at given eps values.

        Parameters
        ----------
        eps : float or list of float

        Returns
        -------
        dict {var_name: float or ndarray}

        Examples
        --------
        >>> r = sol.eval(eps=0.1)
        >>> r['x'], r['y']
        >>> r = sol.eval(eps=[0.1, 0.2, 0.3])
        >>> r['x']  # ndarray
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, **kwargs)

    def to_latex(self, environment='align', show_orders=False, filename=None):
        """
        Export this expansion as LaTeX source.

        Parameters
        ----------
        environment : str
            'align' (default), 'equation', or 'gather'.
        show_orders : bool
            Include each order separately. Default False.
        filename : str, optional
            Save to file if given, otherwise print to console.

        Returns
        -------
        str — the LaTeX source string
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)

    def compare_numeric(self, eps, params=None, **kwargs):
        """
        Compare algebraic system expansion against scipy root-finder.

        Parameters
        ----------
        eps : float or list of float
            Plots x(eps) vs exact roots over the eps range.

        Returns
        -------
        dict with 'eps', 'perturbation', 'numerical', 'fig'
        """
        from asymptotics.numerics import compare_numeric
        return compare_numeric(self, eps, params=params, **kwargs)

    def show(self, orders=None, mode: str = "auto") -> None:
        """
        Render the full coupled hierarchy — coupled equations at each order.

        At each order shows the full system:
            O(ε¹):  2x₁ + y₀ = 0
                    x₀ + 2y₁ = 0
                    ⟹  x₁ = -1/2,  y₁ = -1/2

        Parameters
        ----------
        orders : list of int, optional
        mode : str — "auto", "jupyter", or "text"
        """
        try:
            from IPython.display import display, HTML, Math
            _jupyter = True
        except ImportError:
            _jupyter = False

        if mode == "text" or (mode == "auto" and not _jupyter):
            self._show_text(orders)
            return

        from IPython.display import display, HTML, Math

        # Title
        display(HTML(
            f"<div style='margin-bottom:8px'>"
            f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
            f"&nbsp;&nbsp;"
            f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
            f"font-size:0.85em;color:#555;'>{self._method}</span>"
            f"</div>"
        ))

        # Ansatz for each variable
        n_orders = len(next(iter(self.hierarchies.values())).entries)

        def _eps_prefix(k):
            if k == 0:   return ""
            elif k == 1: return r"\varepsilon\,"
            else:        return r"\varepsilon^{%d}\," % k

        ansatz_parts = []
        for var in self.variables:
            terms = " + ".join(
                _eps_prefix(k) + f"{var}_{{{k}}}"
                for k in range(n_orders)
            )
            ansatz_parts.append(f"{var}(\\varepsilon) = {terms} + \\cdots")

        display(Math(r"\textbf{Ansatz:} \quad " + r",\qquad ".join(ansatz_parts)))

        # Order blocks — show FULL coupled system at each order
        order_keys = sorted(self.coupled_orders.keys())
        if orders is not None:
            order_keys = [k for k in order_keys if k in orders]

        for k in order_keys:
            data = self.coupled_orders[k]
            eqs  = data["equations"]
            sols = data["solutions"]

            eps_label = (
                r"\varepsilon^{" + str(k) + "}" if k > 1
                else (r"\varepsilon" if k == 1 else r"\varepsilon^{0}")
            )

            # Build the system of equations as an array
            eq_lines = r" \\ ".join(latex(eq) for eq in eqs)
            sys_latex = (
                r"\left\{\begin{array}{l}" + eq_lines + r"\end{array}\right."
            )

            # Build the solution as x_k = ..., y_k = ...
            sol_parts = r",\quad ".join(
                latex(Eq(data["unknowns"][i], sols[self.variables[i]]))
                for i in range(len(self.variables))
            )

            display(Math(
                r"\mathcal{O}(" + eps_label + r") : \quad "
                + sys_latex
                + r" \qquad \Longrightarrow \qquad "
                + sol_parts
            ))

        # Expansion for each variable
        display(HTML("<div style='margin-top:8px;font-weight:600;'>Expansion expansions:</div>"))
        for var in self.variables:
            h   = self.hierarchies[var]
            lhs = latex(Symbol(var))
            pieces = []
            for entry in sorted(h.entries, key=lambda e: e.order):
                kk  = entry.order
                val = entry.solution
                if val == 0:
                    continue
                try:
                    is_neg = float(val.evalf()) < 0
                except Exception:
                    is_neg = False
                abs_val   = -val if is_neg else val
                abs_latex = latex(abs_val)
                if abs_val.is_Add:
                    abs_latex = r"\left(" + abs_latex + r"\right)"
                if kk == 0:
                    term = abs_latex
                elif kk == 1:
                    term = r"\varepsilon\," + abs_latex
                else:
                    term = r"\varepsilon^{%d}\," % kk + abs_latex
                pieces.append(("-" if is_neg else "+", term))

            rhs_parts = []
            for i, (sign, term) in enumerate(pieces):
                rhs_parts.append(("-" + term if sign == "-" else term) if i == 0
                                 else sign + " " + term)
            rhs       = " ".join(rhs_parts) if rhs_parts else "0"
            max_order = max(e.order for e in h.entries)
            remainder = r"+ \,\mathcal{O}\!\left(\varepsilon^{%d}\right)" % (max_order + 1)
            display(Math(r"\boxed{" + lhs + " = " + rhs + " " + remainder + r"}"))

    def _show_text(self, orders=None):
        width = 60
        print("=" * width)
        print(f"  {self._method}")
        print("=" * width)

        # Ansatz
        n_orders = len(next(iter(self.hierarchies.values())).entries)
        sup = ['⁰','¹','²','³','⁴','⁵']
        sub = ['₀','₁','₂','₃','₄','₅']
        print("\nAnsatz:")
        for var in self.variables:
            terms = " + ".join(
                ("ε" + (sup[k] if k > 1 else "") + "·" if k > 0 else "") + f"{var}{sub[k]}"
                for k in range(n_orders)
            )
            print(f"  {var}(ε) = {terms} + …")

        # Order blocks
        order_keys = sorted(self.coupled_orders.keys())
        if orders is not None:
            order_keys = [k for k in order_keys if k in orders]

        print()
        for k in order_keys:
            data = self.coupled_orders[k]
            eqs  = data["equations"]
            sols = data["solutions"]
            label = f"O(ε{sup[k] if k < len(sup) else '^'+str(k)})"
            print(f"  {label}  system:")
            for eq in eqs:
                print(f"           {pretty(eq)}")
            sol_str = ",  ".join(
                f"{data['unknowns'][i]} = {sols[self.variables[i]]}"
                for i in range(len(self.variables))
            )
            print(f"           ⟹  {sol_str}")
            print()

        # Expansions
        print("-" * width)
        print("  Expansion expansions:")
        for var in self.variables:
            h = self.hierarchies[var]
            print(f"    {var}(ε) = {pretty(h.expansion)}")
        print("=" * width)
