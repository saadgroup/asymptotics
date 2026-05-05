"""
asymptotics.core.hierarchy
======================
The OrderHierarchy is the central bookkeeping object produced by an
expansion.  It stores every intermediate symbolic object so the user
can inspect, debug, and override any step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from sympy import Expr, Symbol, Eq, pretty, latex, O, Add


@dataclass
class OrderEntry:
    """All information about a single power of eps."""
    order       : int
    equation    : Eq          # the equation x_k must satisfy
    solution    : Expr        # the solved value of x_k
    symbol      : Symbol      # the symbol x_k used in the expansion
    note        : str = ""


    def to_latex(self, environment='align', show_orders=False, filename=None):
        """
        Export this expansion as LaTeX source.

        Parameters
        ----------
        environment : str
            LaTeX math environment: 'align' (default), 'equation', or 'gather'.
        show_orders : bool
            If True, include each order u_k separately. Default False.
        filename : str, optional
            If given, write to this file. Otherwise print to console.

        Returns
        -------
        str — the LaTeX source string

        Examples
        --------
        >>> print(sol.to_latex())
        >>> sol.to_latex(filename="result.tex")
        >>> sol.to_latex(environment='equation', show_orders=True)
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)

    def show(self):
        print(f"  O(eps^{self.order})")
        print(f"    equation : {pretty(self.equation)}")
        print(f"    solution : {self.symbol} = {pretty(self.solution)}")
        if self.note:
            print(f"    note     : {self.note}")


class OrderHierarchy:
    """
    Container for the full order-by-order perturbation hierarchy.

    Attributes
    ----------
    entries : list of OrderEntry
        One entry per order, 0-indexed.
    substituted_equation : Expr
        The original equation after substituting the ansatz.
    collected : dict
        Maps eps**k -> the coefficient expression (the equation at that order).
    composite : Expr
        The assembled composite expansion x = x0 + eps*x1 + ...
    """

    def __init__(self):
        self.entries               : List[OrderEntry] = []
        self.substituted_equation  : Optional[Expr]   = None
        self.collected             : Dict[int, Expr]  = {}   # order -> coeff expr
        self.composite             : Optional[Expr]   = None
        self.small_param           : Optional[Symbol] = None  # the eps symbol
        self._method               : str              = ""
        self._problem_repr         : str              = ""

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def __getitem__(self, order: int) -> OrderEntry:
        return self.entries[order]

    def __len__(self):
        return len(self.entries)

    @property
    def solutions(self) -> Dict[Symbol, Expr]:
        """Return {x_k: value} dict for all solved orders."""
        return {e.symbol: e.solution for e in self.entries}

    @property
    def equations(self) -> List[Eq]:
        return [e.equation for e in self.entries]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show(self, orders=None, mode: str = "auto") -> None:
        """
        Render the perturbation hierarchy.

        Parameters
        ----------
        orders : list of int, optional
            Which orders to display. Default: all.
        mode : str
            "auto"    — LaTeX in Jupyter, plain text elsewhere (default)
            "jupyter" — force LaTeX via IPython
            "text"    — force plain text
        """
        from asymptotics.display.jupyter import show as _show
        _show(self, orders=orders, mode=mode)

    def latex_composite(self) -> str:
        return latex(self.composite)

    def latex_equations(self) -> List[str]:
        return [latex(e.equation) for e in self.entries]

    def _show_body(self, orders=None) -> None:
        """Show order blocks and composite without title header.
        Used by SystemHierarchy to embed per-variable output."""
        from asymptotics.display.jupyter import _show_jupyter_body
        _show_jupyter_body(self, orders=orders)
