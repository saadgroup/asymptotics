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
    expansion : Expr
        The assembled expansion expansion x = x0 + eps*x1 + ...
    """

    def __init__(self):
        self.entries               : List[OrderEntry] = []
        self.substituted_equation  : Optional[Expr]   = None
        self.collected             : Dict[int, Expr]  = {}   # order -> coeff expr
        self.expansion             : Optional[Expr]   = None
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

    def compare_numeric(self, eps, params=None, **kwargs):
        """
        Compare algebraic perturbation expansion against scipy root-finder.

        Plots the perturbation expansion vs the exact root as a function
        of eps over the range [0, eps_max].

        Parameters
        ----------
        eps : float
            Representative eps value (used to set default eps_max = 3*eps).
        eps_max : float, optional
            Maximum eps for the plot. Default: 3*eps.
        n_points : int
            Number of eps values to evaluate. Default 100.

        Returns
        -------
        dict with keys: 'eps', 'perturbation', 'numerical', 'fig'
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, **kwargs)

    def to_latex(self, environment='align', show_orders=False, filename=None):
        """
        Export this expansion as LaTeX source.

        Parameters
        ----------
        environment : str
            LaTeX math environment: 'align' (default), 'equation', or 'gather'.
        show_orders : bool
            If True, include each order solution separately. Default False.
        filename : str, optional
            If given, write to this file. Otherwise print to console.

        Returns
        -------
        str — the LaTeX source string

        Examples
        --------
        >>> sol.to_latex()
        >>> sol.to_latex(show_orders=True)
        >>> sol.to_latex(filename="result.tex")
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)

    def eval(self, eps, at=None, params=None):
        """
        Evaluate the perturbation expansion at given eps and optional point array.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter.
        at : array-like, optional
            Not needed for algebraic equations.

        Returns
        -------
        float if eps is scalar, ndarray if eps is a list

        Examples
        --------
        >>> x = sol.eval(eps=0.1)
        >>> x = sol.eval(eps=[0.1, 0.2, 0.3])
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

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

    def latex_expansion(self) -> str:
        return latex(self.expansion)

    def latex_equations(self) -> List[str]:
        return [latex(e.equation) for e in self.entries]

    def _show_body(self, orders=None) -> None:
        """Show order blocks and expansion without title header.
        Used by SystemHierarchy to embed per-variable output."""
        from asymptotics.display.jupyter import _show_jupyter_body
        _show_jupyter_body(self, orders=orders)
