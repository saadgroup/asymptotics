"""
asymptotics.core.hierarchy
==========================
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
    r"""
    All information about a single power of the small parameter.

    An ``OrderEntry`` is the per-order record used by the generic
    :class:`OrderHierarchy` (produced, for example, by the regular expansion of
    an algebraic equation).  At order ``k`` it stores the equation the unknown
    :math:`x_k` must satisfy and the value that solves it.

    In the algebraic case the expansion is

    .. math::

        x(\varepsilon) = \sum_{k=0}^{N} x_k\, \varepsilon^{k}
                       = x_0 + \varepsilon\, x_1 + \varepsilon^{2}\, x_2
                       + \cdots ,

    and collecting the coefficient of :math:`\varepsilon^{k}` gives a
    (typically linear) equation for :math:`x_k` in terms of the already-known
    lower orders.

    Attributes
    ----------
    order : int
        The perturbation order ``k`` this entry describes.
    equation : sympy.Eq
        The equation :math:`x_k` must satisfy, written as ``Eq(<expr>, 0)``.
    solution : sympy.Expr
        The solved value of :math:`x_k`.
    symbol : sympy.Symbol
        The symbol :math:`x_k` used in the assembled expansion.
    note : str, optional
        Free-form annotation (e.g. ``"complex roots omitted (2 total)"`` at the
        leading order of a polynomial).  Empty by default.

    Examples
    --------
    >>> from asymptotics import AlgebraicEquation
    >>> eq = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
    ...                        small_param="eps")            # doctest: +SKIP
    >>> h = eq.expand_regular(order=3)                       # doctest: +SKIP
    >>> h[1].order                                           # doctest: +SKIP
    1
    >>> h[1].equation                                        # doctest: +SKIP
    Eq(3*x_1 + 1, 0)
    >>> h[1].solution                                        # doctest: +SKIP
    -1/3
    """
    order       : int
    equation    : Eq          # the equation x_k must satisfy
    solution    : Expr        # the solved value of x_k
    symbol      : Symbol      # the symbol x_k used in the expansion
    note        : str = ""



    def show(self):
        r"""
        Print this single order as a plain-text block.

        Writes the order label :math:`O(\varepsilon^{k})`, the equation
        :math:`x_k` satisfies, and its solution (pretty-printed), plus the
        annotation :attr:`note` if one is set.

        Returns
        -------
        None
            Output is printed as a side effect.

        Examples
        --------
        >>> from asymptotics import AlgebraicEquation
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> h[1].show()                                      # doctest: +SKIP
          O(eps^1)
            equation : 3⋅x₁ + 1 = 0
            solution : x_1 = -1/3
        """
        print(f"  O(eps^{self.order})")
        print(f"    equation : {pretty(self.equation)}")
        print(f"    solution : {self.symbol} = {pretty(self.solution)}")
        if self.note:
            print(f"    note     : {self.note}")


class OrderHierarchy:
    r"""
    Container for a full order-by-order perturbation hierarchy.

    ``OrderHierarchy`` is the generic bookkeeping object produced by an
    expansion of an algebraic perturbation problem (e.g.
    :meth:`asymptotics.AlgebraicEquation.expand_regular`).  It stores every
    intermediate symbolic object — the substituted equation, the collected
    per-order coefficients, and one :class:`OrderEntry` per order — so the user
    can inspect, debug, and override any step.  It also exposes the four-method
    result API shared by every hierarchy in :mod:`asymptotics`
    (:meth:`show`, :meth:`eval`, :meth:`compare_numeric`, :meth:`to_latex`).

    The expansion represented is a power series in the small parameter,

    .. math::

        x(\varepsilon) = \sum_{k=0}^{N} x_k\, \varepsilon^{k}
                       = x_0 + \varepsilon\, x_1 + \varepsilon^{2}\, x_2
                       + \cdots ,

    with each coefficient :math:`x_k` determined by the equation stored in the
    corresponding :class:`OrderEntry`.

    Indexing (``h[k]``) returns the :class:`OrderEntry` for order ``k`` and
    ``len(h)`` is the number of orders (``order + 1``).

    Attributes
    ----------
    entries : list of OrderEntry
        One entry per order, 0-indexed.
    substituted_equation : sympy.Expr
        The original equation after substituting the ansatz.
    collected : dict
        Maps order ``k`` -> the coefficient expression (the equation at that
        order).
    expansion : sympy.Expr
        The assembled expansion :math:`x = x_0 + \varepsilon x_1 + \cdots`.
    small_param : sympy.Symbol
        The small parameter :math:`\varepsilon`.

    See Also
    --------
    OrderEntry : Per-order record accessed via ``h[k]``.

    Examples
    --------
    >>> from asymptotics import AlgebraicEquation
    >>> eq = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
    ...                        small_param="eps")            # doctest: +SKIP
    >>> h = eq.expand_regular(order=3)                       # doctest: +SKIP
    >>> len(h)                                               # doctest: +SKIP
    4
    >>> h.expansion                                          # doctest: +SKIP
    eps**3/81 - eps/3 + 1
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
        """
        Return the :class:`OrderEntry` for a given order.

        Parameters
        ----------
        order : int
            Perturbation order ``k``, ``0 <= k <= N``.  Standard Python indexing
            semantics apply (negative indices count from the end).

        Returns
        -------
        OrderEntry
            The per-order record holding the order-``k`` equation and solution.

        Examples
        --------
        >>> h[0].solution        # leading-order value      # doctest: +SKIP
        >>> h[1].equation        # order-1 equation         # doctest: +SKIP
        """
        return self.entries[order]

    def __len__(self):
        """
        Return the number of orders stored (``order + 1``).

        Returns
        -------
        int
            One more than the requested expansion order, since order 0 is
            included.

        Examples
        --------
        >>> len(eq.expand_regular(order=3))      # doctest: +SKIP
        4
        """
        return len(self.entries)

    @property
    def solutions(self) -> Dict[Symbol, Expr]:
        r"""
        Mapping from each order symbol to its solved value.

        Returns
        -------
        dict
            ``{x_k: value}`` for every solved order, i.e. the coefficients of
            the expansion keyed by their symbols :math:`x_0, x_1, \dots`.

        Examples
        --------
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> h.solutions                                      # doctest: +SKIP
        {x_0: 1, x_1: -1/3, x_2: 0, x_3: 1/81}
        """
        return {e.symbol: e.solution for e in self.entries}

    @property
    def equations(self) -> List[Eq]:
        r"""
        The per-order equations, in order.

        Returns
        -------
        list of sympy.Eq
            The equation :math:`x_k` satisfies at each order ``k``, from 0 up to
            ``N``.

        Examples
        --------
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> h.equations                                      # doctest: +SKIP
        [Eq(x_0**3 - 1, 0), Eq(3*x_1 + 1, 0), Eq(3*x_2, 0), Eq(3*x_3 - 1/27, 0)]
        """
        return [e.equation for e in self.entries]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Compare the algebraic perturbation expansion against a SciPy root-finder.

        Evaluates the assembled :attr:`expansion` and the exact root (found
        numerically with :func:`scipy.optimize.fsolve`) as functions of
        :math:`\varepsilon` over the range :math:`[0, \varepsilon_{\max}]`, then
        returns both together with error norms and a comparison figure.  The two
        curves should coincide as :math:`\varepsilon \to 0` and separate as it
        grows.

        Parameters
        ----------
        eps : float
            Representative value of :math:`\varepsilon`, used to set the default
            :math:`\varepsilon_{\max} = 3\,\varepsilon`.
        params : dict, optional
            Numerical values for any extra free symbols in the equation.
        **kwargs
            ``eps_max`` : float
                Maximum :math:`\varepsilon` for the plot.  Default ``3*eps``.
            ``n_points`` : int
                Number of :math:`\varepsilon` values to evaluate.  Default 100.
            ``filename`` : str
                If given, save the figure to this path.

        Returns
        -------
        dict
            Dictionary with keys ``'eps'`` (the :math:`\varepsilon` grid),
            ``'perturbation'`` (expansion values), ``'numerical'`` (exact root),
            ``'errors'`` (L2/Linf absolute and relative errors), ``'settings'``
            (the SciPy solver settings used), and ``'fig'`` (the figure).

        Notes
        -----
        This routine imports :mod:`matplotlib`; select a non-interactive backend
        (``import matplotlib; matplotlib.use('Agg')``) before calling in a
        headless environment.

        Examples
        --------
        >>> import matplotlib; matplotlib.use('Agg')
        >>> from asymptotics import AlgebraicEquation
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> res = h.compare_numeric(eps=0.3)                 # doctest: +SKIP
        >>> sorted(res.keys())                               # doctest: +SKIP
        ['eps', 'errors', 'fig', 'numerical', 'perturbation', 'settings']
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
        r"""
        Evaluate the perturbation expansion numerically.

        Substitutes a concrete value of :math:`\varepsilon` into the assembled
        :attr:`expansion` :math:`\sum_k x_k \varepsilon^{k}` and returns the
        resulting number.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter :math:`\varepsilon`.
        at : array-like, optional
            Unused for algebraic equations (the expansion has no independent
            variable); accepted for API uniformity with the ODE hierarchies.
        params : dict, optional
            Numerical values for any extra free symbols in the expansion.

        Returns
        -------
        float or numpy.ndarray
            A ``float`` if ``eps`` is a scalar, or a 1-D array (one value per
            :math:`\varepsilon`) if ``eps`` is a list.

        Examples
        --------
        >>> from asymptotics import AlgebraicEquation
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> h.eval(eps=0.1)                                  # doctest: +SKIP
        0.966679012345679
        >>> h.eval(eps=[0.1, 0.2])                           # doctest: +SKIP
        array([0.96667901, 0.9334321 ])
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
        r"""
        Return the assembled expansion as a LaTeX string.

        Returns
        -------
        str
            :func:`sympy.latex` rendering of :attr:`expansion`.

        Examples
        --------
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> h.latex_expansion()                              # doctest: +SKIP
        '\\frac{eps^{3}}{81} - \\frac{eps}{3} + 1'
        """
        return latex(self.expansion)

    def latex_equations(self) -> List[str]:
        r"""
        Return the per-order equations as LaTeX strings.

        Returns
        -------
        list of str
            One :func:`sympy.latex` rendering per order, in order 0..N.

        Examples
        --------
        >>> h = AlgebraicEquation("x**3 + eps*x - 1", dependent="x",
        ...                       small_param="eps").expand_regular(order=3)  # doctest: +SKIP
        >>> h.latex_equations()                              # doctest: +SKIP
        ['x_{0}^{3} - 1 = 0', '3 x_{1} + 1 = 0', '3 x_{2} = 0', '3 x_{3} - \\frac{1}{27} = 0']
        """
        return [latex(e.equation) for e in self.entries]

    def _show_body(self, orders=None) -> None:
        """Show order blocks and expansion without title header.
        Used by SystemHierarchy to embed per-variable output."""
        from asymptotics.display.jupyter import _show_jupyter_body
        _show_jupyter_body(self, orders=orders)
