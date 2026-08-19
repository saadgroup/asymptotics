"""
asymptotics.core.system_hierarchy
=================================
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
    r"""
    Perturbation hierarchy for a coupled algebraic system.

    Returned by ``AlgebraicSystem.expand_regular(...)``.  It bundles, per
    dependent variable, an :class:`~asymptotics.OrderHierarchy` (so individual
    variables are reachable via ``sol["x"]``) together with the *full coupled*
    system solved order by order (used by :meth:`show`).  The container is
    dict-like over its variable names and exposes the standard four-method
    API (:meth:`eval`, :meth:`to_latex`, :meth:`compare_numeric`,
    :meth:`show`).

    Attributes
    ----------
    variables : list of str
        Dependent-variable names, in the order they were declared.
    hierarchies : dict
        ``{variable_name: OrderHierarchy}`` — the per-variable expansion.
    coupled_orders : dict
        ``{order_k: {'equations', 'unknowns', 'solutions'}}`` — the coupled
        system, its order-:math:`k` unknowns, and the solved terms at each
        order, used for display.
    small_param : sympy.Symbol
        The small parameter symbol the system is expanded in.

    Examples
    --------
    >>> from asymptotics import AlgebraicSystem
    >>> sys = AlgebraicSystem(equations=["x**2 + eps*y - 1",
    ...                                   "y**2 + eps*x - 1"],
    ...                       dependents=["x", "y"], small_param="eps")
    >>> sol = sys.expand_regular(order=3)
    >>> sol.variables
    ['x', 'y']
    >>> r = sol.eval(eps=0.1)
    >>> round(float(r['x']), 6)
    0.95125
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
        """
        Return the per-variable :class:`OrderHierarchy` for ``var``.

        Parameters
        ----------
        var : str
            A dependent-variable name (e.g. ``"x"``).

        Returns
        -------
        OrderHierarchy
            The expansion hierarchy for that variable, giving access to
            ``sol["x"][k]`` (order-:math:`k` term) and ``sol["x"].expansion``.

        Raises
        ------
        KeyError
            If ``var`` is not a variable of this system.  The message lists the
            available variable names.
        """
        if var not in self.hierarchies:
            raise KeyError(
                f"Variable '{var}' not in system. "
                f"Available: {list(self.hierarchies.keys())}"
            )
        return self.hierarchies[var]

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the dependent-variable names, in declaration order.

        Yields
        ------
        str
            Each variable name, so ``for v in sol:`` and ``list(sol)`` behave
            like iterating the keys of a dict.
        """
        return iter(self.variables)

    def items(self) -> Iterator[Tuple]:
        """
        Iterate over ``(variable_name, OrderHierarchy)`` pairs.

        Yields
        ------
        tuple of (str, OrderHierarchy)
            One pair per variable, in declaration order — the dict-style way to
            walk every per-variable expansion, e.g.
            ``for name, h in sol.items(): ...``.
        """
        return ((v, self.hierarchies[v]) for v in self.variables)

    def __len__(self):
        return len(self.variables)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def eval(self, eps, **kwargs):
        r"""
        Evaluate the coupled-system expansion numerically.

        Substitutes the small parameter into every variable's expansion and
        returns concrete numbers for each variable.  Thin wrapper over
        :func:`asymptotics.eval.eval_hierarchy`.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter.
        **kwargs
            Forwarded to :func:`~asymptotics.eval.eval_hierarchy` (e.g.
            ``params`` for any remaining symbolic parameters).

        Returns
        -------
        dict
            ``{variable_name: value}``.  Each value is a ``float`` when ``eps``
            is scalar, or an ``ndarray`` (one entry per eps) when ``eps`` is a
            list.

        Examples
        --------
        >>> from asymptotics import AlgebraicSystem
        >>> sys = AlgebraicSystem(equations=["x**2 + eps*y - 1",
        ...                                   "y**2 + eps*x - 1"],
        ...                       dependents=["x", "y"], small_param="eps")
        >>> sol = sys.expand_regular(order=3)
        >>> r = sol.eval(eps=0.1)
        >>> round(float(r['x']), 6), round(float(r['y']), 6)
        (0.95125, 0.95125)
        >>> r = sol.eval(eps=[0.1, 0.2, 0.3])
        >>> r['x'].shape
        (3,)
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, **kwargs)

    def to_latex(self, environment='align', show_orders=False, filename=None):
        r"""
        Export this coupled-system expansion as LaTeX source.

        Thin wrapper over :func:`asymptotics.latex_export.to_latex`; produces
        one expansion block per variable, with the small parameter always
        rendered as ``\varepsilon``.

        Parameters
        ----------
        environment : str, optional
            LaTeX math environment — ``'align'`` (default), ``'equation'``, or
            ``'gather'``.
        show_orders : bool, optional
            If ``True``, list each order term separately in addition to the
            assembled expansion.  Default ``False``.
        filename : str, optional
            If given, write the source to this file; otherwise print it to the
            console.  The string is returned in both cases.

        Returns
        -------
        str
            The LaTeX source string: one expansion block per variable
            (``x(varepsilon) = ...``), optionally preceded by the order-by-order
            terms when ``show_orders=True``.

        See Also
        --------
        show : Display the coupled system order by order, including the full
            coupled system at each order.
        __getitem__ : ``sol["x"].to_latex()`` exports a single variable's
            expansion on its own.

        Examples
        --------
        >>> from asymptotics import AlgebraicSystem
        >>> sys = AlgebraicSystem(equations=["x**2 + eps*y - 1",
        ...                                   "y**2 + eps*x - 1"],
        ...                       dependents=["x", "y"], small_param="eps")
        >>> sol = sys.expand_regular(order=3)
        >>> src = sol.to_latex()   # doctest: +SKIP
        >>> src.startswith('%')    # doctest: +SKIP
        True
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)

    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Compare the system expansion against ``scipy.optimize.root``.

        For each eps value the coupled system is solved numerically (warm-started
        from the previous solve) and plotted against the perturbation expansion,
        one subplot per variable.  Thin wrapper over
        :func:`asymptotics.numerics.compare_numeric`.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter used as the x-axis; the numerical
            root and the expansion are compared over these values.
        params : dict, optional
            Numerical values for any remaining symbolic parameters.
        **kwargs
            Forwarded to :func:`~asymptotics.numerics.compare_numeric`.

        Returns
        -------
        dict
            ``'eps'`` : ndarray
                The sorted eps grid.
            ``'perturbation'`` : dict
                ``{variable: ndarray}`` — expansion values over ``'eps'``.
            ``'numerical'`` : dict
                ``{variable: ndarray}`` — numerical roots over ``'eps'``.
            ``'fig'`` : matplotlib.figure.Figure
                The comparison figure.
            ``'errors'`` : dict
                ``{eps_value: {variable: {'abs', 'rel'}}}`` — per-variable
                absolute and relative error at each eps value.
            ``'settings'`` : dict
                The SciPy solver settings used for the reference.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use('Agg')
        >>> from asymptotics import AlgebraicSystem
        >>> sys = AlgebraicSystem(equations=["x**2 + eps*y - 1",
        ...                                   "y**2 + eps*x - 1"],
        ...                       dependents=["x", "y"], small_param="eps")
        >>> sol = sys.expand_regular(order=3)
        >>> res = sol.compare_numeric(eps=[0.1, 0.2, 0.3])
        >>> sorted(res)
        ['eps', 'errors', 'fig', 'numerical', 'perturbation', 'settings']
        >>> res['settings']['solver']
        'scipy.optimize.root'
        """
        from asymptotics.numerics import compare_numeric
        return compare_numeric(self, eps, params=params, **kwargs)

    def show(self, orders=None, mode: str = "auto") -> None:
        r"""
        Display the coupled hierarchy: the full system solved order by order.

        Renders the ansatz for each variable, then, at each order, the full
        coupled system together with the terms it determines, and finally the
        assembled expansion for every variable, for example::

            O(ε¹):  2x₁ + y₀ = 0
                    x₀ + 2y₁ = 0
                    ⟹  x₁ = -1/2,  y₁ = -1/2

        In a Jupyter environment the output is typeset with LaTeX/MathJax; in a
        plain terminal it falls back to a Unicode text rendering.  This method
        prints/displays and returns ``None`` (use :meth:`to_latex` to capture
        the LaTeX source as a string).

        Parameters
        ----------
        orders : list of int, optional
            If given, only these orders are shown.  By default every computed
            order is displayed.
        mode : str, optional
            Rendering mode: ``"auto"`` (default — LaTeX if IPython is
            available, else text), ``"jupyter"`` (force LaTeX display), or
            ``"text"`` (force the plain-text rendering).

        Returns
        -------
        None
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
                is_neg = val.could_extract_minus_sign()
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
