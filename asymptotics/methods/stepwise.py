r"""
asymptotics.methods.stepwise
============================
Interactive, order-by-order (step-by-step) regular perturbation for ODEs.

Where :meth:`asymptotics.ODE.expand_regular` solves every order in one shot,
this module hands back control: the order equations are set up symbolically
immediately, but *nothing is solved* until you ask. Each order can then be
solved with SymPy, or given a hand-supplied solution, one at a time. This is
the tool of choice when SymPy stalls on a hard order, when you want to inspect
or manipulate an order equation before solving it, or when teaching the
mechanics of a perturbation expansion.

Order equations
---------------
Substituting the regular-perturbation ansatz

.. math::

    u(t;\varepsilon) = \sum_{k=0}^{N} \varepsilon^{k}\, u_k(t)

into the ODE and collecting powers of :math:`\varepsilon` gives one equation
per order. The order-:math:`k` equation is linear in the unknown
:math:`u_k(t)`, with a forcing term built from the already-known lower-order
solutions :math:`u_0, \dots, u_{k-1}` — for example, for the Duffing equation
:math:`u'' + u + \varepsilon u^3 = 0`,

.. math::

    \mathcal{O}(\varepsilon^0):\quad & u_0'' + u_0 = 0, \\
    \mathcal{O}(\varepsilon^1):\quad & u_1'' + u_1 = -u_0^{\,3}.

Each order is exposed through :attr:`StepwiseOrderEntry.ode`, which returns a
:class:`_OdePair` wrapping both the raw *symbolic* form (with the
:math:`u_j(t)` still symbolic) and the *substituted* form (lower-order
solutions inserted). Call :meth:`_OdePair.as_sympy` to get the underlying
SymPy :class:`~sympy.core.relational.Eq` for direct manipulation, or solve the
order symbolically/numerically yourself and feed the result back with
:meth:`StepwiseOrderEntry.set_solution`.

Usage
-----
>>> sol = eq.begin_expansion(order=2)
>>> sol.show()                         # see all equations, nothing solved yet
>>>
>>> sol[0].solve()                     # try SymPy
>>> sol[0].set_solution(expr)          # or provide manually
>>>
>>> sol[1].solve()                     # SymPy handles linear orders
>>>
>>> sol.solve_all()                    # try all remaining
>>>
>>> sol.expansion                      # available once all solved
>>> sol.show()
>>> sol.to_latex()
>>> sol.eval(eps=0.1, at=t_vals)
>>> sol.compare_numeric(eps=0.1)
"""

from __future__ import annotations
from sympy import (
    Function, Symbol, symbols, series, expand, diff,
    dsolve, solve, Eq, Add, Integer, sympify,
    exp as _exp, cos as _cos
)
from asymptotics.methods.regular_ode import _bc_value_at_order


# ---------------------------------------------------------------------------
# Order entry for step-by-step expansion
# ---------------------------------------------------------------------------

class StepwiseOrderEntry:
    r"""
    A single order :math:`k` of a step-by-step perturbation expansion.

    One of these is created per order by :func:`begin_expansion_ode` and
    accessed via indexing the hierarchy, ``sol[k]``. It carries the order-:math:`k`
    equation and, once solved, the solution :math:`u_k(t)`. Solve it with
    :meth:`solve` (SymPy) or supply the answer yourself with
    :meth:`set_solution`.

    Attributes
    ----------
    order : int
        The order :math:`k` (the power of :math:`\varepsilon`).
    symbol : sympy.Function
        The unknown at this order, :math:`u_k(t)`.
    general_solution : sympy.Expr or None
        Solution with free integration constants; ``None`` until solved.
    particular_solution : sympy.Expr or None
        Solution with constants fixed by the problem's initial/boundary
        conditions; ``None`` until solved.
    secular : bool
        ``True`` if secular (unbounded, resonant) terms were detected in the
        solution — a signal that regular perturbation is breaking down and a
        method such as Lindstedt–Poincaré or multiple scales is needed.
    is_solved : bool
        Whether this order has been solved or set. Consulted by
        :attr:`ode`, :meth:`solve`, and the hierarchy's finalization.

    See Also
    --------
    ode : the order-:math:`k` equation as a manipulable :class:`_OdePair`.
    solve : attempt a SymPy solution of this order.
    set_solution : supply the order-:math:`k` solution manually.
    """

    def __init__(self, order, ode_symbolic, ode_coeffs, symbol, hierarchy):
        self.order               = order
        self._ode_symbolic       = ode_symbolic   # Eq with uk, lower as Functions
        self._ode_coeffs         = ode_coeffs     # raw coefficients (pre-substitution)
        self.symbol              = symbol
        self._hierarchy          = hierarchy       # back-reference

        self.general_solution    = None
        self.particular_solution = None
        self.secular             = False
        self.is_solved           = False

    @property
    def ode(self):
        r"""
        The order-:math:`k` equation, as a manipulable :class:`_OdePair`.

        The returned pair always carries the *symbolic* form of the equation
        (with the lower-order functions :math:`u_j(t)` left symbolic). If every
        lower order :math:`0, \dots, k-1` has been solved, it also carries the
        *substituted* form, in which those known solutions have been inserted
        so the equation is ready to solve for :math:`u_k`.

        Returns
        -------
        _OdePair
            Pretty-prints both forms. Use :meth:`_OdePair.as_sympy` (or the
            delegated ``.lhs`` / ``.rhs`` / ``.free_symbols`` / ``.subs``) to
            reach the underlying SymPy :class:`~sympy.core.relational.Eq`.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u**3", dependent='u', small_param='eps',
        ...          independent='t', conditions=["u(0) = 1", "u'(0) = 0"])
        >>> sol = eq.begin_expansion(order=2)
        >>> sol[0].ode.as_sympy()                    # leading order
        Eq(u_0(t) + Derivative(u_0(t), (t, 2)), 0)
        >>> sol[0].solve()                           # doctest: +SKIP
        >>> # once order 0 is known, order 1 gains a substituted form:
        >>> sol[1].ode.as_sympy(substituted=False)
        Eq(u_0(t)**3 + u_1(t) + Derivative(u_1(t), (t, 2)), 0)
        """
        h = self._hierarchy
        k = self.order

        # Build substituted version if all lower orders are solved
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            return _OdePair(
                symbolic    = self._ode_symbolic,
                substituted = self._build_substituted_ode(h),
                order       = k,
                eps         = h.small_param,
            )
        else:
            return _OdePair(
                symbolic    = self._ode_symbolic,
                substituted = None,
                order       = k,
                eps         = h.small_param,
            )

    def _build_substituted_ode(self, h):
        """Build the ODE with lower-order solutions substituted."""
        ode_expr = self._ode_coeffs
        for j in range(self.order):
            if h.entries[j].is_solved:
                func = h._u_funcs[j]
                sol  = h.entries[j].particular_solution
                ode_expr = ode_expr.subs(func, sol)
        # Evaluate any Derivative objects that became concrete after substitution
        # (e.g. Derivative(4*eta - 4*eta**2, eta, 4) → 0).
        # SymPy's .subs() replaces the function body but leaves the Derivative
        # wrapper unevaluated; .doit() resolves it.
        ode_expr = ode_expr.doit()
        ode_expr = expand(ode_expr)
        ode_expr = expand(expand(ode_expr.rewrite(_exp)).rewrite(_cos))
        return Eq(ode_expr, 0)

    def solve(self):
        r"""
        Attempt to solve this order automatically with SymPy's ``dsolve``.

        Requires every lower order to be solved first (the order-:math:`k`
        equation depends on :math:`u_0, \dots, u_{k-1}`). On success, the
        substituted equation is solved, the problem's initial/boundary
        conditions are applied to fix the integration constants, secular terms
        are detected, and both the general and particular solutions are stored;
        the order is marked solved. If it was the last pending order, the
        hierarchy's :attr:`~StepwiseHierarchy.expansion` is assembled.

        This method never raises for an unsolvable order: if the lower orders
        are missing, or if ``dsolve`` fails, or the conditions cannot be
        applied, it prints a clear message (with a ``set_solution`` hint) and
        returns ``False`` so an interactive session can continue.

        Returns
        -------
        bool
            ``True`` if the order was solved, ``False`` otherwise.

        See Also
        --------
        set_solution : supply the solution manually when this fails.
        StepwiseHierarchy.solve_all : run :meth:`solve` across all orders.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent='u', small_param='eps',
        ...          independent='t', conditions=['u(0) = 1'])
        >>> sol = eq.begin_expansion(order=2)
        >>> sol[0].solve()                           # doctest: +SKIP
        >>> sol[0].is_solved                         # doctest: +SKIP
        True
        """
        h   = self._hierarchy
        k   = self.order
        t   = h.independent
        uk  = self.symbol

        # Check lower orders are solved
        unsolved_below = [j for j in range(k) if not h.entries[j].is_solved]
        if unsolved_below:
            print(
                f"\n  ⚠️  Cannot solve order {k} — lower orders not yet solved: "
                f"{unsolved_below}\n"
                f"  Solve or set solutions for orders {unsolved_below} first.\n"
            )
            return False

        # Build the substituted ODE
        ode_eq = self._build_substituted_ode(h).rewrite(_cos) if k > 0 \
                 else Eq(expand(self._ode_coeffs), 0)

        # Try dsolve
        try:
            gen_sol = dsolve(ode_eq, uk)
            if isinstance(gen_sol, list):
                gen_sol = gen_sol[0]
            gen_expr = gen_sol.rhs
        except Exception as e:
            self._print_solve_failure(ode_eq, e)
            return False

        # Apply conditions
        part_expr = h._apply_conditions(gen_expr, k, t)
        if part_expr is None:
            self._print_solve_failure(ode_eq, "Could not apply conditions")
            return False

        self._store_solution(gen_expr, part_expr, t)
        h._known_solutions[uk] = part_expr
        print(f"  ✓  Order {k} solved: {dep_name(uk)} = {part_expr}")
        # Auto-finalize if all orders are now solved
        if h.n_pending == 0:
            h._finalize()
        return True

    def residual(self, expr):
        r"""
        Residual of the order-:math:`k` equation for a candidate solution.

        Substitutes ``expr`` for :math:`u_k(t)` into the order-:math:`k`
        equation (with all solved lower orders already inserted), evaluates any
        resulting derivatives, and simplifies. A candidate that solves the
        order returns :math:`0`.

        Parameters
        ----------
        expr : sympy.Expr or str
            Candidate solution :math:`u_k(t)`.

        Returns
        -------
        sympy.Expr
            The simplified residual :math:`\mathcal{L}_k[u_k] - r_k`. Zero iff
            ``expr`` satisfies the order-:math:`k` equation.

        Notes
        -----
        The check uses the *substituted* equation when every lower order is
        solved, and the raw leading-order equation otherwise. It is a symbolic
        identity check: if the residual cannot be simplified to a literal zero
        (e.g. it depends on an as-yet-undetermined constant), it is returned
        unchanged rather than forced to zero.
        """
        from sympy import simplify, trigsimp
        if isinstance(expr, str):
            expr = sympify(expr)
        h  = self._hierarchy
        k  = self.order
        uk = self.symbol
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            eq = self._build_substituted_ode(h)
        else:
            eq = Eq(expand(self._ode_coeffs), 0)
        res = (eq.lhs - eq.rhs).subs(uk, expr).doit()
        return trigsimp(simplify(expand(res)))

    def set_solution(self, expr, check=True):
        r"""
        Supply the solution :math:`u_k(t)` for this order by hand.

        Use this when SymPy cannot solve the order, or when you have obtained
        the solution elsewhere (by hand, or in Mathematica/Maple). The
        expression should be the *particular* solution with all free
        constants already fixed by the conditions — it is stored as both the
        general and particular solution, secular terms are detected, and the
        order is marked solved (finalizing the expansion if it was the last
        pending order).

        By default the supplied expression is **verified** against the
        order-:math:`k` equation: it is substituted in and the residual is
        simplified, and a :class:`ValueError` is raised if the residual is a
        provably non-zero expression. This catches a common mistake — supplying
        a function that does not actually solve the stated equation (for
        example, forgetting an eigenvalue or forcing term). Pass
        ``check=False`` to bypass the verification (e.g. when the residual
        vanishes only after a later solvability condition is imposed).

        Parameters
        ----------
        expr : sympy.Expr or str
            The solution :math:`u_k(t)`. Strings are parsed with
            :func:`sympy.sympify`.
        check : bool, optional
            If ``True`` (default), verify that ``expr`` satisfies the
            order-:math:`k` equation and raise :class:`ValueError` on a
            non-zero residual. Set ``False`` to skip the check.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``check`` is ``True`` and ``expr`` does not satisfy the
            order-:math:`k` equation (non-zero residual). The message reports
            the residual so the discrepancy is visible.

        See Also
        --------
        solve : attempt an automatic SymPy solution instead.
        residual : compute the residual without storing the solution.

        Examples
        --------
        >>> from sympy import sympify
        >>> sol[0].set_solution(sympify("4*eta*(1 - eta)"))   # doctest: +SKIP
        >>> sol[0].set_solution("4*eta - 4*eta**2")           # doctest: +SKIP
        """
        h  = self._hierarchy
        t  = h.independent
        uk = self.symbol

        if isinstance(expr, str):
            expr = sympify(expr)

        # Verify the candidate against the order-k equation.
        if check:
            res = self.residual(expr)
            if res != 0:
                raise ValueError(
                    f"\n\n  set_solution(): the supplied expression does not satisfy "
                    f"the order-{self.order} equation.\n"
                    f"  Residual (should be 0): {res}\n\n"
                    f"  Common causes: a missing eigenvalue/forcing term, or a "
                    f"solution that is only valid after a later solvability\n"
                    f"  condition is imposed. If the residual is expected to vanish "
                    f"only under such a condition, pass check=False.\n"
                )

        self._store_solution(expr, expr, t)
        h._known_solutions[uk] = expr
        print(f"  ✓  Order {self.order} set manually: {dep_name(uk)} = {expr}")
        # Auto-finalize if all orders are now solved
        if h.n_pending == 0:
            h._finalize()

    def _store_solution(self, gen_expr, part_expr, t):
        """Store the solution and detect secular terms."""
        from asymptotics.methods.regular_ode import _has_secular_terms
        self.general_solution    = gen_expr
        self.particular_solution = expand(part_expr)
        self.secular             = _has_secular_terms(part_expr, t)
        self.is_solved           = True

    def _print_solve_failure(self, ode_eq, error):
        k = self.order
        eps_str = str(self._hierarchy.small_param)
        sup = ['⁰','¹','²','³','⁴','⁵']
        order_str = f"ε{sup[k]}" if k < len(sup) else f"ε^{k}"
        print(
            f"\n  ✗  Could not solve O({order_str}) equation automatically:\n"
            f"     {ode_eq}\n\n"
            f"  Provide the solution manually:\n"
            f"     sol[{k}].set_solution(your_expr)\n"
            f"  or solve in Mathematica/Maple and paste the result.\n"
        )

    def __repr__(self):
        status = "solved" if self.is_solved else "not solved"
        return f"StepwiseOrderEntry(order={self.order}, {status})"


def dep_name(uk):
    """Extract u_k name from Function."""
    return str(uk.func)


# ---------------------------------------------------------------------------
# ODE pair: symbolic + substituted
# ---------------------------------------------------------------------------

class _OdePair:
    r"""Both forms of an order-:math:`k` equation, with pass-through to SymPy.

    Returned by :attr:`StepwiseOrderEntry.ode`. It holds the *symbolic* form
    of the order equation (lower-order functions :math:`u_j(t)` left symbolic)
    and, when the lower orders are known, the *substituted* form (those
    solutions inserted). It pretty-prints both, but is otherwise a thin,
    transparent wrapper over a live SymPy :class:`~sympy.core.relational.Eq`:

    - :meth:`as_sympy` returns the underlying ``Eq`` (substituted by default);
    - :attr:`lhs` / :attr:`rhs` give that equation's sides;
    - any other attribute access is delegated to the equation (see
      :meth:`__getattr__`), so ``.free_symbols``, ``.subs(...)``,
      ``.rewrite(...)``, ``.atoms(...)``, ``.args`` all work directly.

    Attributes
    ----------
    symbolic : sympy.Eq
        The order equation with lower-order unknowns still symbolic.
    substituted : sympy.Eq or None
        The order equation with known lower-order solutions inserted; ``None``
        until all lower orders are solved.
    order : int
        The order :math:`k`.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("u'' + u + eps*u**3", dependent='u', small_param='eps',
    ...          independent='t', conditions=["u(0) = 1", "u'(0) = 0"])
    >>> sol = eq.begin_expansion(order=2)
    >>> pair = sol[0].ode
    >>> pair.lhs                                  # delegated to the SymPy Eq
    u_0(t) + Derivative(u_0(t), (t, 2))
    >>> pair.rhs
    0
    >>> pair.free_symbols                         # delegated attribute
    {t}
    """

    def __init__(self, symbolic, substituted, order, eps):
        self.symbolic    = symbolic
        self.substituted = substituted
        self.order       = order
        self._eps        = eps

    # -- Low-level access -------------------------------------------------
    # The order-k equation is a live SymPy object.  In addition to the
    # ``symbolic`` and ``substituted`` attributes, the pair transparently
    # exposes the underlying equation's SymPy interface so it can be
    # manipulated directly, e.g.::
    #
    #     eqn = sol[k].ode                 # this pair (pretty-prints)
    #     eqn.lhs, eqn.rhs                 # SymPy expressions
    #     eqn.free_symbols                 # -> set of symbols
    #     eqn.subs(...), eqn.rewrite(...)  # SymPy methods
    #     raw = sol[k].ode.as_sympy()      # the SymPy Eq itself
    #
    def as_sympy(self, substituted=True):
        r"""Return the underlying SymPy ``Eq`` for direct manipulation.

        This is the escape hatch to raw SymPy: the returned object is an
        ordinary :class:`~sympy.core.relational.Eq` you can differentiate,
        substitute into, ``rewrite``, ``lambdify``, or solve yourself.

        Parameters
        ----------
        substituted : bool, optional
            If ``True`` (default), return the form with the known lower-order
            solutions inserted (falling back to the symbolic form when no
            substituted form exists yet). If ``False``, always return the
            purely symbolic form.

        Returns
        -------
        sympy.Eq
            The order-:math:`k` equation, ``<expr> = 0``.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u'' + u + eps*u**3", dependent='u', small_param='eps',
        ...          independent='t', conditions=["u(0) = 1", "u'(0) = 0"])
        >>> sol = eq.begin_expansion(order=2)
        >>> eqn = sol[0].ode.as_sympy()
        >>> eqn
        Eq(u_0(t) + Derivative(u_0(t), (t, 2)), 0)
        >>> eqn.lhs                                  # a normal SymPy expression
        u_0(t) + Derivative(u_0(t), (t, 2))
        """
        if substituted and self.substituted is not None:
            return self.substituted
        return self.symbolic

    @property
    def lhs(self):
        """Left-hand side of the (substituted, if available) order equation.

        Equivalent to ``self.as_sympy().lhs`` — a SymPy expression.
        """
        return self.as_sympy().lhs

    @property
    def rhs(self):
        """Right-hand side of the (substituted, if available) order equation.

        Equivalent to ``self.as_sympy().rhs`` — normally the SymPy integer 0.
        """
        return self.as_sympy().rhs

    def __getattr__(self, name):
        r"""Delegate unknown attribute access to the underlying SymPy ``Eq``.

        Any non-dunder attribute not defined on the pair itself is looked up on
        the substituted equation (or the symbolic one when no substituted form
        exists). This makes the pair behave like the equation for read access,
        so SymPy methods and properties work transparently::

            pair.free_symbols        # -> set of symbols
            pair.subs(...)           # substitute
            pair.rewrite(...)        # rewrite in another basis
            pair.atoms(...), pair.args

        Raises
        ------
        AttributeError
            If the name is a dunder, or is absent on the underlying equation.
        """
        # Delegate unknown (non-dunder) attribute access to the underlying
        # SymPy equation so the order equation can be manipulated directly:
        # .free_symbols, .subs(...), .rewrite(...), .atoms(...), .args, etc.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        d = object.__getattribute__(self, '__dict__')
        target = d.get('substituted') if d.get('substituted') is not None \
            else d.get('symbolic')
        if target is not None and hasattr(target, name):
            return getattr(target, name)
        raise AttributeError(f"'_OdePair' object has no attribute {name!r}")

    def __repr__(self):
        from sympy import latex
        eps_sym = self._eps
        sup = ['⁰','¹','²','³','⁴','⁵']
        k   = self.order
        order_str = f"ε{sup[k]}" if k < len(sup) else f"ε^{k}"

        s = f"O({order_str}) — symbolic:\n  {self.symbolic}"
        if self.substituted is not None:
            s += f"\n\nO({order_str}) — substituted:\n  {self.substituted}"
        return s

    def _repr_latex_(self):
        """Rich Jupyter display."""
        try:
            from IPython.display import display, Math, HTML
            from sympy import latex
            eps_sym = self._eps
            k       = self.order
            sup_u   = ['⁰','¹','²','³','⁴','⁵']
            order_str = f"ε{sup_u[k]}" if k < len(sup_u) else f"ε^{{{k}}}"

            def _lx(expr):
                return latex(expr).replace(str(eps_sym), r'\varepsilon')

            eps_label = r'\varepsilon^{' + str(k) + '}' if k > 1 \
                else (r'\varepsilon' if k == 1 else r'\varepsilon^{0}')

            html  = f"<div style='margin:6px 0;font-weight:500;border-left:3px solid #7F77DD;padding-left:8px;'>"
            html += f"O({order_str})</div>"
            display(HTML(html))

            display(Math(
                r'\textbf{Symbolic:} \quad '
                + _lx(self.symbolic.lhs) + ' = 0'
            ))

            if self.substituted is not None:
                display(Math(
                    r'\textbf{Substituted:} \quad '
                    + _lx(self.substituted.lhs) + ' = 0'
                ))
        except Exception:
            pass
        return ''


# ---------------------------------------------------------------------------
# Stepwise hierarchy
# ---------------------------------------------------------------------------

class StepwiseHierarchy:
    r"""
    A regular-perturbation hierarchy solved order by order, under user control.

    Created by :meth:`asymptotics.ODE.begin_expansion`. The order equations for
    :math:`u_0, \dots, u_N` are set up symbolically at construction, but none is
    solved until requested. Index the hierarchy to reach a single order,
    ``sol[k]`` (a :class:`StepwiseOrderEntry`), and solve it with
    ``sol[k].solve()`` or ``sol[k].set_solution(expr)``; or run
    :meth:`solve_all` to attempt every remaining order.

    Once **all** orders are solved the assembled expansion

    .. math::

        u(t;\varepsilon) = \sum_{k=0}^{N} \varepsilon^{k}\, u_k(t)
                           + \mathcal{O}(\varepsilon^{N+1})

    becomes available on :attr:`expansion`, and the shared hierarchy API
    (:meth:`show`, :meth:`to_latex`, :meth:`eval`, :meth:`compare_numeric`)
    turns on. Those four methods raise :class:`RuntimeError` if called while any
    order is still pending.

    Attributes
    ----------
    entries : list of StepwiseOrderEntry
        The per-order entries, index :math:`k = 0, \dots, N`.
    small_param : sympy.Symbol
        The small parameter :math:`\varepsilon`.
    independent : sympy.Symbol
        The independent variable :math:`t`.
    expansion : sympy.Expr or None
        The assembled expansion; ``None`` until all orders are solved.

    See Also
    --------
    n_solved, n_pending : progress counters.
    begin_expansion_ode : the constructor.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("u' + u + eps*u**2", dependent='u', small_param='eps',
    ...          independent='t', conditions=['u(0) = 1'])
    >>> sol = eq.begin_expansion(order=2)
    >>> len(sol), sol.n_solved, sol.n_pending
    (3, 0, 3)
    >>> sol.solve_all()                              # doctest: +SKIP
    >>> sol.expansion                                # doctest: +SKIP
    eps**2*(exp(-t) - 2*exp(-2*t) + exp(-3*t)) + eps*(-exp(-t) + exp(-2*t)) + exp(-t)
    """

    def __init__(self):
        self.entries         = []     # list of StepwiseOrderEntry
        self.small_param     = None
        self.independent     = None
        self._problem        = None
        self._problem_type   = None
        self._method         = "Regular perturbation (step-by-step)"
        self._u_funcs        = []
        self._known_solutions = {}   # u_k func -> particular solution expr
        self._n_orders       = 0
        self.expansion       = None  # set after finalize

    def __getitem__(self, k: int) -> StepwiseOrderEntry:
        r"""Return the order-:math:`k` entry, ``sol[k]``.

        Parameters
        ----------
        k : int
            Order index, :math:`0 \le k \le N`.

        Returns
        -------
        StepwiseOrderEntry

        Raises
        ------
        IndexError
            If ``k`` is outside ``0 .. len(sol) - 1``. Negative indexing is
            not supported.
        """
        if k < 0 or k >= len(self.entries):
            raise IndexError(
                f"\n\n  Order {k} out of range. Available: 0 to {len(self.entries)-1}\n"
            )
        return self.entries[k]

    def __len__(self):
        r"""Number of orders in the hierarchy, i.e. :math:`N + 1`."""
        return len(self.entries)

    @property
    def n_solved(self):
        """int : How many orders have been solved or set so far."""
        return sum(1 for e in self.entries if e.is_solved)

    @property
    def n_pending(self):
        """int : How many orders remain unsolved."""
        return sum(1 for e in self.entries if not e.is_solved)

    def solve_all(self):
        r"""
        Attempt to solve every still-unsolved order, in ascending order.

        Calls :meth:`StepwiseOrderEntry.solve` on each pending order from the
        lowest up. Stops at the first order SymPy cannot handle, printing a
        ``set_solution`` hint; solve that order manually and call
        :meth:`solve_all` again to resume. When the last order is solved the
        expansion is finalized and :attr:`expansion` becomes available.

        Returns
        -------
        None

        See Also
        --------
        StepwiseOrderEntry.solve, StepwiseOrderEntry.set_solution

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent='u', small_param='eps',
        ...          independent='t', conditions=['u(0) = 1'])
        >>> sol = eq.begin_expansion(order=2)
        >>> sol.solve_all()                          # doctest: +SKIP
        >>> sol.n_pending                            # doctest: +SKIP
        0
        """
        for e in self.entries:
            if not e.is_solved:
                success = e.solve()
                if not success:
                    print(
                        f"\n  Stopped at order {e.order}. "
                        f"Provide solution manually with:\n"
                        f"    sol[{e.order}].set_solution(your_expr)\n"
                        f"  then call sol.solve_all() again.\n"
                    )
                    return
        self._finalize()

    def _check_all_solved(self, method_name):
        if self.n_pending > 0:
            pending = [e.order for e in self.entries if not e.is_solved]
            solved  = [e.order for e in self.entries if e.is_solved]
            lines   = []
            for k in pending:
                lines.append(
                    f"    sol[{k}].solve()   or   sol[{k}].set_solution(expr)"
                )
            raise RuntimeError(
                f"\n\n  Cannot call '{method_name}' — not all orders are solved.\n"
                f"  Solved:  {solved}\n"
                f"  Pending: {pending}\n\n"
                f"  Solve remaining orders:\n" +
                "\n".join(lines) +
                f"\n  Or: sol.solve_all()\n"
            )

    def _finalize(self):
        """Assemble expansion once all orders are solved."""
        eps = self.small_param
        self.expansion = Add(*[
            self.entries[k].particular_solution * eps**k
            for k in range(len(self.entries))
        ])
        # Also set expansion_t for Lindstedt compatibility
        self.expansion_t = self.expansion
        print(f"\n  ✓  All orders solved. expansion is now available.\n")

    def _apply_conditions(self, gen_expr, k, t):
        """Apply conditions to fix integration constants."""
        from sympy import diff as _diff, Eq as _Eq, solve as _solve
        from sympy import nan as _nan, zoo as _zoo
        from sympy import limit as _lim
        from asymptotics.core.conditions import LimitCondition
        from asymptotics.methods.regular_ode import _apply_limit_condition, _has_secular_terms

        conds = self._problem.conditions
        deriv_syms = self._problem._deriv_syms

        free_consts = sorted(
            [s for s in gen_expr.free_symbols
             if str(s).startswith('C') and str(s)[1:].isdigit()],
            key=lambda s: int(str(s)[1:])
        )

        eps = self.small_param
        cond_equations = []
        for cond in conds:
            if isinstance(cond, LimitCondition):
                eq = _apply_limit_condition(
                    cond, gen_expr, t,
                    self._problem._dependent_name,
                    deriv_syms, k, eps=eps
                )
                if eq is not None and eq is not True:
                    cond_equations.append(eq)
            else:
                pt  = cond.point
                val = _bc_value_at_order(cond.value, eps, k)
                if cond.deriv_order == 0:
                    expr_at_pt = gen_expr.subs(t, pt)
                else:
                    expr_at_pt = _diff(gen_expr, t, cond.deriv_order).subs(t, pt)

                from sympy import nan as _nan2, zoo as _zoo2
                if expr_at_pt in (_nan2, _zoo2) or expr_at_pt.has(_nan2, _zoo2):
                    expr_at_pt = _lim(
                        gen_expr if cond.deriv_order == 0
                        else _diff(gen_expr, t, cond.deriv_order),
                        t, pt, '+'
                    )
                cond_equations.append(_Eq(expr_at_pt, val))

        if not free_consts:
            return gen_expr

        try:
            const_sol = _solve(cond_equations, free_consts)
            if isinstance(const_sol, dict):
                return gen_expr.subs(const_sol)
            elif isinstance(const_sol, list) and const_sol:
                if isinstance(const_sol[0], dict):
                    return gen_expr.subs(const_sol[0])
            return gen_expr
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Standard API — available once all orders solved
    # ------------------------------------------------------------------

    def show(self, mode: str = "auto") -> None:
        """
        Display the hierarchy: order equations plus solutions where available.

        Unlike the other three standard-API methods, :meth:`show` works at any
        stage — it shows every order's equation (symbolic, and substituted once
        the lower orders are known), the solution for each solved order, and the
        assembled expansion once everything is solved.

        Parameters
        ----------
        mode : {'auto', 'text', 'latex'}, optional
            Rendering mode. ``'auto'`` (default) renders LaTeX in Jupyter and
            plain text in a terminal.

        Returns
        -------
        None
            Output is displayed as a side effect.

        Examples
        --------
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent='u', small_param='eps',
        ...          independent='t', conditions=['u(0) = 1'])
        >>> sol = eq.begin_expansion(order=2)
        >>> sol.show(mode='text')                    # doctest: +SKIP
        """
        _show_stepwise(self, mode=mode)

    def to_latex(self, environment='align', show_orders=False, filename=None):
        r"""
        Export the solved expansion as LaTeX source.

        Requires all orders to be solved.

        Parameters
        ----------
        environment : str, optional
            LaTeX math environment: ``'align'`` (default), ``'equation'``, or
            ``'gather'``.
        show_orders : bool, optional
            If ``True``, also emit each order :math:`u_k` separately. Default
            ``False``.
        filename : str, optional
            If given, write the source to this file; otherwise return it.

        Returns
        -------
        str
            The LaTeX source. The small parameter is rendered as
            ``\varepsilon``.

        Raises
        ------
        RuntimeError
            If any order is still pending.
        """
        self._check_all_solved('to_latex')
        from asymptotics.latex_export import to_latex as _to_latex
        return _to_latex(self._as_ode_hierarchy(),
                         environment=environment,
                         show_orders=show_orders,
                         filename=filename)

    def eval(self, eps, at=None, params=None):
        r"""
        Evaluate the assembled expansion numerically. Requires all orders solved.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter :math:`\varepsilon`.
        at : array-like, optional
            Values of the independent variable :math:`t` at which to evaluate.
        params : dict, optional
            Values for any remaining free symbolic parameters.

        Returns
        -------
        numpy.ndarray or dict
            An ndarray of :math:`u` values when ``eps`` is scalar; a dict
            ``{eps: ndarray}`` when ``eps`` is a list.

        Raises
        ------
        RuntimeError
            If any order is still pending.

        Examples
        --------
        >>> import numpy as np
        >>> from asymptotics import ODE
        >>> eq = ODE("u' + u + eps*u**2", dependent='u', small_param='eps',
        ...          independent='t', conditions=['u(0) = 1'])
        >>> sol = eq.begin_expansion(order=2)
        >>> sol.solve_all()                          # doctest: +SKIP
        >>> sol.eval(eps=0.1, at=np.array([0.0, 1.0]))   # doctest: +SKIP
        array([1.        , 0.34609498])
        """
        self._check_all_solved('eval')
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Compare the expansion against a SciPy numerical solution.

        Requires all orders to be solved. Integrates the original ODE with
        SciPy and returns a comparison (with a plot) against the assembled
        expansion.

        Parameters
        ----------
        eps : float
            Value of the small parameter :math:`\varepsilon`.
        params : dict, optional
            Values for any remaining free symbolic parameters.
        **kwargs
            Forwarded to :func:`asymptotics.compare_numeric` (e.g.
            ``plot_range``, ``n_points``, ``filename``).

        Returns
        -------
        dict
            Comparison results, including sampled points, the expansion and
            numerical solutions, error norms, solver settings, and a matplotlib
            figure.

        Raises
        ------
        RuntimeError
            If any order is still pending.
        """
        self._check_all_solved('compare_numeric')
        from asymptotics.numerics import compare_numeric
        return compare_numeric(self, eps, params=params, **kwargs)

    def _as_ode_hierarchy(self):
        """Convert to ODEHierarchy for use with existing display/export."""
        from asymptotics.methods.regular_ode import ODEHierarchy, ODEOrderEntry
        h = ODEHierarchy()
        h.small_param   = self.small_param
        h.independent   = self.independent
        h._method       = self._method
        h._problem_type = self._problem_type
        h._problem      = self._problem
        h.expansion     = self.expansion
        h.expansion_t   = self.expansion
        for e in self.entries:
            entry = ODEOrderEntry(
                order               = e.order,
                ode                 = e._ode_symbolic,
                general_solution    = e.general_solution,
                particular_solution = e.particular_solution,
                constants           = {},
                symbol              = e.symbol,
                secular             = e.secular,
            )
            h.entries.append(entry)
        return h


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _show_stepwise(h, mode="auto"):
    try:
        from IPython.display import display, Math, HTML
        _jupyter = True
    except ImportError:
        _jupyter = False

    if mode == "text" or (mode == "auto" and not _jupyter):
        _show_text(h)
        return

    from IPython.display import display, Math, HTML
    from sympy import latex

    eps = h.small_param
    sup = ['⁰','¹','²','³','⁴','⁵']

    def _lx(expr):
        return latex(expr).replace(str(eps), r'\varepsilon')

    # Title
    n_solved = h.n_solved
    n_total  = len(h.entries)
    status   = f"{n_solved}/{n_total} orders solved"

    display(HTML(
        f"<div style='margin-bottom:6px'>"
        f"<span style='font-size:1.1em;font-weight:600;'>Perturbation Hierarchy</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:#f0f0f0;padding:2px 8px;border-radius:4px;"
        f"font-size:0.85em;color:#555;'>{h._method}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='background:{'#d4edda' if n_solved==n_total else '#fff3cd'};"
        f"padding:2px 8px;border-radius:4px;font-size:0.85em;'>{status}</span>"
        f"</div>"
    ))

    for e in h.entries:
        k = e.order
        order_str = 'ε' + (sup[k] if k < len(sup) else str(k))
        solved_badge = (
            "<span style='color:#28a745;font-size:0.8em;'>✓ solved</span>"
            if e.is_solved else
            "<span style='color:#dc3545;font-size:0.8em;'>✗ pending</span>"
        )

        display(HTML(
            f"<div style='margin-top:10px;font-weight:500;"
            f"border-left:3px solid #7F77DD;padding-left:8px;'>"
            f"Order {order_str} &nbsp; {solved_badge}</div>"
        ))

        # Show symbolic ODE
        display(Math(
            r'\textbf{ODE:} \quad '
            + _lx(e._ode_symbolic.lhs) + r' = 0'
        ))

        # Show substituted ODE if lower orders are solved
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            sub_ode = e._build_substituted_ode(h)
            display(Math(
                r'\textbf{Substituted:} \quad '
                + _lx(sub_ode.lhs) + r' = 0'
            ))

        # Show solution if solved
        if e.is_solved:
            display(Math(
                r'\textbf{Solution:} \quad '
                + _lx(e.symbol) + r' = '
                + _lx(e.particular_solution)
            ))

    # Show expansion if all solved
    if h.n_solved == n_total and h.expansion is not None:
        display(HTML(
            "<div style='margin-top:10px;font-weight:600;'>Expansion:</div>"
        ))
        remainder = r'\mathcal{O}(\varepsilon^{' + str(n_total) + r'})'
        display(Math(
            r'\boxed{u = ' + _lx(h.expansion) + r' + ' + remainder + r'}'
        ))


def _show_text(h):
    sup = ['⁰','¹','²','³','⁴','⁵']
    width = 64
    print("=" * width)
    print(f"  {h._method}  ({h.n_solved}/{len(h.entries)} solved)")
    print("=" * width)

    for e in h.entries:
        k = e.order
        order_str = 'ε' + (sup[k] if k < len(sup) else str(k))
        status = '✓' if e.is_solved else '✗'
        print(f"\n  {status} O({order_str})")
        print(f"    ODE: {e._ode_symbolic}")
        if k > 0 and all(h.entries[j].is_solved for j in range(k)):
            sub = e._build_substituted_ode(h)
            print(f"    Substituted: {sub}")
        if e.is_solved:
            print(f"    Solution: {e.symbol} = {e.particular_solution}")

    if h.expansion is not None:
        print(f"\n  Expansion: {h.expansion}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Setup — called by ODE.begin_expansion()
# ---------------------------------------------------------------------------

def begin_expansion_ode(problem, order: int) -> StepwiseHierarchy:
    r"""
    Build a step-by-step perturbation hierarchy without solving anything.

    Substitutes the regular-perturbation ansatz
    :math:`u = \sum_{k=0}^{N} \varepsilon^{k} u_k(t)` into the problem's ODE,
    expands in powers of :math:`\varepsilon` up to order ``N``, and collects the
    coefficient of each power as the order-:math:`k` equation. The equations are
    stored (symbolic and unsolved) on a fresh :class:`StepwiseHierarchy`; no
    ``dsolve`` is attempted here. This is the backend for
    :meth:`asymptotics.ODE.begin_expansion`.

    Parameters
    ----------
    problem : ODE
        The perturbation problem. Its small parameter must appear in the
        equation.
    order : int
        Highest power :math:`N` of :math:`\varepsilon` to expand to; produces
        orders :math:`0, \dots, N`.

    Returns
    -------
    StepwiseHierarchy
        With ``order + 1`` unsolved entries.

    Raises
    ------
    NoSmallParameterError
        If the small parameter does not appear in the equation.

    Examples
    --------
    >>> from asymptotics import ODE
    >>> eq = ODE("u'' + u + eps*u**3", dependent='u', small_param='eps',
    ...          independent='t', conditions=["u(0) = 1", "u'(0) = 0"])
    >>> sol = eq.begin_expansion(order=2)        # delegates here
    >>> len(sol)
    3
    >>> sol[0].ode.as_sympy()
    Eq(u_0(t) + Derivative(u_0(t), (t, 2)), 0)
    """
    from asymptotics.core.exceptions import NoSmallParameterError

    eps        = problem.small_param
    t          = problem._indep_sym
    N          = order
    dep        = problem._dependent_name
    deriv_syms = problem._deriv_syms
    f          = problem.equation
    ptype      = problem.problem_type

    if eps not in f.free_symbols:
        raise NoSmallParameterError(eps, f)

    # Build u_k(t) functions
    u_funcs = [Function(f"{dep}_{k}")(t) for k in range(N + 1)]

    # Build ansatz
    u_ans = sum(eps**k * u_funcs[k] for k in range(N + 1))

    # Substitute ansatz
    dep_sym = problem.dependent
    f_sub   = f.subs(dep_sym, u_ans)
    for k_deriv, dsym in deriv_syms.items():
        f_sub = f_sub.subs(dsym, diff(u_ans, t, k_deriv))

    # Series expand
    f_series = series(f_sub, eps, 0, N + 1)

    # Extract coefficients
    coeffs = {k: f_series.coeff(eps, k) for k in range(N + 1)}

    # Build symbolic ODEs (with u_k functions as unknowns)
    # These are shown as-is — no substitution yet
    symbolic_odes = {}
    for k in range(N + 1):
        uk = u_funcs[k]
        # The symbolic ODE has u_k as unknown, lower orders as symbols
        # Just use the raw coefficient — it already has u_j(t) functions
        ode_expr = expand(coeffs[k])
        symbolic_odes[k] = Eq(ode_expr, 0)

    # Build hierarchy
    h = StepwiseHierarchy()
    h.small_param     = eps
    h.independent     = t
    h._problem        = problem
    h._problem_type   = ptype
    h._method         = f"Regular perturbation — ODE ({'IVP' if ptype == 'ivp' else 'BVP'})"
    h._u_funcs        = u_funcs
    h._n_orders       = N
    h._problem        = problem

    for k in range(N + 1):
        entry = StepwiseOrderEntry(
            order        = k,
            ode_symbolic = symbolic_odes[k],
            ode_coeffs   = coeffs[k],
            symbol       = u_funcs[k],
            hierarchy    = h,
        )
        h.entries.append(entry)

    return h
