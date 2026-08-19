"""
asymptotics.methods.regular_ode_system
======================================
Regular perturbation expansion for coupled ODE systems.

Algorithm
---------
At each order k, the equations for u_k^{(i)}(t) are DECOUPLED:
each depends only on lower-order solutions u_j^{(i)} for j < k.

This is because coupling terms like eps*v always involve
the other variable at a previous order after substitution.

So at each order we solve N independent ODEs, one per variable.
"""

from __future__ import annotations
from sympy import (
    Symbol, Function, symbols, series, expand,
    diff, dsolve, solve, Add, Integer, Eq, simplify,
    cos, exp
)
from asymptotics.core.exceptions import NoSmallParameterError


# ---------------------------------------------------------------------------
# Per-variable hierarchy entry
# ---------------------------------------------------------------------------

class ODESystemOrderEntry:
    r"""
    One order of the perturbation expansion for a single variable.

    Holds the order-:math:`k` ODE and its solution for one dependent
    variable of a coupled system.  Instances are produced by
    :func:`expand_regular_ode_system` and reached via
    ``sol["u"][k]`` (per-variable, per-order access); the same objects back
    the dicts returned by the order-view properties of
    :class:`_SystemOrderView`.

    Attributes
    ----------
    order : int
        The order :math:`k` (power of :math:`\varepsilon`) this entry
        represents.
    ode : sympy.Eq
        The order-:math:`k` ODE :math:`\mathcal{L}\,u_k = g_k = 0`, after
        substitution of all known lower-order solutions.
    equation : sympy.Eq
        Alias of :attr:`ode`, provided for a uniform per-order API shared
        with the scalar hierarchies.
    general_solution : sympy.Expr
        The general solution of :attr:`ode`, carrying free integration
        constants :math:`C_1, C_2, \dots`.
    particular_solution : sympy.Expr
        The solution with integration constants fixed by the conditions
        (the original conditions at :math:`k=0`, homogeneous conditions for
        :math:`k>0`).
    solution : sympy.Expr
        Alias of :attr:`particular_solution`.
    symbol : sympy.Function
        The undetermined function :math:`u_k(t)` this entry solves for.

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> import io, contextlib
    >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
    ...     sys = ODESystem(
    ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...         dependents  = ["u", "v"], independent = "t",
    ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
    >>> sol = sys.expand_regular(order=2)
    >>> entry = sol["v"][1]
    >>> entry.order
    1
    >>> entry.particular_solution
    -t*exp(-2*t)
    >>> entry.solution is entry.particular_solution
    True
    """

    def __init__(self, order, ode, general_solution, particular_solution, symbol):
        self.order               = order
        self.ode                 = ode
        self.general_solution    = general_solution
        self.particular_solution = particular_solution
        self.symbol              = symbol
        self.solution            = particular_solution   # alias
        self.equation            = ode   # uniform per-order API


class ODESystemVarHierarchy:
    r"""
    Perturbation hierarchy for ONE variable in a coupled system.

    Collects the :class:`ODESystemOrderEntry` objects for a single
    dependent variable and the assembled truncated series
    :math:`u(t,\varepsilon) = \sum_k \varepsilon^k u_k(t)`.  Mimics the
    scalar ``ODEHierarchy`` interface so the same display and access
    patterns work.  Obtained by indexing an :class:`ODESystemHierarchy`
    with a variable name: ``sol["u"]``.

    Attributes
    ----------
    name : str
        The dependent-variable name, e.g. ``"u"``.
    entries : list of ODESystemOrderEntry
        One entry per order, ``entries[k]`` being order :math:`k`.
    expansion : sympy.Expr
        The assembled truncated series in :math:`\varepsilon` for this
        variable (set once the solve completes).

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> import io, contextlib
    >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
    ...     sys = ODESystem(
    ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...         dependents  = ["u", "v"], independent = "t",
    ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
    >>> sol = sys.expand_regular(order=2)
    >>> uh = sol["u"]
    >>> len(uh)                                 # orders 0, 1, 2
    3
    >>> uh[0].particular_solution               # same as uh.__getitem__(0)
    exp(-t)
    >>> uh.expansion
    eps**2*(-t*exp(-2*t) + exp(-t) - exp(-2*t)) + eps*(-exp(-t) + exp(-2*t)) + exp(-t)
    """

    def __init__(self, name):
        self.name      = name
        self.entries   = []
        self.expansion = None

    def __getitem__(self, order: int):
        r"""
        Return the :class:`ODESystemOrderEntry` for a given order.

        Parameters
        ----------
        order : int
            The order :math:`k`; standard list indexing (negatives count
            from the end).

        Returns
        -------
        ODESystemOrderEntry
            The entry for order :math:`k`, exposing ``.equation``,
            ``.general_solution``, ``.particular_solution``, etc.
        """
        return self.entries[order]

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Top-level hierarchy container
# ---------------------------------------------------------------------------

class _SystemOrderView:
    r"""Order-:math:`k` view across every variable of a coupled system.

    Returned by ``sol[k]`` (integer key) on an :class:`ODESystemHierarchy`,
    giving the same ``.equation`` / ``.solution`` interface as a scalar
    order entry, except that each member is a **dict keyed by variable
    name**.  This is the transpose of per-variable access: ``sol[k]`` fixes
    the order and ranges over variables, whereas ``sol["u"][k]`` fixes the
    variable and picks the order.

    For an order-:math:`k` view :math:`\text{sol}[k]`,

    .. math::

        \text{sol}[k].\texttt{equation} =
            \bigl\{\, i : \mathcal{L}_i\,u^{(i)}_k = g^{(i)}_k = 0 \,\bigr\},
        \qquad
        \text{sol}[k].\texttt{solution} =
            \bigl\{\, i : u^{(i)}_k(t) \,\bigr\},

    with one key per dependent variable :math:`i`::

        sol[k].equation             # {'u': Eq(...), 'v': Eq(...)}
        sol[k].solution             # {'u': expr,    'v': expr}
        sol[k].particular_solution  # likewise

    (Per-variable access ``sol["u"][k]`` remains available and returns the
    single :class:`ODESystemOrderEntry` for that variable and order.)

    Parameters
    ----------
    hierarchy : ODESystemHierarchy
        The parent hierarchy this view reads from.
    order : int
        The fixed order :math:`k`.

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> import io, contextlib
    >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
    ...     sys = ODESystem(
    ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...         dependents  = ["u", "v"], independent = "t",
    ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
    >>> sol = sys.expand_regular(order=2)
    >>> view = sol[1]                           # order-1 view over all vars
    >>> sorted(view.solution)                   # dict keyed by variable
    ['u', 'v']
    >>> view.solution["v"]
    -t*exp(-2*t)
    >>> # equivalent single-variable access:
    >>> sol["v"][1].particular_solution
    -t*exp(-2*t)
    """

    def __init__(self, hierarchy, order):
        self.order = order
        self._h    = hierarchy

    def _collect(self, attr):
        return {v: getattr(self._h.hierarchies[v][self.order], attr)
                for v in self._h.variables}

    @property
    def equation(self):
        r"""dict — order-:math:`k` ODE :math:`\mathcal{L}_i u^{(i)}_k = 0`
        (a :class:`sympy.Eq`) for each variable, keyed by variable name."""
        return self._collect('equation')
    @property
    def ode(self):
        r"""dict — same as :attr:`equation`; the order-:math:`k` ODE keyed
        by variable name."""
        return self._collect('ode')
    @property
    def solution(self):
        r"""dict — order-:math:`k` particular solution
        :math:`u^{(i)}_k(t)` for each variable, keyed by variable name
        (alias of :attr:`particular_solution`)."""
        return self._collect('solution')
    @property
    def general_solution(self):
        r"""dict — order-:math:`k` general solution (with free integration
        constants) for each variable, keyed by variable name."""
        return self._collect('general_solution')
    @property
    def particular_solution(self):
        r"""dict — order-:math:`k` solution with constants fixed by the
        conditions, for each variable, keyed by variable name."""
        return self._collect('particular_solution')

    def __repr__(self):
        return (f"<order-{self.order} view over variables "
                f"{list(self._h.variables)}>")


class ODESystemHierarchy:
    r"""
    Result of a regular perturbation expansion for a coupled ODE system.

    Returned by :meth:`asymptotics.ODESystem.expand_regular` (via
    :func:`expand_regular_ode_system`).  Supports **dual indexing** through
    :meth:`__getitem__`:

    * a **string** key selects one variable's sub-hierarchy —
      ``sol["u"]`` is an :class:`ODESystemVarHierarchy`, so
      ``sol["u"][k]`` is the order-:math:`k` entry for ``u``;
    * an **integer** key selects an order across all variables —
      ``sol[k]`` is a :class:`_SystemOrderView` whose ``.equation`` /
      ``.solution`` / ``.particular_solution`` return dicts keyed by
      variable name.

    Thus ``sol["u"][k].particular_solution`` and
    ``sol[k].particular_solution["u"]`` refer to the same expression
    :math:`u_k(t)`.

    Attributes
    ----------
    variables : list of str
        The dependent-variable names, in order.
    hierarchies : dict
        Mapping ``{var_name: ODESystemVarHierarchy}``.
    small_param : sympy.Symbol
        The small parameter :math:`\varepsilon`.
    independent : sympy.Symbol
        The independent variable, e.g. :math:`t`.
    _method : str
        Human-readable label for the expansion method.

    See Also
    --------
    ODESystemVarHierarchy : Per-variable sub-hierarchy (``sol["u"]``).
    _SystemOrderView : Per-order cross-variable view (``sol[k]``).

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> import io, contextlib
    >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
    ...     sys = ODESystem(
    ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...         dependents  = ["u", "v"], independent = "t",
    ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
    >>> sol = sys.expand_regular(order=2)
    >>> sol.variables
    ['u', 'v']
    >>> len(sol)                                # number of orders
    3
    >>> sol["u"][0].particular_solution         # string key -> variable
    exp(-t)
    >>> sol[1].solution["v"]                    # int key -> order (dict)
    -t*exp(-2*t)
    """

    def __init__(self):
        self.variables   = []
        self.hierarchies = {}
        self.small_param = None
        self.independent = None
        self._method     = "Regular perturbation — ODE system"

    def __len__(self):
        """Number of orders (identical for every variable)."""
        first = next(iter(self.hierarchies.values()), None)
        return len(first) if first is not None else 0

    def __getitem__(self, key):
        r"""
        Dual-purpose indexing over variables and orders.

        Parameters
        ----------
        key : str or int
            * ``str`` — a variable name; returns that variable's
              :class:`ODESystemVarHierarchy` (e.g. ``sol["u"]``), which is
              itself indexable by order (``sol["u"][k]``).
            * ``int`` — an order :math:`k`; returns a
              :class:`_SystemOrderView` whose ``.equation`` / ``.solution``
              / ``.particular_solution`` are dicts keyed by variable name
              (e.g. ``sol[k].solution["u"]``).  The key is coerced with
              ``int(key)``.

        Returns
        -------
        ODESystemVarHierarchy or _SystemOrderView
            An :class:`ODESystemVarHierarchy` for a string key, or a
            :class:`_SystemOrderView` for an integer key.

        Raises
        ------
        KeyError
            If a string *key* is not one of :attr:`variables`.

        Examples
        --------
        >>> from asymptotics import ODESystem
        >>> import io, contextlib
        >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
        ...     sys = ODESystem(
        ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
        ...         dependents  = ["u", "v"], independent = "t",
        ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
        >>> sol = sys.expand_regular(order=2)
        >>> sol["u"][1].particular_solution         # variable, then order
        -exp(-t) + exp(-2*t)
        >>> sol[1].particular_solution["u"]         # order, then variable
        -exp(-t) + exp(-2*t)
        """
        # String key -> per-variable hierarchy:  sol["u"][k]
        # Integer key -> order-k view across all variables:  sol[k].solution -> {var: expr}
        if isinstance(key, str):
            if key not in self.hierarchies:
                raise KeyError(
                    f"\n\n  Variable '{key}' not in system.\n"
                    f"  Available: {self.variables}\n"
                )
            return self.hierarchies[key]
        return _SystemOrderView(self, int(key))


    def to_latex(self, environment='align', show_orders=False, filename=None):
        r"""
        Export this expansion as LaTeX source.

        Emits one aligned equation per dependent variable, with the small
        parameter rendered as ``\varepsilon``.

        Parameters
        ----------
        environment : str
            LaTeX math environment: ``'align'`` (default), ``'equation'``,
            or ``'gather'``.
        show_orders : bool
            If True, include each order :math:`u_k` separately.
            Default False.
        filename : str, optional
            If given, write the source to this file; otherwise return it.

        Returns
        -------
        str
            The LaTeX source string.

        Examples
        --------
        >>> from asymptotics import ODESystem
        >>> import io, contextlib
        >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
        ...     sys = ODESystem(
        ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
        ...         dependents  = ["u", "v"], independent = "t",
        ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
        >>> sol = sys.expand_regular(order=1)
        >>> with contextlib.redirect_stdout(io.StringIO()):   # to_latex also prints
        ...     src = sol.to_latex()
        >>> "varepsilon" in src
        True
        """
        from asymptotics.latex_export import to_latex
        return to_latex(self, environment=environment,
                        show_orders=show_orders, filename=filename)


    def eval(self, eps, at=None, params=None):
        r"""
        Numerically evaluate the truncated expansion of every variable.

        Substitutes a value of :math:`\varepsilon` into each variable's
        assembled series and evaluates it on a grid of the independent
        variable.  Because a system has several dependent variables, the
        result is keyed by variable name.

        Parameters
        ----------
        eps : float or list of float
            Value(s) of the small parameter :math:`\varepsilon`.
        at : array-like
            Grid of independent-variable values on which to sample each
            solution.
        params : dict, optional
            Values for any extra symbolic parameters in the expansion.

        Returns
        -------
        dict
            If *eps* is scalar, a dict ``{var_name: ndarray}`` giving each
            variable's samples on *at*.  If *eps* is a list, a nested dict
            ``{eps_value: {var_name: ndarray}}``.

        Examples
        --------
        >>> import numpy as np
        >>> from asymptotics import ODESystem
        >>> import io, contextlib
        >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
        ...     sys = ODESystem(
        ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
        ...         dependents  = ["u", "v"], independent = "t",
        ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
        >>> sol = sys.expand_regular(order=2)
        >>> out = sol.eval(eps=0.1, at=np.array([0.0, 1.0]))
        >>> sorted(out)
        ['u', 'v']
        >>> float(out["u"][0])                      # u(0, eps=0.1) = 1
        1.0
        >>> multi = sol.eval(eps=[0.1, 0.2], at=np.array([0.0, 1.0]))
        >>> sorted(multi)                           # keyed by eps, then var
        [0.1, 0.2]
        """
        from asymptotics.eval import eval_hierarchy
        return eval_hierarchy(self, eps, at=at, params=params)

    def show(self, mode: str = "auto") -> None:
        r"""
        Pretty-print the expansion, one block per variable.

        Renders rich LaTeX (:class:`IPython.display.Math`) in a Jupyter
        notebook and falls back to plain text in a terminal.  For each
        dependent variable it shows the order-by-order ODEs and solutions
        and the assembled truncated series in :math:`\varepsilon`.

        Parameters
        ----------
        mode : {'auto', 'latex', 'text'}, optional
            ``'auto'`` (default) picks LaTeX in Jupyter and text otherwise;
            ``'latex'`` and ``'text'`` force the respective renderer.

        Returns
        -------
        None
            Output is displayed/printed as a side effect.

        Examples
        --------
        >>> from asymptotics import ODESystem
        >>> import io, contextlib
        >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
        ...     sys = ODESystem(
        ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
        ...         dependents  = ["u", "v"], independent = "t",
        ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
        >>> sol = sys.expand_regular(order=1)
        >>> sol.show(mode="text")                   # doctest: +SKIP
        """
        from asymptotics.display.ode_system_display import show_ode_system
        show_ode_system(self, mode=mode)

    def compare_numeric(self, eps, params=None, **kwargs):
        r"""
        Compare this expansion against a direct numerical solution.

        Integrates the original coupled system numerically (via
        :func:`scipy.integrate.solve_ivp`) at the given :math:`\varepsilon`
        and compares it against the perturbation series, returning error
        norms and a comparison figure.  The original :class:`ODESystem` is
        recovered automatically from the stored ``_problem`` reference.

        Parameters
        ----------
        eps : float
            Value of the small parameter for the comparison.
        params : dict, optional
            Values for any extra symbolic parameters.
        plot_range : list of float, optional
            ``[a, b]`` interval of the independent variable to plot over.
        n_points : int, optional
            Number of sample points.

        Returns
        -------
        dict
            Contains ``'t'`` (grid), ``'u_pert'`` and ``'u_numerical'``
            (per-variable samples), ``'fig'`` (the Matplotlib figure),
            ``'errors'`` (L2/Linf absolute and relative errors keyed by the
            :math:`\varepsilon` value, per variable), and ``'settings'``
            (the SciPy solver configuration used).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from asymptotics import ODESystem
        >>> import io, contextlib
        >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
        ...     sys = ODESystem(
        ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
        ...         dependents  = ["u", "v"], independent = "t",
        ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
        >>> sol = sys.expand_regular(order=2)
        >>> result = sol.compare_numeric(eps=0.1)
        >>> sorted(result)
        ['errors', 'fig', 'settings', 't', 'u_numerical', 'u_pert']
        """
        from asymptotics.numerics import compare_numeric
        problem = getattr(self, '_problem', None)
        return compare_numeric(self, eps, params=params, problem=problem, **kwargs)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def expand_regular_ode_system(problem, order: int = 2) -> ODESystemHierarchy:
    r"""
    Apply regular perturbation to a coupled ODE system.

    This is the engine behind :meth:`asymptotics.ODESystem.expand_regular`.
    Given a problem with ansatz
    :math:`u^{(i)} = \sum_{k} \varepsilon^k u^{(i)}_k(t)`, it substitutes
    the series into every equation, expands in :math:`\varepsilon` to the
    requested order, and solves the resulting hierarchy order by order.

    At each order :math:`k` the known lower-order solutions are substituted
    into the collected coefficient, leaving — for a weakly coupled system —
    a scalar ODE for :math:`u^{(i)}_k` alone:

    .. math::

        \mathcal{L}_i\, u^{(i)}_k(t)
            = g^{(i)}_k\!\bigl(u^{(j)}_{<k}\bigr).

    The leading order applies the original conditions; higher orders use
    homogeneous conditions.  Before calling :func:`sympy.dsolve`, a
    **decoupling guard** checks that the order-:math:`k` equation for
    variable :math:`i` no longer contains any *other* variable's unknown
    :math:`u^{(j)}_m`; if it does, the system is coupled at :math:`O(1)`
    and a :class:`RuntimeError` is raised rather than returning an
    unresolved or incorrect result.

    Parameters
    ----------
    problem : ODESystem
        The parsed coupled-system problem.
    order : int, optional
        Highest power of :math:`\varepsilon` to compute. Default 2.

    Returns
    -------
    ODESystemHierarchy
        The solved hierarchy; each per-variable ``expansion`` and the
        ``_problem`` back-reference are populated.

    Raises
    ------
    NoSmallParameterError
        If :math:`\varepsilon` appears in none of the equations.
    RuntimeError
        If the decoupling guard finds an order-:math:`k` equation still
        coupled to another variable's unknown, or if :func:`sympy.dsolve`
        fails to solve an order-:math:`k` ODE.

    See Also
    --------
    asymptotics.ODESystem.expand_regular : Public method wrapper.

    Examples
    --------
    >>> from asymptotics import ODESystem
    >>> from asymptotics.methods.regular_ode_system import (
    ...     expand_regular_ode_system, ODESystemHierarchy)
    >>> import io, contextlib
    >>> with contextlib.redirect_stdout(io.StringIO()):  # hide inferred-var banner
    ...     sys = ODESystem(
    ...         equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    ...         dependents  = ["u", "v"], independent = "t",
    ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
    >>> sol = expand_regular_ode_system(sys, order=2)
    >>> isinstance(sol, ODESystemHierarchy)
    True
    >>> sol["v"][1].particular_solution
    -t*exp(-2*t)

    A system coupled at leading order is rejected:

    >>> with contextlib.redirect_stdout(io.StringIO()):  # v enters u's eq at O(1)
    ...     bad = ODESystem(
    ...         equations   = ["u' + u + v", "v' + 2*v + eps*u**2"],
    ...         dependents  = ["u", "v"], independent = "t",
    ...         small_param = "eps", conditions = ["u(0) = 1", "v(0) = 1"])
    >>> expand_regular_ode_system(bad, order=2)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError: ...
    """
    eps  = problem.small_param
    t    = problem.independent
    deps = problem.dependent_names
    N    = order
    C1, C2 = symbols('C1 C2')

    # Check eps appears in at least one equation
    has_eps = any(eps in problem.equations[dep].free_symbols for dep in deps)
    if not has_eps:
        raise NoSmallParameterError(eps, list(problem.equations.values())[0])

    # ------------------------------------------------------------------
    # Build u_k^{(i)}(t) Function objects for each variable and order
    # u_funcs[dep][k] = Function('dep_k')(t)
    # ------------------------------------------------------------------
    u_funcs = {
        dep: [Function(f"{dep}_{k}")(t) for k in range(N + 1)]
        for dep in deps
    }

    # ------------------------------------------------------------------
    # Build ansatz for each variable:
    # dep_ans[dep] = dep_0(t) + eps*dep_1(t) + eps^2*dep_2(t) + ...
    # ------------------------------------------------------------------
    u_ans = {
        dep: sum(eps**k * u_funcs[dep][k] for k in range(N + 1))
        for dep in deps
    }

    # ------------------------------------------------------------------
    # Substitute ansatz into each equation and expand in eps
    # ------------------------------------------------------------------
    # For each equation for dep i, substitute ALL dependent variables
    # and their derivatives with the ansatz
    coeffs = {dep: {} for dep in deps}

    for dep in deps:
        f_orig = problem.equations[dep]
        dep_syms   = problem._dep_syms
        deriv_syms = problem._deriv_syms

        f_sub = f_orig

        # Substitute each variable's ansatz
        for d in deps:
            f_sub = f_sub.subs(dep_syms[d], u_ans[d])
            # Substitute derivative symbols
            for k_deriv, dsym in deriv_syms[d].items():
                f_sub = f_sub.subs(dsym, diff(u_ans[d], t, k_deriv))

        # Series expand in eps
        f_series = series(f_sub, eps, 0, N + 1)
        for k in range(N + 1):
            coeffs[dep][k] = expand(f_series.coeff(eps, k))

    # ------------------------------------------------------------------
    # Solve order by order — variables are decoupled at each order
    # ------------------------------------------------------------------
    h = ODESystemHierarchy()
    h.variables   = deps
    h.small_param = eps
    h.independent = t
    h._method     = "Regular perturbation — ODE system"

    # Initialize per-variable hierarchies
    for dep in deps:
        h.hierarchies[dep] = ODESystemVarHierarchy(dep)

    # Track known solutions: {Function_k(t): expr}
    known = {}   # all variables all orders

    for k in range(N + 1):
        for dep in deps:
            uk = u_funcs[dep][k]

            # Substitute known lower-order solutions
            ode_expr = coeffs[dep][k]
            for func, sol_expr in known.items():
                ode_expr = ode_expr.subs(func, sol_expr)
            ode_expr = expand(ode_expr)

            # ------------------------------------------------------------------
            # Decoupling guard.  After substituting every known solution, the
            # equation for u_k must involve only u_k.  If it still references
            # another variable's unknown function, the variables are coupled at
            # this order (typically O(1) leading-order coupling), which violates
            # the decoupling assumption of this solver.  Fail loudly instead of
            # letting dsolve return an unresolved / incorrect result.
            # ------------------------------------------------------------------
            from sympy.core.function import AppliedUndef
            all_unknowns = {u_funcs[d][j] for d in deps for j in range(N + 1)}
            foreign = (ode_expr.atoms(AppliedUndef) & all_unknowns) - {uk}
            if foreign:
                foreign_names = ', '.join(sorted(str(f) for f in foreign))
                raise RuntimeError(
                    f"\n\n  The order-{k} equation for '{dep}' remains coupled to "
                    f"other variables ({foreign_names})\n"
                    f"  after substituting all known lower-order solutions:\n"
                    f"      {Eq(ode_expr, 0)}\n\n"
                    f"  expand_regular() for systems assumes the equations decouple "
                    f"into scalar ODEs\n"
                    f"  at each order, which holds only when the inter-variable "
                    f"coupling enters at\n"
                    f"  O({eps}) (i.e. the leading-order operator is diagonal).  This "
                    f"system is coupled\n"
                    f"  at order {k}; such strongly coupled systems are not supported "
                    f"by the automated\n"
                    f"  workflow.\n"
                )

            # Rewrite to exp form so dsolve can use variation of parameters /
            # undetermined coefficients cleanly.  Do NOT rewrite back to cos/sin
            # here — that converts exp(-t) → cosh(t)-sinh(t) which breaks
            # undetermined coefficients for terms like t*exp(-2t).
            from sympy import exp as _exp
            ode_expr = expand(ode_expr.rewrite(_exp))
            ode_eq   = Eq(ode_expr, 0)

            # Get the ODE order for this variable
            ode_order = problem.ode_orders[dep]

            # Solve
            try:
                gen_sol  = dsolve(ode_eq, uk)
                if isinstance(gen_sol, list):
                    gen_sol = gen_sol[0]
                gen_expr = gen_sol.rhs
            except Exception as e:
                raise RuntimeError(
                    f"\n\n  Could not solve order-{k} ODE for '{dep}':\n"
                    f"  {ode_eq}\n"
                    f"  Error: {e}\n"
                ) from e

            # Apply conditions
            conds_dep = problem.conditions.get(dep, [])

            # k=0: use original ICs; k>0: homogeneous ICs (value=0)
            free_consts = sorted(
                [s for s in gen_expr.free_symbols
                 if str(s).startswith('C') and str(s)[1:].isdigit()],
                key=lambda s: int(str(s)[1:])
            )

            ic_eqs = []
            for cond in conds_dep:
                val = cond.value if k == 0 else Integer(0)
                if cond.deriv_order == 0:
                    ic_eqs.append(Eq(gen_expr.subs(t, cond.point), val))
                else:
                    ic_eqs.append(Eq(
                        diff(gen_expr, t, cond.deriv_order).subs(t, cond.point),
                        val
                    ))

            try:
                const_sol = solve(ic_eqs, free_consts)
            except Exception:
                const_sol = {}

            if isinstance(const_sol, dict):
                part_expr = expand(gen_expr.subs(const_sol))
            else:
                part_expr = gen_expr

            known[uk] = part_expr

            entry = ODESystemOrderEntry(
                order               = k,
                ode                 = ode_eq,
                general_solution    = gen_expr,
                particular_solution = part_expr,
                symbol              = uk,
            )
            h.hierarchies[dep].entries.append(entry)

    # ------------------------------------------------------------------
    # Assemble expansions
    # ------------------------------------------------------------------
    for dep in deps:
        h.hierarchies[dep].expansion = Add(*[
            known[u_funcs[dep][k]] * eps**k for k in range(N + 1)
        ])

    h._problem = problem
    return h
