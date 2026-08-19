"""
asymptotics.core.exceptions
===========================
Custom exceptions for clear, actionable error messages.
"""


class PerturbationError(Exception):
    """
    Base class for all ``asymptotics`` errors.

    Every domain-specific exception raised by the toolkit inherits from this
    class, so ``except PerturbationError`` catches any expected failure of an
    expansion (missing small parameter, unsolvable order, complex-only roots,
    ...) while letting unrelated Python errors propagate.

    Notes
    -----
    This base class is raised directly only in rare, otherwise-uncategorised
    cases; ordinarily one of the more specific subclasses below is raised.
    """
    pass


class NoSmallParameterError(PerturbationError):
    """
    The declared small parameter does not appear in the equation.

    Raised at the start of an expansion when the symbol passed as
    ``small_param`` is absent from the equation — usually because the parameter
    was forgotten in the equation string, or the wrong symbol was named as the
    small parameter.  Without the parameter there is nothing to expand in.

    Parameters
    ----------
    param : sympy.Symbol or str
        The small parameter that was expected but not found.  Stored on the
        exception as ``.param``.
    equation : sympy.Expr
        The (residual) equation ``f`` that was inspected.  Stored as
        ``.equation``.
    """
    def __init__(self, param, equation):
        self.param    = param
        self.equation = equation
        super().__init__(
            f"\n\n  The small parameter '{param}' does not appear in the equation:\n"
            f"    f = {equation}\n\n"
            f"  Did you forget to include '{param}' in your equation?\n"
            f"  Example:  AlgebraicEquation(x**3 + eps*x - 1, dependent=x, small_param=eps)\n"
        )


class NoLeadingOrderSolutionError(PerturbationError):
    r"""
    The leading-order :math:`\mathcal{O}(1)` equation has no closed-form root.

    Raised when SymPy cannot solve the unperturbed (``eps = 0``) equation
    symbolically.  This is typical of transcendental problems such as
    ``x - cos(x) = eps``, whose leading-order root cannot be written in closed
    form.  The remedy is to supply a numerical ``root_hint``, rearrange the
    equation so the leading-order part is solvable, or check that the small
    parameter and unknown were identified correctly.

    Parameters
    ----------
    equation : sympy.Expr
        The leading-order equation (set to zero) that could not be solved.
        Stored on the exception as ``.equation``.
    param : sympy.Symbol or str
        The small parameter, used only to build the guidance message.
    variable : sympy.Symbol or str
        The unknown being solved for, used only in the guidance message.
    """
    def __init__(self, equation, param, variable):
        self.equation = equation
        super().__init__(
            f"\n\n  Could not solve the leading-order O(1) equation:\n"
            f"    {equation} = 0\n\n"
            f"  This happens when SymPy cannot find a closed-form root.\n"
            f"  Possible fixes:\n"
            f"    1. Provide a numerical root hint:\n"
            f"       AlgebraicEquation(..., root_hint=0.739)  # approximate value\n"
            f"    2. Rearrange your equation so the O(1) part is solvable.\n"
            f"    3. Check that '{param}' is actually the small parameter\n"
            f"       and '{variable}' is the unknown.\n"
        )


class NoHigherOrderSolutionError(PerturbationError):
    r"""
    The order-:math:`k` equation (:math:`k \ge 1`) could not be solved.

    Raised while marching up the hierarchy when SymPy cannot solve the equation
    at some order :math:`k` for its unknown term.  In a well-posed regular
    expansion each order is *linear* in its highest unknown; failure here
    usually signals that the equation is nonlinear in that unknown, that the
    problem is actually singular (needs Lindstedt, multiple scales, or matched
    asymptotics instead of regular perturbation), or that the wrong
    leading-order branch was chosen.

    Parameters
    ----------
    order : int
        The order :math:`k` at which solving failed.  Stored as ``.order``.
    symbol : sympy.Symbol or list of sympy.Symbol
        The order-:math:`k` unknown(s) being solved for (e.g. :math:`x_k`).
    equation : sympy.Expr
        The order-:math:`k` equation (set to zero) that could not be solved.
        Stored as ``.equation``.
    known : dict
        The already-determined lower-order solutions available at this point,
        included in the message to aid debugging.
    """
    def __init__(self, order, symbol, equation, known):
        self.order    = order
        self.equation = equation
        super().__init__(
            f"\n\n  Could not solve the order-{order} equation for {symbol}:\n"
            f"    {equation} = 0\n\n"
            f"  This may mean:\n"
            f"    1. The regular perturbation expansion breaks down at this order.\n"
            f"       (The equation may be nonlinear in {symbol}.)\n"
            f"    2. This is a singular perturbation problem — try a different method.\n"
            f"    3. The leading-order solution chosen is not the right branch.\n"
            f"       Try a different root_hint or root_index.\n"
            f"\n"
            f"  Known values at this point: {known}\n"
        )


class OnlyComplexRootsError(PerturbationError):
    r"""
    The leading-order equation has only complex roots and none was selected.

    Raised when the unperturbed :math:`\mathcal{O}(1)` equation is solvable but
    every root is complex (e.g. ``x**2 + 1 + eps*x`` gives ``x0 = ±i``), so
    there is no real branch to expand about.  To follow a particular complex
    branch, pass a ``root_hint`` near the desired root.

    Parameters
    ----------
    equation : sympy.Expr
        The leading-order equation (set to zero) whose roots are all complex.
    roots : list of sympy.Expr
        The complex roots that were found.  Stored on the exception as
        ``.roots`` (and shown in the message).
    """
    def __init__(self, equation, roots):
        self.roots = roots
        super().__init__(
            f"\n\n  The leading-order equation:\n"
            f"    {equation} = 0\n"
            f"  has only complex roots: {roots}\n\n"
            f"  If you want to follow a complex branch, provide a root_hint:\n"
            f"    AlgebraicEquation(..., root_hint=-0.5 + 0.866j)\n"
        )


class NotReadyError(PerturbationError, RuntimeError):
    r"""
    A result method was called on a step-by-step hierarchy before every order
    was solved.

    Raised by the four-method result interface of a
    :class:`~asymptotics.StepwiseHierarchy` — :meth:`~StepwiseHierarchy.show`,
    :meth:`~StepwiseHierarchy.eval`, :meth:`~StepwiseHierarchy.compare_numeric`,
    :meth:`~StepwiseHierarchy.to_latex` — while one or more orders are still
    unsolved.  The message lists the solved and pending orders and how to
    finish them (:meth:`~StepwiseOrderEntry.solve`,
    :meth:`~StepwiseOrderEntry.set_solution`, or
    :meth:`~StepwiseHierarchy.solve_all`).

    It subclasses both :class:`PerturbationError` and :class:`RuntimeError`, so
    it is caught by ``except PerturbationError`` (the toolkit-wide handler) and
    by ``except RuntimeError`` (backward compatible with earlier releases that
    raised a bare :class:`RuntimeError`).

    Parameters
    ----------
    method_name : str
        The result method that was called too early (e.g. ``"to_latex"``).
        Stored on the exception as ``.method_name``.
    solved : list of int
        Orders already solved.  Stored as ``.solved``.
    pending : list of int
        Orders still unsolved.  Stored as ``.pending``.
    """
    def __init__(self, method_name, solved, pending):
        self.method_name = method_name
        self.solved      = list(solved)
        self.pending     = list(pending)
        steps = "\n".join(
            f"    sol[{k}].solve()   or   sol[{k}].set_solution(expr)"
            for k in self.pending
        )
        super().__init__(
            f"\n\n  Cannot call '{method_name}' — not all orders are solved.\n"
            f"  Solved:  {self.solved}\n"
            f"  Pending: {self.pending}\n\n"
            f"  Solve remaining orders:\n{steps}\n"
            f"  Or: sol.solve_all()\n"
        )
