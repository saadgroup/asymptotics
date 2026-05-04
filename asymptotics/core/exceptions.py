"""
asymptotics.core.exceptions
=======================
Custom exceptions for clear, actionable error messages.
"""


class PerturbationError(Exception):
    """Base class for all asymptotics errors."""
    pass


class NoSmallParameterError(PerturbationError):
    """
    Raised when the equation does not contain the declared small parameter.

    This usually means the user forgot to include eps in the equation,
    or passed the wrong symbol as small_param.
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
    """
    Raised when SymPy cannot solve the O(1) equation symbolically.

    This happens with transcendental equations like x = cos(x) where
    the unperturbed root cannot be expressed in closed form.
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
    """
    Raised when SymPy cannot solve the equation at order k.

    This can happen if the higher-order equation is nonlinear in x_k
    (which would indicate a singular perturbation problem or a
    breakdown of the regular expansion).
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
    """
    Raised when the O(1) equation has only complex roots and no root_hint
    is provided to select one.
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
