"""
asymptotics — A perturbation theory toolkit built on SymPy.

Quick start
-----------
>>> from asymptotics import ODE, AlgebraicEquation
>>>
>>> # Algebraic equation
>>> eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
>>> sol = eq.expand_regular(order=3)
>>> sol.show()
>>>
>>> # ODE — IVP
>>> eq  = ODE("u'' + u + eps*u**3", dependent="u", small_param="eps", independent="t",
...           conditions=["u(0) = 1", "u'(0) = 0"])
>>> sol = eq.expand_lindstedt(order=2)
>>> sol.show()
>>>
>>> # Numeric comparison
>>> result = sol.compare_numeric(eps=0.1, problem=eq)
>>> result['fig']   # matplotlib Figure

Available methods
-----------------
AlgebraicEquation.expand_regular()
AlgebraicSystem.expand_regular()
ODE.expand_regular()          — regular perturbation (IVP + BVP)
ODE.expand_lindstedt()        — Lindstedt-Poincare (nonlinear oscillators)
ODE.expand_multiple_scales()  — method of multiple scales
ODE.expand_boundary_layer()   — matched asymptotic expansions (BVP)

All hierarchies support:
    sol.show()                           -- LaTeX display in Jupyter
    sol.compare_numeric(eps, problem=eq) -- numeric comparison + plot
"""

# ---------------------------------------------------------------------------
# Equation types
# ---------------------------------------------------------------------------
from asymptotics.core.problem import (
    PerturbationEquation,
    AlgebraicEquation,
    AlgebraicSystem,
    ODE,
)

# ---------------------------------------------------------------------------
# Hierarchy types
# ---------------------------------------------------------------------------
from asymptotics.core.hierarchy          import OrderHierarchy, OrderEntry
from asymptotics.core.system_hierarchy   import SystemHierarchy
from asymptotics.methods.regular_ode     import ODEHierarchy
from asymptotics.methods.lindstedt       import LindstedtHierarchy
from asymptotics.methods.multiple_scales import MultScalesHierarchy
from asymptotics.methods.boundary_layer  import BoundaryLayerHierarchy

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
from asymptotics.core.exceptions import (
    PerturbationError,
    NoSmallParameterError,
    NoLeadingOrderSolutionError,
    NoHigherOrderSolutionError,
    OnlyComplexRootsError,
)
from asymptotics.core.conditions import ConditionError

# ---------------------------------------------------------------------------
# Numeric comparison — callable standalone or via sol.compare_numeric()
# ---------------------------------------------------------------------------
from asymptotics.core.ode_system import ODESystem
from asymptotics.methods.regular_ode_system import ODESystemHierarchy
from asymptotics.numerics import compare_numeric
from asymptotics.eval import eval_hierarchy as eval
from asymptotics.latex_export import to_latex

# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------
from asymptotics.methods.regular_algebraic import expand_regular_algebraic as expand_regular

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.1.0"

__all__ = [
    # Equation types
    "PerturbationEquation",
    "AlgebraicEquation",
    "AlgebraicSystem",
    "ODE",
    # Hierarchy types
    "OrderHierarchy",
    "OrderEntry",
    "SystemHierarchy",
    "ODEHierarchy",
    "LindstedtHierarchy",
    "MultScalesHierarchy",
    "BoundaryLayerHierarchy",
    # Exceptions
    "PerturbationError",
    "NoSmallParameterError",
    "NoLeadingOrderSolutionError",
    "NoHigherOrderSolutionError",
    "OnlyComplexRootsError",
    "ConditionError",
    # Numeric comparison
    "ODESystem",
    "ODESystemHierarchy",
    "compare_numeric",
    "eval",
    "to_latex",
    # Convenience
    "expand_regular",
]
