"""
asymptotics — A symbolic perturbation theory toolkit built on SymPy.

Write your perturbation problem as a string. Get symbolic order-by-order
results, LaTeX display, numerical verification, and direct evaluation.

Quick start
-----------
>>> from asymptotics import ODE, AlgebraicEquation

>>> # Algebraic equation
>>> eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
>>> sol = eq.expand_regular(order=3)
>>> sol.show()
>>> sol.eval(eps=0.1)                            # float
>>> sol.compare_numeric(eps=0.3)                 # plot vs scipy.fsolve

>>> # ODE — IVP (dependent and independent inferred from conditions)
>>> eq  = ODE("u'' + u + eps*u**3", small_param="eps",
...           conditions=["u(0) = 1", "u'(0) = 0"])
>>> sol = eq.expand_lindstedt(order=2)
>>> sol.show()
>>> sol.omega_expansion                          # ω(ε) = 1 + 3ε/8 - 21ε²/256

>>> import numpy as np
>>> sol.eval(eps=0.1, at=np.linspace(0, 40, 500))  # ndarray
>>> sol.compare_numeric(eps=0.1)                 # plot vs scipy.solve_ivp
>>> sol.to_latex(filename="duffing.tex")         # export LaTeX source

Available problem classes
-------------------------
AlgebraicEquation   — single nonlinear algebraic equation  f(x, ε) = 0
AlgebraicSystem     — coupled algebraic system
ODE                 — ordinary differential equation (IVP or BVP, orders 1–6)
ODESystem           — coupled system of ODEs

Expansion methods
-----------------
AlgebraicEquation.expand_regular(order, root_index=0, gauge=None)
AlgebraicSystem.expand_regular(order)
ODE.expand_regular(order, gauge=None)        — regular perturbation (IVP + BVP)
ODE.expand_lindstedt(order)                  — Lindstedt–Poincaré (nonlinear oscillators)
ODE.expand_multiple_scales(order)            — method of multiple scales
ODE.expand_boundary_layer()                  — matched asymptotic expansions
ODE.begin_expansion(order)                   — step-by-step control (StepwiseHierarchy)
ODESystem.expand_regular(order)              — coupled ODE systems

Four-method API on every hierarchy
-----------------------------------
sol.show(orders=None, mode='auto')
    LaTeX in Jupyter; plain text in terminal.

sol.eval(eps, at=None, params=None)
    Evaluate composite as a NumPy array (or float for algebraic).
    Accepts scalar or list for eps.

sol.compare_numeric(eps, params=None, plot_range=None, filename=None)
    Numerical verification via SciPy + comparison plot.
    Returns dict with 't', 'u_pert', 'u_numerical', 'fig' (ODE types).

sol.to_latex(environment='align', show_orders=False, filename=None)
    Export results as LaTeX source.
    Small parameter always rendered as \\varepsilon.

Per-order access
----------------
sol[k].ode                   — ODE/equation at order k
sol[k].general_solution      — with free integration constants
sol[k].particular_solution   — constants fixed by ICs/BCs
sol[k].secular               — True if secular terms detected (ODE only)
sol.composite                — assembled SymPy expansion

Lindstedt extras: sol.omega_0, sol.omega_expansion, sol.composite_t,
                  sol[k].omega_k_val, sol[k].omega_k_sym, sol[k].secularity_condition
Multiple-scales extras: sol.T0, sol.T1, sol.amplitude_A, sol.amplitude_B,
                        sol[k].solvability_A, sol[k].solvability_B,
                        sol[k].pde  (underlying PDE; sol[k].ode is an alias)

Step-by-step API (StepwiseHierarchy)
--------------------------------------
sol[k].solve()               — attempt SymPy solution (fails gracefully)
sol[k].set_solution(expr)    — provide solution manually (str or SymPy)
sol.solve_all()              — attempt all remaining orders
sol[k].is_solved             — bool
sol.n_solved / sol.n_pending — int

Gauge sequences
---------------
Standard gauge {1, ε, ε², …} is used by default.
Override:
    eq.expand_regular(order=3, gauge="sqrt(eps)")          # geometric
    eq.expand_regular(order=2, gauge=["1","log(eps)","log(eps)**2"])  # explicit
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
# ODE system
# ---------------------------------------------------------------------------
from asymptotics.core.ode_system import ODESystem
from asymptotics.methods.regular_ode_system import ODESystemHierarchy

# ---------------------------------------------------------------------------
# Step-by-step API
# ---------------------------------------------------------------------------
from asymptotics.methods.stepwise import StepwiseHierarchy

# ---------------------------------------------------------------------------
# Standalone functions (also available as sol.method() on every hierarchy)
# ---------------------------------------------------------------------------
from asymptotics.numerics    import compare_numeric
from asymptotics.eval        import eval_hierarchy as eval
from asymptotics.latex_export import to_latex

# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------
from asymptotics.methods.regular_algebraic import expand_regular_algebraic as expand_regular

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.2.0"

__all__ = [
    # Problem classes
    "PerturbationEquation",
    "AlgebraicEquation",
    "AlgebraicSystem",
    "ODE",
    "ODESystem",
    # Hierarchy types
    "OrderHierarchy",
    "OrderEntry",
    "SystemHierarchy",
    "ODEHierarchy",
    "LindstedtHierarchy",
    "MultScalesHierarchy",
    "BoundaryLayerHierarchy",
    "ODESystemHierarchy",
    "StepwiseHierarchy",
    # Exceptions
    "PerturbationError",
    "NoSmallParameterError",
    "NoLeadingOrderSolutionError",
    "NoHigherOrderSolutionError",
    "OnlyComplexRootsError",
    "ConditionError",
    # Standalone functions
    "compare_numeric",
    "eval",
    "to_latex",
    # Convenience
    "expand_regular",
]
