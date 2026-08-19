# Quickstart

Every workflow follows the same three steps: **(1)** build a problem object from
a string, **(2)** call an `expand_*` method to get an order-by-order hierarchy,
and **(3)** inspect, display, evaluate, or verify the result. Every hierarchy
exposes the same four-method API — `show()`, `eval()`, `compare_numeric()`,
`to_latex()` — and the same per-order indexing, `sol[k]`.

## Algebraic equations

```python
from asymptotics import AlgebraicEquation

eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
sol = eq.expand_regular(order=3)

sol.show()             # order-by-order solutions
sol.eval(eps=0.1)      # 0.9666...  (a float)
sol[1].solution        # the O(eps) correction
sol.compare_numeric(eps=0.3)   # plot vs scipy root-finding
```

## ODEs — regular perturbation

```python
from asymptotics import ODE

eq  = ODE("u' + u + eps*u**2", small_param="eps", conditions=["u(0) = 1"])
sol = eq.expand_regular(order=2)

sol[0].solution        # leading order, exp(-t)
sol[1].equation        # the O(eps) linear ODE (uniform accessor)
sol[1].secular         # False -> a naive expansion is valid here
```

## Nonlinear oscillators — Lindstedt–Poincaré

For problems with secular (resonant) terms, the strained-coordinate method
$\tau = \omega(\varepsilon)\, t$ removes them order by order:

```python
eq  = ODE("u'' + u + eps*u**3", small_param="eps",
          conditions=["u(0) = 1", "u'(0) = 0"])
sol = eq.expand_lindstedt(order=2)

sol.omega_expansion    # 1 + 3*eps/8 - 21*eps**2/256
sol[1].secular         # True at this order in a naive expansion
```

## Boundary layers — matched asymptotics

```python
eq  = ODE("eps*u'' + u' + u", small_param="eps", conditions=["u(0) = 0", "u(1) = 1"])
sol = eq.expand_boundary_layer()

parts   = sol.components   # dict of every substep, as live SymPy objects
outer   = sol.outer        # leading-order outer solution
inner   = sol.inner        # inner (boundary-layer) solution
comp    = sol.expansion    # additive composite
```

## Coupled systems

```python
from asymptotics import ODESystem

sys_ = ODESystem(["u' + v + eps*u", "v' - u + eps*v"], small_param="eps",
                 conditions=["u(0) = 1", "v(0) = 0"])
sol  = sys_.expand_regular(order=2)

sol[1].equation        # dict keyed by variable: {'u': ..., 'v': ...}
sol["u"][1].solution   # the O(eps) correction for u alone
```

## Step-by-step control

When an order cannot be solved in closed form, drop to the stepwise API and
solve that order however you like — including numerically:

```python
sol = eq.begin_expansion(order=2)
raw = sol[1].ode.as_sympy()    # a raw SymPy Eq you can manipulate or dsolve
sol[1].set_solution("...")     # provide a solution manually
sol.solve_all()                # attempt every remaining order
sol.n_solved, sol.n_pending
```

## Numerical verification

```python
result = sol.compare_numeric(eps=0.1)
result["errors"]     # L2 / Linf, absolute and relative
result["settings"]   # the SciPy solver, method, and tolerances used
```
