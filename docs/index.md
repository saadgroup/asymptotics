# asymptotics

**A symbolic perturbation-theory toolkit built on [SymPy](https://www.sympy.org/).**

Write your perturbation problem as a string, identify the small parameter, and
get symbolic order-by-order equations and solutions, LaTeX display, numerical
verification against SciPy, and direct numerical evaluation — for algebraic
equations, ODEs (regular, Lindstedt–Poincaré, multiple scales, matched
asymptotics), and coupled systems.

```python
from asymptotics import ODE

# Duffing oscillator: u'' + u + eps*u**3 = 0,  u(0)=1, u'(0)=0
eq  = ODE("u'' + u + eps*u**3", small_param="eps",
          conditions=["u(0) = 1", "u'(0) = 0"])
sol = eq.expand_lindstedt(order=2)

sol.omega_expansion          # -> 1 + 3*eps/8 - 21*eps**2/256
sol[1].secularity_condition  # the no-resonance condition that fixes omega_1
sol.compare_numeric(eps=0.1) # plot the expansion against scipy.solve_ivp
```

Install from PyPI:

```bash
pip install asymptotics
```

```{toctree}
:maxdepth: 2
:caption: Documentation

quickstart
api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
