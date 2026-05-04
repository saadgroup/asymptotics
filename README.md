<div align="center">

# asymptotics

**A symbolic perturbation theory toolkit for Python**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![SymPy](https://img.shields.io/badge/built%20on-SymPy-green)](https://www.sympy.org)
[![Tests](https://img.shields.io/badge/tests-164%20passing-brightgreen)](/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

*Write your perturbation problem as a string. Get symbolic results, LaTeX display, and numerical verification — automatically.*

</div>

---

## What is asymptotics?

`asymptotics` is a Python library that automates classical perturbation methods for algebraic equations and ODEs. Instead of manually deriving order-by-order equations, substituting ansätze, and solving each level — you write your problem as a plain string and call a method:

```python
from asymptotics import ODE

eq  = ODE("u'' + u + eps*u**3", small_param="eps",
          conditions=["u(0) = 1", "u'(0) = 0"])

sol = eq.expand_lindstedt(order=2)
sol.show()   # Beautiful LaTeX in Jupyter
```

The library handles the rest: symbolic expansion, secular term elimination, condition application, and optional numerical comparison.

---

## Installation

```bash
pip install asymptotics
```

Or for local development with notebooks:

```bash
git clone https://github.com/your-username/asymptotics
cd asymptotics
pip install -e ".[dev,notebook]"
```

**Requirements:** Python ≥ 3.10 · SymPy ≥ 1.12 · NumPy · SciPy · Matplotlib

---

## Methods at a glance

| Class | Method | Use when... |
|:------|:-------|:------------|
| `AlgebraicEquation` | `.expand_regular(order)` | Nonlinear algebraic equation $f(x, \varepsilon) = 0$ |
| `AlgebraicSystem` | `.expand_regular(order)` | Coupled algebraic system |
| `ODE` | `.expand_regular(order)` | ODE with small nonlinear term (IVP or BVP) |
| `ODE` | `.expand_lindstedt(order)` | Nonlinear oscillator — removes secular terms by straining time |
| `ODE` | `.expand_multiple_scales(order)` | Oscillator with slow amplitude/phase modulation |
| `ODE` | `.expand_boundary_layer()` | Singular BVP — $\varepsilon$ multiplies highest derivative |
| `ODESystem` | `.expand_regular(order)` | Coupled system of ODEs |

Every result supports:
- `sol.show()` — LaTeX display in Jupyter, plain text in terminal
- `sol.compare_numeric(eps, problem=eq)` — numerical verification + plot
- `sol[k].particular_solution` — symbolic result at order $k$

---

## Examples

### Algebraic equation

```python
from asymptotics import AlgebraicEquation

eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
sol = eq.expand_regular(order=3)
sol.show()

sol[0].solution   # x₀ = 1
sol[1].solution   # x₁ = -1/3
sol.composite     # 1 - ε/3 - ε³/81 + O(ε⁴)
```

### Coupled algebraic system

```python
from asymptotics import AlgebraicSystem

sys = AlgebraicSystem(
    equations   = ["x**2 + eps*sin(y) - 1", "y - eps*cos(x)"],
    dependents  = ["x", "y"],
    small_param = "eps",
)
sol = sys.expand_regular(order=3)
sol.show()

sol["x"].composite   # full expansion for x(ε)
sol["y"].composite   # full expansion for y(ε)
```

### Regular perturbation — IVP

```python
from asymptotics import ODE

# dependent='u' and independent='t' are inferred from the conditions
eq = ODE(
    "u'' + u + eps*u**3",
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_regular(order=2)
sol.show()

# Secular terms detected and flagged — use Lindstedt or multiple scales instead
sol[1].secular   # True — Duffing oscillator has secular terms at O(ε)
```

### Regular perturbation — BVP

```python
# BVP detected automatically from two distinct boundary points
# independent='x' inferred
eq = ODE(
    "u'' + eps*u",
    small_param = "eps",
    conditions  = ["u(0) = 0", "u(1) = 1"],
)
sol = eq.expand_regular(order=3)
sol.show()
```

### Lindstedt–Poincaré

Fix secular terms in nonlinear oscillators by straining the time coordinate
$\tau = \omega(\varepsilon)\,t$.

```python
eq = ODE(
    "u'' + u + eps*u**3",      # Duffing oscillator
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_lindstedt(order=2)
sol.show()

sol.omega_0          # ω₀ = 1   (auto-detected from unperturbed equation)
sol.omega_expansion  # ω(ε) = 1 + 3ε/8 - 21ε²/256
sol[1].omega_k_val   # ω₁ = 3/8
sol.composite_t      # u(t, ε) — uniformly valid for large t
```

Works for any natural frequency — detected automatically:

```python
eq = ODE("u'' + 4*u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"])
sol = eq.expand_lindstedt(order=2)
sol.omega_0   # 2 — detected from u'' + 4u = 0
```

### Multiple scales

For problems where amplitude or phase evolve slowly — e.g. weakly damped oscillators or limit cycles.

```python
# Weakly damped oscillator
eq = ODE(
    "u'' + u + eps*u'",
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_multiple_scales(order=1)
sol.show()

sol.amplitude_A   # A(T₁) = e^{-T₁/2}  — solved exactly by dsolve
sol.composite_t   # e^{-εt/2} · cos(t)  — matches exact solution at leading order
```

```python
# Van der Pol oscillator — limit cycle
eq = ODE(
    "u'' + u + eps*(u**2 - 1)*u'",
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_multiple_scales(order=1)
sol.amplitude_A   # 2√(eᵀ¹/(eᵀ¹+3))  → 2 as T₁→∞  (Bernoulli ODE, solved exactly)
```

### Boundary layers

For singular BVPs where $\varepsilon$ multiplies the highest derivative.
The layer location is detected automatically from the sign of $p(x)$.

```python
eq = ODE(
    "eps*u'' + u' + u",    # p(0)=1 > 0  →  layer at x=0
    small_param = "eps",
    conditions  = ["u(0) = 0", "u(1) = 1"],
)
sol = eq.expand_boundary_layer()
sol.show()

sol.layer_location   # 'x = 0'
sol.outer            # outer solution (away from layer)
sol.inner            # inner solution U(ξ) in stretched coord ξ = x/ε
sol.composite        # u_out + u_in − u_match  (Van Dyke rule)
```

Variable coefficients are fully supported:

```python
eq = ODE(
    "eps*u'' + (1+x)*u' - u",   # p(0)=1 > 0  →  layer at x=0
    small_param = "eps",
    conditions  = ["u(0) = 1", "u(1) = 2"],
)
```

### Coupled ODE system

```python
from asymptotics import ODESystem

sys = ODESystem(
    equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
    dependents  = ["u", "v"],
    small_param = "eps",
    independent = "t",
    conditions  = ["u(0) = 1", "v(0) = 1"],
)
sol = sys.expand_regular(order=2)
sol.show()

sol["u"].composite          # full expansion for u(t, ε)
sol["v"].composite          # full expansion for v(t, ε)
sol["u"][1].particular_solution   # u₁(t)
```

Works for any number of equations — 2, 3, or more.

### Numerical comparison

Every hierarchy includes `.compare_numeric()` for validation:

```python
sol = eq.expand_lindstedt(order=2)
result = sol.compare_numeric(eps=0.1, problem=eq)

result['fig']           # matplotlib Figure
result['t']             # evaluation points
result['u_pert']        # perturbation composite
result['u_numerical']   # exact numerical solution (scipy)
```

The plot range is inferred automatically from the problem's conditions —
a BVP with `u(0)=0, u(2)=1` will plot over `[0, 2]` without any extra arguments.
Override explicitly with `plot_range=[0, 20]`.

For boundary layers, all three pieces are shown and returned:

```python
result['u_outer']       # outer solution
result['u_inner']       # inner solution
result['u_composite']   # composite
```

For coupled systems, results are dicts keyed by variable name:

```python
result['u_pert']['u']   # perturbation for u
result['u_pert']['v']   # perturbation for v
```

---

## Accessing intermediate steps

Every hierarchy exposes the full symbolic work at each order:

```python
sol = eq.expand_regular(order=3)

sol[k].ode                   # the ODE at order k
sol[k].general_solution      # with free integration constants
sol[k].particular_solution   # constants fixed by BCs/ICs
sol[k].secular               # True if secular terms detected

sol.composite                # full assembled expansion u(t, ε)
sol.small_param              # the ε symbol
sol._problem_type            # 'ivp' or 'bvp'
```

Lindstedt-specific:

```python
sol.omega_0                  # unperturbed frequency
sol.omega_expansion          # ω(ε) series
sol[k].omega_k_val           # frequency correction ωₖ
sol[k].secularity_condition  # the equation that determined ωₖ
sol.composite_t              # u(t, ε) with τ = ω(ε)·t substituted
```

Multiple scales-specific:

```python
sol.T0, sol.T1               # fast and slow time symbols
sol.amplitude_A              # A(T₁) — solved if possible
sol.amplitude_B              # B(T₁)
sol[k].solvability_A         # the amplitude ODE dA/dT₁ = ...
```

---

## Display

`sol.show()` renders LaTeX in Jupyter and clean text in the terminal.
The small parameter is always displayed as $\varepsilon$, regardless of what you named it (`eps`, `epsilon`, `mu`, etc.).

```python
sol.show()                    # full hierarchy
sol.show(orders=[0, 1])       # selected orders only
sol.show(mode='text')         # force plain text (e.g. in scripts)
```

---

## Condition syntax

Conditions are plain strings — same notation you'd write on paper:

```python
conditions = ["u(0) = 1"]                          # 1st-order IVP
conditions = ["u(0) = 1", "u'(0) = 0"]             # 2nd-order IVP
conditions = ["u(0) = 0", "u(1) = 1"]              # BVP
conditions = ["u(pi) = 0", "u'(0) = sqrt(2)"]      # symbolic points and values
conditions = ["u(0) = 1/2", "u'(0) = 0"]           # rational values
conditions = ["u(0) = 0.9", "u'(0) = 0"]           # floats auto-converted to rationals
```

`asymptotics` automatically:
- Detects IVP vs BVP from the number of distinct boundary points
- Infers the dependent variable name (`u`) from the condition strings
- Infers the independent variable (`t` for IVPs, `x` for BVPs) from the equation or defaults
- Catches wrong count, conflicting, or inconsistent conditions with clear error messages

---

## Auto-inference of dependent and independent variables

For `ODE`, both `dependent` and `independent` are optional:

```python
# Fully minimal — everything inferred
eq = ODE("u'' + u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"])
# ℹ️  dependent = 'u' (inferred from conditions), independent = 't' (inferred from equation)

# Override when needed
eq = ODE("u'' + sin(tau)*u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"],
         independent = "tau")
```

Independent variable inference looks for `{x, y, z, t}` in the equation.
Everything else (`a`, `b`, `lambda`, `q`, ...) is treated as a parameter.

---

## Error messages

`asymptotics` raises clear, actionable errors:

```
ConditionError: 3 conditions provided but the ODE is order 2 — need exactly 2.
  IVP: ["u(0) = 1", "u'(0) = 0"]
  BVP: ["u(0) = 0", "u(1) = 1"]

ConditionError: Conflicting conditions at t=0:
  u(0) = 1
  u(0) = 2

TypeError: 'expand_regular' order must be an integer, got str: 'two'
  Example: eq.expand_regular(order=2)

ValueError: Lindstedt-Poincare requires an oscillatory problem (omega_0^2 > 0).
  Suggestions: use expand_regular() for non-oscillatory problems,
  or expand_multiple_scales() for damped oscillators.
```

---

## Project layout

```
asymptotics/
├── __init__.py                    ← public API
├── numerics.py                    ← compare_numeric() and numerical solvers
├── core/
│   ├── problem.py                 ← AlgebraicEquation, AlgebraicSystem, ODE, ODESystem
│   ├── ode_system.py              ← ODESystem class
│   ├── hierarchy.py               ← OrderHierarchy, OrderEntry
│   ├── system_hierarchy.py        ← SystemHierarchy (coupled algebraic)
│   ├── conditions.py              ← condition parser and validator
│   └── exceptions.py              ← custom exceptions
├── methods/
│   ├── regular_algebraic.py       ← AlgebraicEquation solver
│   ├── regular_algebraic_system.py← AlgebraicSystem solver
│   ├── regular_ode.py             ← ODE regular perturbation
│   ├── regular_ode_system.py      ← ODESystem solver
│   ├── lindstedt.py               ← Lindstedt–Poincaré
│   ├── multiple_scales.py         ← Multiple scales
│   └── boundary_layer.py          ← Matched asymptotic expansions
├── display/
│   ├── jupyter.py                 ← Algebraic LaTeX display
│   ├── ode_display.py
│   ├── ode_system_display.py
│   ├── lindstedt_display.py
│   ├── multiple_scales_display.py
│   └── boundary_layer_display.py
└── tests/                         ← 164 tests, all passing
    ├── test_regular_algebraic.py
    ├── test_algebraic_system.py
    ├── test_errors.py
    ├── test_ode.py
    ├── test_lindstedt.py
    ├── test_multiple_scales.py
    ├── test_boundary_layer.py
    └── test_ode_system.py
```

---

## Notebooks

Seven Jupyter notebooks covering every method with worked examples and plots:

| Notebook | Topic |
|:---------|:------|
| `01_introduction.ipynb` | Algebraic equations — the basics |
| `02_transcendental.ipynb` | Transcendental equations and coupled algebraic systems |
| `03_ode.ipynb` | Regular perturbation for ODEs — IVP and BVP |
| `04_lindstedt.ipynb` | Lindstedt–Poincaré — nonlinear oscillators |
| `05_multiple_scales.ipynb` | Multiple scales — damped oscillators, Van der Pol |
| `06_boundary_layers.ipynb` | Matched asymptotic expansions |
| `07_ode_system.ipynb` | Coupled ODE systems |

---

## Running tests

```bash
pytest                     # all 164 tests
pytest -v                  # verbose output
pytest --cov=asymptotics   # with coverage report
```

---

## Design philosophy

- **String-based input** — write `"u'' + u + eps*u**3"` not `u.diff(t,2) + u + eps*u**3`
- **Inspect everything** — every intermediate step is a symbolic SymPy expression
- **Fail clearly** — errors tell you what went wrong and how to fix it
- **Consistent API** — `expand_*`, `show()`, `compare_numeric()` work the same everywhere
- **No SymPy wrestling** — the library handles all the symbolic machinery internally

---

## License

MIT © 2025

