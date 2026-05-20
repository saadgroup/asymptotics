<div align="center">

# asymptotics

**A symbolic perturbation theory toolkit for Python**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![SymPy](https://img.shields.io/badge/built%20on-SymPy-green)](https://www.sympy.org)
[![Tests](https://img.shields.io/badge/tests-195%20passing-brightgreen)](/asymptotics/tests)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

*Write your perturbation problem as a string. Get symbolic order-by-order results,  
LaTeX display, numerical verification, and direct evaluation — automatically.*

</div>

---

## What is asymptotics?

`asymptotics` is a Python library that automates classical perturbation methods for
algebraic equations and ordinary differential equations (ODEs). Instead of manually
deriving order-by-order equations, substituting ansätze, and solving each level —
you write your problem as a plain string and call a method:

```python
from asymptotics import ODE

eq  = ODE("u'' + u + eps*u**3", small_param="eps",
          conditions=["u(0) = 1", "u'(0) = 0"])

sol = eq.expand_lindstedt(order=2)
sol.show()       # beautiful LaTeX in Jupyter; clean text in terminal
```

The library handles the rest: symbolic expansion, secular term elimination,
condition application, and optional numerical comparison against SciPy.

Unlike black-box tools (Mathematica's `AsymptoticDSolveValue`), `asymptotics`
exposes every intermediate step — each order's ODE, its general and particular
solution, secular terms, frequency corrections — as live SymPy expressions you
can inspect and manipulate.

---

## Installation

```bash
pip install asymptotics
```

For local development with example notebooks:

```bash
git clone https://github.com/saadgroup/asymptotics
cd asymptotics
pip install -e ".[dev,notebook]"
```

**Requirements:** Python ≥ 3.10 · SymPy ≥ 1.12 · NumPy ≥ 1.24 · SciPy · Matplotlib

---

## Methods at a glance

| Class | Method | Use when… |
|:------|:-------|:----------|
| `AlgebraicEquation` | `.expand_regular(order)` | Nonlinear algebraic equation $f(x,\varepsilon)=0$ |
| `AlgebraicSystem` | `.expand_regular(order)` | Coupled algebraic system |
| `ODE` | `.expand_regular(order)` | ODE with small nonlinear/forcing term (IVP or BVP) |
| `ODE` | `.expand_lindstedt(order)` | Nonlinear oscillator — strains time to remove secular terms |
| `ODE` | `.expand_multiple_scales(order)` | Oscillator with slow amplitude/phase modulation |
| `ODE` | `.expand_boundary_layer()` | Singular BVP — $\varepsilon$ multiplies highest derivative |
| `ODE` | `.begin_expansion(order)` | Step-by-step control when SymPy cannot solve a leading order |
| `ODESystem` | `.expand_regular(order)` | Coupled system of ODEs |

Every result supports a consistent four-method API:

| Method | What it does |
|:-------|:-------------|
| `sol.show()` | LaTeX display in Jupyter; plain text in terminal |
| `sol.eval(eps, at=...)` | Evaluate composite as a NumPy array |
| `sol.compare_numeric(eps)` | Numerical verification plot via SciPy |
| `sol.to_latex(...)` | Export ready-to-paste LaTeX source |

---

## Quick examples

### Algebraic equation

```python
from asymptotics import AlgebraicEquation

eq  = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
sol = eq.expand_regular(order=3)
sol.show()
# x(ε) = 1 - ε/3 - ε³/81 + O(ε⁴)

sol[0].solution    # x₀ = 1
sol[1].solution    # x₁ = -1/3
sol.composite      # full SymPy expression

sol.eval(eps=0.1)                          # float
sol.eval(eps=[0.1, 0.2, 0.3])             # ndarray
sol.compare_numeric(eps=0.3)              # plot vs scipy.fsolve
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

sol["x"].composite   # expansion for x(ε)
sol["y"].composite   # expansion for y(ε)
```

### Regular perturbation — IVP

```python
from asymptotics import ODE

# dependent='u' and independent='t' inferred automatically from conditions
eq = ODE(
    "u'' + u + eps*u**3",
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_regular(order=2)
sol.show()

sol[1].secular   # True — secular terms detected; use Lindstedt or multiple scales
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

sol.omega_0           # ω₀ = 1   (auto-detected from unperturbed equation)
sol.omega_expansion   # ω(ε) = 1 + 3ε/8 - 21ε²/256
sol[1].omega_k_val    # ω₁ = 3/8  (classical result)
sol.composite_t       # u(t, ε) — uniformly valid in physical time t

import numpy as np
t_vals = np.linspace(0, 40, 500)
sol.eval(eps=0.1, at=t_vals)        # ndarray
sol.compare_numeric(eps=0.1)        # plot vs scipy.solve_ivp
```

Works for any natural frequency — detected automatically from the unperturbed equation:

```python
eq = ODE("u'' + 4*u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"])
sol = eq.expand_lindstedt(order=2)
sol.omega_0   # 2 — detected from u'' + 4u = 0
```

### Multiple scales

For problems where amplitude or phase evolve slowly.

```python
# Weakly damped oscillator
eq = ODE(
    "u'' + u + eps*u'",
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_multiple_scales(order=1)
sol.show()

sol.amplitude_A   # A(T₁) = e^{-T₁/2}  — exact Bernoulli solution
sol.composite_t   # e^{-εt/2} · cos(t)  — matches exact result at O(1)
```

```python
# Van der Pol oscillator — limit cycle
eq = ODE(
    "u'' + u + eps*(u**2 - 1)*u'",
    small_param = "eps",
    conditions  = ["u(0) = 1", "u'(0) = 0"],
)
sol = eq.expand_multiple_scales(order=1)
sol.amplitude_A   # 2√(eᵀ¹/(eᵀ¹+3))  →  2  as T₁→∞  (limit cycle)
```

### Boundary layers

For singular BVPs where $\varepsilon$ multiplies the highest derivative.
The layer location is detected automatically from the sign of $p(x)$.

```python
eq = ODE(
    "eps*u'' + u' + u",   # p(0) = 1 > 0  →  layer at x = 0
    small_param = "eps",
    conditions  = ["u(0) = 0", "u(1) = 1"],
)
sol = eq.expand_boundary_layer()
sol.show()

sol.layer_location   # 'x = 0'
sol.outer            # outer solution (away from layer)
sol.inner            # inner solution U(ξ) in stretched coord ξ = x/ε
sol.composite        # u_out + u_in − u_match  (Van Dyke rule)

sol.compare_numeric(eps=0.05)
```

Variable coefficients are fully supported:

```python
eq = ODE(
    "eps*u'' + (1 + x)*u' - u",
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

sol["u"].composite              # full expansion for u(t, ε)
sol["v"].composite              # full expansion for v(t, ε)
sol["u"][1].particular_solution # u₁(t)

import numpy as np
t_vals = np.linspace(0, 10, 200)
sol.eval(eps=0.1, at=t_vals)    # {'u': ndarray, 'v': ndarray}
sol.compare_numeric(eps=0.1)
```

Works for any number of coupled equations.

---

## Step-by-step API

When SymPy cannot solve a leading-order equation (e.g. a nonlinear ODE at O(1)),
`begin_expansion()` gives you full manual control while letting the library handle
all linear higher-order equations automatically.

```python
sol = eq.begin_expansion(order=2)

# Inspect all equations immediately — nothing is solved yet
sol.show()

# Examine order-k ODE
sol[0].ode    # O(1) equation — purely symbolic
sol[1].ode    # O(ε) equation — symbolic if O(1) unsolved;
              #                  fully substituted if O(1) is solved

# Solve or provide solutions
sol[0].solve()                      # try SymPy — fails gracefully with clear message
sol[0].set_solution("4*eta*(1-eta)") # provide expression as string or SymPy expr
sol[1].solve()                      # SymPy handles linear higher orders
sol.solve_all()                     # attempt all remaining

# Inspect state
sol[k].is_solved                    # bool
sol[k].particular_solution          # SymPy expression (if solved)
sol[k].general_solution             # SymPy expression (if solved)
sol.n_solved                        # number of solved orders
sol.n_pending                       # number of unsolved orders

# Full standard API once all orders are solved
sol.composite
sol.show()
sol.to_latex()
sol.eval(eps=0.1, at=t_vals)
sol.compare_numeric(eps=0.1)
```

If `composite`, `eval`, `compare_numeric`, or `to_latex` is called before all
orders are solved, a clear `NotReadyError` lists exactly which orders are pending.

---

## Non-standard gauge sequences

By default the ansatz uses the standard power sequence $\{1, \varepsilon, \varepsilon^2, \ldots\}$.
You can supply a non-standard gauge when the problem calls for it:

```python
# Half-power gauge: {1, √ε, ε, ε^(3/2), ...}
sol = eq.expand_regular(order=3, gauge="sqrt(eps)")

# Logarithmic gauge: {1, log(ε), log²(ε), ...}
sol = eq.expand_regular(order=2, gauge=["1", "log(eps)", "log(eps)**2"])

# Inspect the gauge used
sol.gauge   # list of SymPy gauge functions
```

---

## Symbolic parameters

Parameters other than the dependent variable, independent variable, and small
parameter are detected automatically and a warning is issued:

```python
eq = ODE("u'' + a*u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"])
# ⚠️  symbolic parameters detected: {'a'}

# Parameters also work in conditions
eq = ODE("u'' + u + eps*u**3", small_param="eps",
         conditions=["u(0) = A", "u'(0) = 0"])
# ⚠️  symbolic parameters detected: {'A'}

# show() and to_latex() always work — results stay symbolic
sol.show()

# Provide values at eval/compare time
sol.eval(eps=0.1, at=t_vals, params={"a": 2.0, "A": 1.5})
sol.compare_numeric(eps=0.1, params={"a": 2.0, "A": 1.5})
# Missing params → clear error listing all required names
```

---

## The four core methods

### `sol.show()`

LaTeX rendering in Jupyter; clean plain text in the terminal.
The small parameter is always displayed as $\varepsilon$, regardless of the name
you chose (`eps`, `mu`, `delta`, …).

```python
sol.show()                   # full hierarchy
sol.show(orders=[0, 1])      # selected orders only
sol.show(mode='text')        # force plain text (useful in scripts)
```

### `sol.eval(eps, at=None, params=None)`

Evaluate the composite expansion and return a NumPy array.

```python
import numpy as np
t_vals = np.linspace(0, 20, 400)

# ODE — single eps
u = sol.eval(eps=0.1, at=t_vals)             # ndarray

# ODE — multiple eps values
u = sol.eval(eps=[0.1, 0.2], at=t_vals)      # dict {0.1: ndarray, 0.2: ndarray}

# Algebraic
x = sol.eval(eps=0.1)                        # float
x = sol.eval(eps=[0.1, 0.2, 0.3])           # ndarray

# ODESystem
r = sol.eval(eps=0.1, at=t_vals)            # {'u': ndarray, 'v': ndarray}

# With symbolic parameters
sol.eval(eps=0.1, at=t_vals, params={"a": 2.0})
```

For Lindstedt and multiple scales, `composite_t` (solution in physical time $t$)
is used automatically — no need to handle the strained-time symbol yourself.

### `sol.compare_numeric(eps, params=None, plot_range=None, filename=None)`

Generate a comparison plot between the perturbation expansion and a SciPy
numerical reference solution.

```python
sol.compare_numeric(eps=0.1)
sol.compare_numeric(eps=[0.1, 0.2, 0.3])           # overlay multiple eps
sol.compare_numeric(eps=0.1, params={"a": 2.0})
sol.compare_numeric(eps=0.1, plot_range=[0, 40])    # override default range
sol.compare_numeric(eps=0.1, filename="fig.pdf")    # save to file
```

The plot range is inferred automatically from the problem's conditions.
No `problem=` argument is required — the problem is stored on the hierarchy.

Returns a `dict` with keys that depend on problem type:

| Problem type | Keys |
|:-------------|:-----|
| All ODE methods | `'t'`, `'u_pert'`, `'u_numerical'`, `'fig'` |
| Boundary layer | additionally `'u_outer'`, `'u_inner'`, `'u_composite'` |
| ODE system | `'u_pert'` and `'u_numerical'` are dicts keyed by variable name |
| Algebraic | `'eps'`, `'perturbation'`, `'numerical'`, `'fig'` |

SciPy solvers used: `solve_ivp` (IVP, Lindstedt, multiple scales),
`solve_bvp` (BVP, boundary layer), `fsolve` (algebraic).

### `sol.to_latex(environment='align', show_orders=False, filename=None)`

Export results as ready-to-paste LaTeX source. The small parameter is always
rendered as `\varepsilon`.

```python
sol.to_latex()                              # print to console
sol.to_latex(filename="result.tex")        # save to file
sol.to_latex(show_orders=True)             # include u₀, u₁, u₂, … before composite
sol.to_latex(environment='equation')       # default: 'align'
sol.to_latex(environment='gather')
```

Lindstedt output includes the frequency expansion $\omega(\varepsilon)$,
the composite in strained time $\tau$, and the composite in physical time $t$.
Boundary layer output includes outer, inner, and composite separately.

---

## Inspecting intermediate steps

Every hierarchy exposes the full symbolic work at each order:

```python
sol = eq.expand_regular(order=3)

# Per-order entries
sol[k].ode                   # the ODE/equation at order k
sol[k].general_solution      # with free integration constants
sol[k].particular_solution   # constants fixed by ICs/BCs
sol[k].secular               # True if secular terms detected (ODE only)

# Assembly
sol.composite                # assembled expansion u(t, ε) as SymPy expression
sol.small_param              # the ε symbol
```

Lindstedt-specific attributes:

```python
sol.omega_0                  # unperturbed frequency (auto-detected)
sol.omega_expansion          # ω(ε) as a SymPy series
sol[k].omega_k_val           # frequency correction ωₖ at order k
sol[k].secularity_condition  # the equation that determined ωₖ
sol.composite_t              # u(t, ε) with τ = ω(ε)·t substituted
```

Multiple scales-specific attributes:

```python
sol.T0, sol.T1               # fast and slow time symbols
sol.amplitude_A              # A(T₁) — solved if B = 0 and Bernoulli ODE applies
sol.amplitude_B              # B(T₁)
sol[k].solvability_A         # amplitude ODE for A
```

---

## Condition syntax

Conditions are plain strings — the same notation you'd write on paper:

```python
conditions = ["u(0) = 1"]                          # 1st-order IVP
conditions = ["u(0) = 1", "u'(0) = 0"]             # 2nd-order IVP
conditions = ["u(0) = 0", "u(1) = 1"]              # BVP
conditions = ["u(pi) = 0", "u'(0) = sqrt(2)"]      # symbolic points/values
conditions = ["u(0) = 1/2", "u'(0) = 0"]           # rational values
conditions = ["u(0) = 0.9", "u'(0) = 0"]           # floats auto-converted
```

`asymptotics` automatically:
- Detects IVP vs BVP from the number of distinct boundary points
- Infers the dependent variable name from condition strings
- Infers the independent variable (`t` for IVPs, `x` for BVPs) from the equation
- Reports exactly what was inferred (with override syntax) at construction time
- Raises clear `ConditionError` for wrong count, conflicts, or inconsistencies

**Limit conditions** for regularity at singular points:

```python
conditions = [
    "F(0) = 0",
    "F(1) = 1",
    "lim(sqrt(2*eta)*F'', eta, 0) = 0",   # lim(expr, var, point) = value
]
```

At each order the library substitutes the general solution, identifies which
free constants cause divergence, and sets them to zero automatically.

---

## Auto-inference of variables

For `ODE`, `dependent` and `independent` are both optional:

```python
# Fully minimal — both inferred
eq = ODE("u'' + u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"])
# ℹ️  dependent = 'u' (inferred from conditions)
# ℹ️  independent = 't' (IVP default)
# To override: ODE(..., dependent='u', independent='t')

# Override when the equation contains a non-standard independent variable
eq = ODE("u'' + sin(tau)*u + eps*u**3", small_param="eps",
         conditions=["u(0) = 1", "u'(0) = 0"],
         independent="tau")
```

Independent variable candidates inferred from the equation: `{x, y, z, t}`.
All other symbols are treated as parameters. If a candidate symbol is
ambiguous (could be independent variable or parameter), a hard error asks you
to specify `independent` explicitly.

---

## Higher-order ODEs

ODEs up to 6th order are supported using prime notation:

```python
# 4th-order ODE
eq = ODE(
    "eta*F'''' + alpha*(eta*F''' + 2*F'') + eps/2*(F*F''' - F'*F'') + 2*F'''",
    small_param = "eps",
    independent = "eta",
    conditions  = ["F(0) = 0", "F(1/2) = 1", "F'(1/2) = 0",
                   "lim(sqrt(2*eta)*F'', eta, 0) = 0"],
)
sol = eq.begin_expansion(order=1)
```

Prime notation: `u'` (1st), `u''` (2nd), `u'''` (3rd), `u''''` (4th),
`u'''''` (5th), `u''''''` (6th).

Lindstedt and multiple scales are limited to 2nd-order oscillators.
Regular perturbation and `begin_expansion` work for any supported order.

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

ValueError: Could not parse equation: 'u^2 + eps*u - 1'
  Use ** for powers:  u**2  not  u^2

NotReadyError: Cannot access 'composite' — 1 order(s) not yet solved: [0]
  Call sol[0].solve() or sol[0].set_solution(expr) first.
```

---

## Project layout

```
asymptotics/
├── __init__.py                      ← public API
├── numerics.py                      ← compare_numeric() dispatcher + SciPy solvers
├── eval.py                          ← eval() for all hierarchy types
├── latex_export.py                  ← to_latex() for all hierarchy types
├── gauge.py                         ← non-standard gauge sequence support
├── core/
│   ├── problem.py                   ← AlgebraicEquation, AlgebraicSystem, ODE
│   ├── ode_system.py                ← ODESystem
│   ├── hierarchy.py                 ← OrderHierarchy, OrderEntry
│   ├── system_hierarchy.py          ← SystemHierarchy (coupled algebraic)
│   ├── conditions.py                ← parser: ParsedCondition, LimitCondition
│   └── exceptions.py               ← custom exceptions
├── methods/
│   ├── regular_algebraic.py         ← AlgebraicEquation solver
│   ├── regular_algebraic_system.py  ← AlgebraicSystem solver
│   ├── regular_ode.py               ← ODE regular perturbation
│   ├── regular_ode_system.py        ← ODESystem solver
│   ├── lindstedt.py                 ← Lindstedt–Poincaré
│   ├── multiple_scales.py           ← Method of multiple scales
│   ├── boundary_layer.py            ← Matched asymptotic expansions
│   └── stepwise.py                  ← StepwiseHierarchy, begin_expansion
├── display/
│   ├── jupyter.py                   ← algebraic LaTeX display
│   ├── ode_display.py
│   ├── ode_system_display.py
│   ├── lindstedt_display.py
│   ├── multiple_scales_display.py
│   └── boundary_layer_display.py
└── tests/                           ← 195 tests, all passing
    ├── test_regular_algebraic.py
    ├── test_algebraic_system.py
    ├── test_errors.py
    ├── test_gauge.py
    ├── test_ode.py
    ├── test_lindstedt.py
    ├── test_multiple_scales.py
    ├── test_boundary_layer.py
    └── test_ode_system.py
```

---

## Example notebooks

Ten Jupyter notebooks covering every feature with worked examples:

| Notebook | Topic |
|:---------|:------|
| `01_introduction.ipynb` | Algebraic equations — root selection, symbolic parameters |
| `02_transcendental.ipynb` | Transcendental equations and coupled algebraic systems |
| `03_ode.ipynb` | Regular perturbation for ODEs — IVP and BVP, secular detection |
| `04_lindstedt.ipynb` | Lindstedt–Poincaré — Duffing oscillator, amplitude dependence |
| `05_multiple_scales.ipynb` | Multiple scales — damped oscillator, Van der Pol limit cycle |
| `06_boundary_layers.ipynb` | Matched asymptotic expansions — left/right layers, variable coefficients |
| `07_ode_system.ipynb` | Coupled ODE systems, predator-prey |
| `08_advanced.ipynb` | Advanced features and edge cases |
| `08_stepwise.ipynb` | Step-by-step API, higher-order ODEs, limit BCs |
| `09_gauge.ipynb` | Non-standard gauge sequences |

---

## Running the tests

```bash
pytest                      # all 195 tests
pytest -v                   # verbose output
pytest --cov=asymptotics    # with coverage report
```

---

## Design philosophy

- **String-based input** — write `"u'' + u + eps*u**3"`, not `u.diff(t, 2) + u + eps*u**3`
- **Transparent by default** — every intermediate step is a live SymPy expression you can inspect and reuse
- **Consistent API** — `expand_*`, `show()`, `eval()`, `compare_numeric()`, `to_latex()` work identically across all problem types
- **Self-contained results** — the problem is stored on the hierarchy; no need to pass it again to `compare_numeric` or `eval`
- **Fail clearly** — errors name the problem, quote your input, and suggest a fix
- **No SymPy wrestling** — the library manages all symbolic machinery internally; users stay at the mathematical level

---

## Citation

If you use `asymptotics` in published work, please cite:

> Tony Saad, *asymptotics: A symbolic perturbation theory toolkit for Python*,
> University of Utah, 2025. https://github.com/saadgroup/asymptotics

---

## License

MIT © 2025 Tony Saad, University of Utah
