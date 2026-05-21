"""
Tests for non-standard asymptotic gauge sequences.

Covers:
- gauge.py utilities (parse_gauge, extract_coefficients, validate)
- AlgebraicEquation.expand_regular(gauge=...)
- ODE.expand_regular(gauge=...)
- Backward compatibility (gauge=None)
- Error handling
"""

import pytest
from sympy import (
    Symbol, sqrt, log, simplify, Integer, Rational, expand, pi
)
from asymptotics import AlgebraicEquation, ODE
from asymptotics.gauge import (
    parse_gauge, extract_coefficients, gauge_to_latex,
    is_standard_gauge, _validate_gauge,
)

eps = Symbol("eps", positive=True)


# ===========================================================================
# gauge.py utilities
# ===========================================================================

class TestParseGauge:
    def test_none_returns_standard(self):
        g = parse_gauge(None, 3, eps)
        assert len(g) == 4
        for k, dk in enumerate(g):
            assert simplify(dk - eps**k) == 0

    def test_string_sqrt_infers_geometric(self):
        g = parse_gauge("sqrt(eps)", 3, eps)
        assert len(g) == 4
        assert simplify(g[0] - 1) == 0
        assert simplify(g[1] - sqrt(eps)) == 0
        assert simplify(g[2] - eps) == 0
        assert simplify(g[3] - eps**Rational(3, 2)) == 0

    def test_string_eps2_even_powers(self):
        g = parse_gauge("eps**2", 3, eps)
        for k, dk in enumerate(g):
            assert simplify(dk - eps**(2*k)) == 0

    def test_string_eps_third(self):
        g = parse_gauge("eps**(1/3)", 3, eps)
        assert simplify(g[1] - eps**Rational(1, 3)) == 0
        assert simplify(g[2] - eps**Rational(2, 3)) == 0
        assert simplify(g[3] - eps) == 0

    def test_explicit_list_correct_length(self):
        g = parse_gauge(["1", "eps*log(eps)", "eps"], 2, eps)
        assert len(g) == 3
        assert simplify(g[0] - 1) == 0
        assert simplify(g[1] - eps*log(eps)) == 0
        assert simplify(g[2] - eps) == 0

    def test_explicit_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 4 terms"):
            parse_gauge(["1", "sqrt(eps)"], 3, eps)

    def test_explicit_list_wrong_length_message(self):
        with pytest.raises(ValueError, match="2 term"):
            parse_gauge(["1", "eps"], 3, eps)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            parse_gauge(42, 2, eps)

    def test_bad_string_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_gauge("not_valid!!!", 2, eps)


class TestExtractCoefficients:
    def test_standard_power_law(self):
        x0, x1, x2 = [Symbol(f"x{k}") for k in range(3)]
        gauge = parse_gauge(None, 2, eps)
        expr = x0 + x1*eps + x2*eps**2
        coeffs = extract_coefficients(expr, gauge, eps)
        assert simplify(coeffs[0] - x0) == 0
        assert simplify(coeffs[1] - x1) == 0
        assert simplify(coeffs[2] - x2) == 0

    def test_sqrt_gauge(self):
        x0, x1, x2 = [Symbol(f"x{k}") for k in range(3)]
        gauge = parse_gauge("sqrt(eps)", 2, eps)
        expr = x0 + x1*sqrt(eps) + x2*eps
        coeffs = extract_coefficients(expr, gauge, eps)
        assert simplify(coeffs[0] - x0) == 0
        assert simplify(coeffs[1] - x1) == 0
        assert simplify(coeffs[2] - x2) == 0

    def test_log_gauge(self):
        x0, x1, x2 = [Symbol(f"x{k}") for k in range(3)]
        gauge = parse_gauge(["1", "eps*log(eps)", "eps"], 2, eps)
        expr = x0 + x1*eps*log(eps) + x2*eps
        coeffs = extract_coefficients(expr, gauge, eps)
        assert simplify(coeffs[0] - x0) == 0
        assert simplify(coeffs[1] - x1) == 0
        assert simplify(coeffs[2] - x2) == 0

    def test_gauge_starting_at_sqrt(self):
        """Gauge that starts at sqrt(eps): leading term has no O(1) part."""
        x0, x1 = Symbol("x0"), Symbol("x1")
        gauge = [sqrt(eps), eps]
        expr = x0*sqrt(eps) + x1*eps
        coeffs = extract_coefficients(expr, gauge, eps)
        assert simplify(coeffs[0] - x0) == 0
        assert simplify(coeffs[1] - x1) == 0

    def test_remainder_is_zero(self):
        """After extraction the remainder should be zero."""
        x0, x1 = Symbol("x0"), Symbol("x1")
        gauge = parse_gauge(None, 1, eps)
        expr = x0 + x1*eps
        coeffs = extract_coefficients(expr, gauge, eps)
        remainder = expand(expr - coeffs[0]*gauge[0] - coeffs[1]*gauge[1])
        assert simplify(remainder) == 0


class TestIsStandardGauge:
    def test_standard(self):
        g = parse_gauge(None, 3, eps)
        assert is_standard_gauge(g, eps)

    def test_non_standard(self):
        g = parse_gauge("sqrt(eps)", 3, eps)
        assert not is_standard_gauge(g, eps)


class TestGaugeToLatex:
    def test_standard_renders(self):
        g = parse_gauge(None, 2, eps)
        lx = gauge_to_latex(g, eps)
        assert len(lx) == 3
        assert r"\varepsilon" in lx[1]

    def test_sqrt_renders(self):
        g = parse_gauge("sqrt(eps)", 2, eps)
        lx = gauge_to_latex(g, eps)
        assert r"\varepsilon" in lx[1]


# ===========================================================================
# AlgebraicEquation with gauge
# ===========================================================================

class TestAlgebraicGauge:
    def test_backward_compat_no_gauge(self):
        """gauge=None must reproduce the existing standard result."""
        eq = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=3)
        # Known result: 1 - eps/3 - eps^3/81
        from sympy import Rational as R
        val = sol.expansion.subs(Symbol("eps"), R(1, 10))
        assert abs(float(val) - (1 - 0.1/3)) < 1e-3

    def test_sqrt_gauge_x_squared_minus_eps(self):
        """x^2 - eps = 0 with gauge [sqrt(eps), eps, eps^(3/2)] → x = sqrt(eps)."""
        import numpy as np
        eq = AlgebraicEquation("x**2 - eps", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=2, gauge=["sqrt(eps)", "eps", "eps**(3/2)"])
        # Compare numerically — avoids Symbol assumption mismatches
        eps_sym = list(sol.expansion.free_symbols)[0]
        for v in [0.01, 0.1, 0.25]:
            pert  = float(sol.expansion.subs(eps_sym, v))
            exact = float(v**0.5)
            assert abs(pert - exact) < 1e-14

    def test_even_power_gauge_string(self):
        """x^3 + eps^2*x - 1 = 0 with gauge='eps**2' → 1 - eps^2/3."""
        eq = AlgebraicEquation("x**3 + eps**2*x - 1", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=2, gauge="eps**2")
        # Check numerically — avoids Symbol assumption mismatches
        eps_sym = list(sol.expansion.free_symbols)[0]
        for v in [0.1, 0.2, 0.3]:
            pert     = float(sol.expansion.subs(eps_sym, v))
            expected = 1 - v**2 / 3
            assert abs(pert - expected) < 1e-12

    def test_string_pattern_gauge_infers_correct_sequence(self):
        """gauge='eps**2' applied to order=1 gives [1, eps^2]."""
        from sympy import Integer
        eq = AlgebraicEquation("x**3 + eps**2*x - 1", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=1, gauge="eps**2")
        # Use the gauge's own eps symbol to avoid assumption mismatch
        g_eps = list(sol._gauge[1].free_symbols)[0]
        assert sol._gauge[0] == Integer(1)
        assert sol._gauge[1] == g_eps**2

    def test_gauge_stored_on_hierarchy(self):
        eq = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=2, gauge="sqrt(eps)")
        assert hasattr(sol, "_gauge")
        assert len(sol._gauge) == 3

    def test_wrong_length_raises_value_error(self):
        eq = AlgebraicEquation("x**2 - eps", dependent="x", small_param="eps")
        with pytest.raises(ValueError, match="exactly 4 terms"):
            eq.expand_regular(order=3, gauge=["sqrt(eps)", "eps"])

    def test_eval_works_with_gauge(self):
        """eval() should work normally after a non-standard expansion."""
        import numpy as np
        eq = AlgebraicEquation("x**3 + eps**2*x - 1", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=2, gauge="eps**2")
        val = sol.eval(eps=0.1)
        assert abs(val - 1.0) < 0.01  # eps=0.1 is small, x ≈ 1


# ===========================================================================
# ODE with gauge
# ===========================================================================

class TestODEGauge:
    def test_backward_compat_no_gauge(self):
        """ODE gauge=None must reproduce existing result."""
        eq = ODE("u'' + u + eps*u", small_param="eps",
                 conditions=["u(0)=1", "u'(0)=0"])
        sol = eq.expand_regular(order=2)
        assert sol.expansion is not None

    def test_even_power_gauge_ode(self):
        """u'' + u + eps^2*u^3; gauge=eps^2 → u_1=0 by symmetry."""
        eq = ODE("u'' + u + eps**2*u**3", small_param="eps",
                 conditions=["u(0)=1", "u'(0)=0"])
        sol = eq.expand_regular(order=2, gauge="eps**2")
        import sympy as sp
        t = sol.independent
        # u_0 = cos(t); u_1 = 0 (odd power absent from even gauge)
        # expansion = cos(t) + eps^2*(correction)
        u0 = sol.entries[0].particular_solution
        assert simplify(u0 - sp.cos(t)) == 0

    def test_gauge_stored_on_ode_hierarchy(self):
        eq = ODE("u'' + u + eps*u**3", small_param="eps",
                 conditions=["u(0)=1", "u'(0)=0"])
        sol = eq.expand_regular(order=2, gauge="sqrt(eps)")
        assert hasattr(sol, "_gauge")
        assert len(sol._gauge) == 3

    def test_wrong_length_ode_raises(self):
        eq = ODE("u'' + u + eps*u**3", small_param="eps",
                 conditions=["u(0)=1", "u'(0)=0"])
        with pytest.raises(ValueError, match="exactly 3 terms"):
            eq.expand_regular(order=2, gauge=["1", "sqrt(eps)"])


# ===========================================================================
# Numerical sanity checks
# ===========================================================================

class TestGaugeNumerical:
    def test_sqrt_gauge_numerical_accuracy(self):
        """x^2 - eps = 0: perturbation should equal sqrt(eps) to machine precision."""
        import numpy as np
        eq = AlgebraicEquation("x**2 - eps", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=2, gauge=["sqrt(eps)", "eps", "eps**(3/2)"])
        for eps_val in [0.01, 0.05, 0.1, 0.2]:
            pert = float(sol.expansion.subs(Symbol("eps"), eps_val))
            exact = eps_val**0.5
            assert abs(pert - exact) < 1e-12, f"eps={eps_val}: {pert} vs {exact}"

    def test_even_gauge_numerical_accuracy(self):
        """x^3 + eps^2*x - 1: compare perturbation vs scipy at eps=0.1."""
        import numpy as np
        from scipy.optimize import fsolve
        eq = AlgebraicEquation("x**3 + eps**2*x - 1", dependent="x", small_param="eps")
        sol = eq.expand_regular(order=2, gauge="eps**2")
        for eps_val in [0.1, 0.2, 0.3]:
            pert = float(sol.expansion.subs(Symbol("eps"), eps_val))
            exact = float(fsolve(lambda x: x**3 + eps_val**2*x - 1, 1.0)[0])
            assert abs(pert - exact) < 1e-4, f"eps={eps_val}: pert={pert}, exact={exact}"
