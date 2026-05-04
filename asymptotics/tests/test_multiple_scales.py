"""
Tests for the method of multiple scales.
"""

import pytest
import numpy as np
from sympy import (
    exp, cos, sin, simplify, symbols, Function,
    Rational, sqrt, Symbol
)
from scipy.integrate import solve_ivp

from asymptotics import ODE
from asymptotics.methods.multiple_scales import MultScalesHierarchy


T1_sym = Symbol('T_1')
T0_sym = Symbol('T_0')


class TestDampedOscillator:
    """
    u'' + u + eps*u' = 0,  u(0)=1, u'(0)=0

    Exact solution: e^{-eps*t/2} * cos(sqrt(1-eps^2/4)*t)
    Leading-order multiple scales: e^{-eps*t/2} * cos(t)

    Solvability: dA/dT1 = -A/2,  dB/dT1 = -B/2
    Solution: A(T1) = e^{-T1/2},  B(T1) = 0
    """

    def setup_method(self):
        eq = ODE(
            "u'' + u + eps*u'",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        self.sol = eq.expand_multiple_scales(order=1)
        self.eps = self.sol.small_param
        self.t   = self.sol.independent

    def test_returns_hierarchy(self):
        assert isinstance(self.sol, MultScalesHierarchy)

    def test_omega0(self):
        assert self.sol.omega_0 == 1

    def test_amplitude_A(self):
        A = Function('A')(T1_sym)
        expected = exp(-T1_sym / 2)
        assert simplify(self.sol.amplitude_A - expected) == 0

    def test_amplitude_B(self):
        assert simplify(self.sol.amplitude_B) == 0

    def test_solvability_A(self):
        """dA/dT1 = -A/2"""
        eq_A = self.sol.entries[1].solvability_A
        assert eq_A is not None
        A = Function('A')(T1_sym)
        assert simplify(eq_A.rhs + A/2) == 0

    def test_solvability_B(self):
        """dB/dT1 = -B/2"""
        eq_B = self.sol.entries[1].solvability_B
        assert eq_B is not None

    def test_composite_matches_exact_leading(self):
        """Composite at order 1 should match e^{-eps*t/2}*cos(t)"""
        t = self.t
        eps = self.eps
        expected = exp(-eps*t/2) * cos(t)
        diff = simplify(self.sol.composite_t - expected)
        assert diff == 0

    def test_numerical_accuracy(self):
        """Composite should match numerical solution for small eps."""
        eps_val  = 0.1
        t_vals   = np.linspace(0, 20, 500)
        eps_sym  = self.sol.small_param
        t_sym    = self.sol.independent

        from sympy import lambdify
        comp_fn = lambdify(t_sym, self.sol.composite_t.subs(eps_sym, eps_val), 'numpy')
        u_pert  = comp_fn(t_vals)

        def rhs(t, y): return [y[1], -y[0] - eps_val*y[1]]
        num = solve_ivp(rhs, [0, 20], [1.0, 0.0], dense_output=True, rtol=1e-12)
        u_exact = num.sol(t_vals)[0]

        max_err = np.max(np.abs(u_pert - u_exact))
        assert max_err < 0.05   # reasonable for order-1 approximation


class TestDuffingMultipleScales:
    """
    u'' + u + eps*u^3 = 0,  u(0)=1, u'(0)=0

    Solvability (with B=0):
        dA/dT1 = 0  => A = 1 (constant)
        dB/dT1 = -3*A^2/8 => frequency shift (same as Lindstedt)

    SymPy can't solve the coupled A,B ODEs in general — leaves symbolic.
    """

    def setup_method(self):
        eq = ODE(
            "u'' + u + eps*u**3",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        self.sol = eq.expand_multiple_scales(order=1)

    def test_returns_hierarchy(self):
        assert isinstance(self.sol, MultScalesHierarchy)

    def test_omega0(self):
        assert self.sol.omega_0 == 1

    def test_solvability_conditions_exist(self):
        """Solvability conditions should be found."""
        entry = self.sol.entries[1]
        assert entry.solvability_A is not None or entry.solvability_B is not None

    def test_secular_terms_found(self):
        """Secular coefficients should be non-zero."""
        entry = self.sol.entries[1]
        assert entry.secular_cos != 0 or entry.secular_sin != 0

    def test_u1_no_secular(self):
        """u_1(T0, T1) should not contain T0*cos(T0) or T0*sin(T0)."""
        u1 = self.sol.entries[1].particular_solution
        # Secular terms would have T_0 multiplying trig — check string
        u1_str = str(u1)
        assert 'T_0*cos' not in u1_str and 'T_0*sin' not in u1_str


class TestHigherFrequencyDamped:
    """
    u'' + 4u + eps*u' = 0,  u(0)=1, u'(0)=0
    omega_0 = 2 — auto-detected
    """

    def setup_method(self):
        eq = ODE(
            "u'' + 4*u + eps*u'",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        self.sol = eq.expand_multiple_scales(order=1)

    def test_omega0_detected(self):
        assert self.sol.omega_0 == 2

    def test_amplitude_A(self):
        """A(T1) = e^{-T1/2} regardless of omega_0."""
        expected = exp(-T1_sym / 2)
        assert simplify(self.sol.amplitude_A - expected) == 0


class TestErrorChecking:

    def test_first_order_raises(self):
        eq = ODE(
            "u' + u + eps*u**2",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1"],
        )
        with pytest.raises(ValueError, match="2nd-order"):
            eq.expand_multiple_scales(order=1)

    def test_no_small_param_raises(self):
        from asymptotics.core.exceptions import NoSmallParameterError
        eq = ODE(
            "u'' + u",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        with pytest.raises(NoSmallParameterError):
            eq.expand_multiple_scales(order=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
