"""
Tests for the Lindstedt–Poincaré method.

Key verified results:
- Duffing oscillator u'' + u + eps*u^3 = 0, u(0)=1, u'(0)=0:
    omega_1 = 3/8  (classical result)
    u_0 = cos(tau)
    u_1 = (cos(3*tau) - cos(tau)) / 32
"""

import pytest
import numpy as np
from sympy import cos, sin, Rational, simplify, symbols, pi, sqrt
from scipy.integrate import solve_ivp

from asymptotics import ODE
from asymptotics.methods.lindstedt import LindstedtHierarchy

tau_sym = symbols('tau')
t_sym   = symbols('t')


class TestDuffingOscillator:
    """
    u'' + u + eps*u^3 = 0,  u(0)=1, u'(0)=0

    Classical Lindstedt results:
        omega_0 = 1
        omega_1 = 3/8
        u_0(tau) = cos(tau)
        u_1(tau) = (cos(3*tau) - cos(tau)) / 32
    """

    def setup_method(self):
        eq = ODE(
            "u'' + u + eps*u**3",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        self.sol = eq.expand_lindstedt(order=2)

    def test_returns_lindstedt_hierarchy(self):
        assert isinstance(self.sol, LindstedtHierarchy)

    def test_omega0(self):
        assert self.sol.omega_0 == 1

    def test_omega1(self):
        assert self.sol[1].omega_k_val == Rational(3, 8)

    def test_u0(self):
        assert simplify(self.sol[0].particular_solution - cos(tau_sym)) == 0

    def test_u1(self):
        expected = (cos(3*tau_sym) - cos(tau_sym)) / 32
        diff = simplify(self.sol[1].particular_solution - expected)
        assert diff == 0

    def test_u0_ics(self):
        u0 = self.sol[0].particular_solution
        assert simplify(u0.subs(tau_sym, 0) - 1) == 0
        assert simplify(u0.diff(tau_sym).subs(tau_sym, 0)) == 0

    def test_u1_homogeneous_ics(self):
        u1 = self.sol[1].particular_solution
        assert simplify(u1.subs(tau_sym, 0)) == 0
        assert simplify(u1.diff(tau_sym).subs(tau_sym, 0)) == 0

    def test_no_secular_in_u1(self):
        """u_1 must not contain tau*sin(tau) or tau*cos(tau)."""
        from sympy import preorder_traversal, Mul
        u1 = self.sol[1].particular_solution
        for term in preorder_traversal(u1):
            if term.is_Mul:
                has_tau = any(str(a) == 'tau' for a in term.args)
                has_trig = any(isinstance(a, (type(cos(tau_sym)), type(sin(tau_sym)))) for a in term.args)
                assert not (has_tau and has_trig), f"Secular term found: {term}"

    def test_hierarchy_length(self):
        assert len(self.sol) == 3   # orders 0, 1, 2

    def test_period_accuracy(self):
        """Period from Lindstedt should match numerical integration closely."""
        eps_val  = 0.1
        eps_sym  = self.sol.small_param
        omega_val = float(self.sol.omega_expansion.subs(eps_sym, eps_val))
        T_lindstedt = 2 * np.pi / omega_val

        # Numerical period
        def rhs(t, y): return [y[1], -y[0] - eps_val*y[0]**3]
        num = solve_ivp(rhs, [0, 50], [1.0, 0.0], dense_output=True, rtol=1e-12)
        t_fine = np.linspace(0, 20, 5000)
        u_fine = num.sol(t_fine)[0]
        peaks  = [i for i in range(1, len(u_fine)-1)
                  if u_fine[i] > u_fine[i-1] and u_fine[i] > u_fine[i+1]]
        T_num = t_fine[peaks[1]] - t_fine[peaks[0]]

        assert abs(T_lindstedt - T_num) < 1e-3

    def test_omega_expansion(self):
        """omega = 1 + 3/8*eps + omega_2*eps^2."""
        eps_sym = self.sol.small_param
        omega = self.sol.omega_expansion
        assert simplify(omega.coeff(eps_sym, 0) - 1) == 0
        assert simplify(omega.coeff(eps_sym, 1) - Rational(3, 8)) == 0


class TestHigherFrequencyOscillator:
    """
    u'' + 4u + eps*u^3 = 0,  u(0)=1, u'(0)=0
    omega_0 = 2  (auto-detected)
    """

    def setup_method(self):
        eq = ODE(
            "u'' + 4*u + eps*u**3",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        self.sol = eq.expand_lindstedt(order=1)

    def test_omega0_detected(self):
        assert self.sol.omega_0 == 2

    def test_u0(self):
        # In strained time tau, u0 = cos(tau) regardless of omega_0
        # because tau = omega_0*t already absorbs the frequency
        assert simplify(self.sol[0].particular_solution - cos(tau_sym)) == 0

    def test_period_accuracy(self):
        eps_val   = 0.1
        eps_sym   = self.sol.small_param
        omega_val = float(self.sol.omega_expansion.subs(eps_sym, eps_val))
        T_lindstedt = 2 * np.pi / omega_val

        def rhs(t, y): return [y[1], -4*y[0] - eps_val*y[0]**3]
        num = solve_ivp(rhs, [0, 20], [1.0, 0.0], dense_output=True, rtol=1e-12)
        t_fine = np.linspace(0, 10, 5000)
        u_fine = num.sol(t_fine)[0]
        peaks  = [i for i in range(1, len(u_fine)-1)
                  if u_fine[i] > u_fine[i-1] and u_fine[i] > u_fine[i+1]]
        T_num = t_fine[peaks[1]] - t_fine[peaks[0]]

        assert abs(T_lindstedt - T_num) < 0.05


class TestErrorChecking:
    """Error handling for Lindstedt method."""

    def test_first_order_raises(self):
        """Lindstedt requires 2nd-order ODE."""
        eq = ODE(
            "u' + u + eps*u**2",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1"],
        )
        with pytest.raises(ValueError, match="2nd-order"):
            eq.expand_lindstedt(order=2)

    def test_no_small_param_raises(self):
        from asymptotics.core.exceptions import NoSmallParameterError
        eq = ODE(
            "u'' + u",
            dependent='u', small_param='eps', independent='t',
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        with pytest.raises(NoSmallParameterError):
            eq.expand_lindstedt(order=2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
