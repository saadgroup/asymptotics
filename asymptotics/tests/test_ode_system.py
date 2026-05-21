"""Tests for ODESystem — coupled ODE perturbation expansion."""

import pytest
import numpy as np
from sympy import exp, simplify, Symbol, symbols
from scipy.integrate import solve_ivp

from asymptotics import ODESystem
from asymptotics.methods.regular_ode_system import ODESystemHierarchy
from asymptotics.core.conditions import ConditionError

t_sym = Symbol('t')


class TestTwoEquationSystem:
    """
    u' + u + eps*v = 0,   u(0)=1
    v' + 2v + eps*u^2=0,  v(0)=1

    O(1): u0=e^{-t}, v0=e^{-2t}
    O(eps): u1=e^{-2t}-e^{-t}, v1=-t*e^{-2t}
    """

    def setup_method(self):
        sys = ODESystem(
            equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
            dependents  = ['u', 'v'],
            small_param = 'eps',
            independent = 't',
            conditions  = ['u(0) = 1', 'v(0) = 1'],
        )
        self.sol = sys.expand_regular(order=2)
        self.eps = self.sol.small_param

    def test_returns_hierarchy(self):
        assert isinstance(self.sol, ODESystemHierarchy)

    def test_variables(self):
        assert self.sol.variables == ['u', 'v']

    def test_access_by_name(self):
        assert self.sol['u'] is not None
        assert self.sol['v'] is not None

    def test_leading_order_u(self):
        u0 = self.sol['u'][0].particular_solution
        assert simplify(u0 - exp(-t_sym)) == 0

    def test_leading_order_v(self):
        v0 = self.sol['v'][0].particular_solution
        assert simplify(v0 - exp(-2*t_sym)) == 0

    def test_order1_u(self):
        u1 = self.sol['u'][1].particular_solution
        expected = exp(-2*t_sym) - exp(-t_sym)
        assert simplify(u1 - expected) == 0

    def test_order1_v(self):
        v1 = self.sol['v'][1].particular_solution
        expected = -t_sym * exp(-2*t_sym)
        assert simplify(v1 - expected) == 0

    def test_ics_satisfied_order0(self):
        assert simplify(self.sol['u'][0].particular_solution.subs(t_sym, 0) - 1) == 0
        assert simplify(self.sol['v'][0].particular_solution.subs(t_sym, 0) - 1) == 0

    def test_ics_homogeneous_higher_orders(self):
        for k in [1, 2]:
            assert simplify(self.sol['u'][k].particular_solution.subs(t_sym, 0)) == 0
            assert simplify(self.sol['v'][k].particular_solution.subs(t_sym, 0)) == 0

    def test_numerical_accuracy(self):
        eps_val = 0.1
        from sympy import lambdify
        u_fn = lambdify(t_sym, self.sol['u'].expansion.subs(self.eps, eps_val), 'numpy')
        v_fn = lambdify(t_sym, self.sol['v'].expansion.subs(self.eps, eps_val), 'numpy')

        def rhs(t, y): return [-y[0] - eps_val*y[1], -2*y[1] - eps_val*y[0]**2]
        num    = solve_ivp(rhs, [0, 5], [1.0, 1.0], dense_output=True, rtol=1e-10)
        t_vals = np.linspace(0, 5, 100)
        u_num, v_num = num.sol(t_vals)

        assert np.max(np.abs(u_fn(t_vals) - u_num)) < 0.01
        assert np.max(np.abs(v_fn(t_vals) - v_num)) < 0.01


class TestThreeEquationSystem:
    """
    u' + u + eps*v = 0,      u(0)=1
    v' + 2v + eps*u^2 = 0,   v(0)=1
    w' + w + eps*(u+v) = 0,  w(0)=0
    """

    def setup_method(self):
        sys = ODESystem(
            equations   = ["u' + u + eps*v", "v' + 2*v + eps*u**2",
                           "w' + w + eps*(u+v)"],
            dependents  = ['u', 'v', 'w'],
            small_param = 'eps',
            independent = 't',
            conditions  = ['u(0) = 1', 'v(0) = 1', 'w(0) = 0'],
        )
        self.sol = sys.expand_regular(order=1)

    def test_variables(self):
        assert self.sol.variables == ['u', 'v', 'w']

    def test_w0_zero(self):
        """w has zero IC and no O(1) forcing, so w0=0."""
        assert simplify(self.sol['w'][0].particular_solution) == 0

    def test_w1_ic(self):
        """w1 should satisfy homogeneous IC w1(0)=0."""
        w1 = self.sol['w'][1].particular_solution
        assert simplify(w1.subs(t_sym, 0)) == 0


class TestErrorChecking:

    def test_wrong_number_of_equations(self):
        with pytest.raises(ValueError, match="Number of equations"):
            ODESystem(
                equations  = ["u' + u + eps*v"],
                dependents = ['u', 'v'],
                small_param= 'eps',
                independent= 't',
                conditions = ['u(0) = 1', 'v(0) = 0'],
            )

    def test_wrong_condition_count(self):
        with pytest.raises(ConditionError):
            ODESystem(
                equations  = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
                dependents = ['u', 'v'],
                small_param= 'eps',
                independent= 't',
                conditions = ['u(0) = 1'],   # missing v condition
            )

    def test_no_derivative_raises(self):
        with pytest.raises(ValueError, match="No derivatives"):
            ODESystem(
                equations  = ["u + eps*v", "v' + 2*v"],
                dependents = ['u', 'v'],
                small_param= 'eps',
                independent= 't',
                conditions = ['u(0) = 1', 'v(0) = 0'],
            )

    def test_unrecognized_condition(self):
        with pytest.raises(ConditionError):
            ODESystem(
                equations  = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
                dependents = ['u', 'v'],
                small_param= 'eps',
                independent= 't',
                conditions = ['u(0) = 1', 'w(0) = 0'],   # 'w' not in system
            )

    def test_bad_order_type(self):
        sys = ODESystem(
            equations  = ["u' + u + eps*v", "v' + 2*v + eps*u**2"],
            dependents = ['u', 'v'],
            small_param= 'eps',
            independent= 't',
            conditions = ['u(0) = 1', 'v(0) = 0'],
        )
        with pytest.raises(TypeError):
            sys.expand_regular(order='two')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
