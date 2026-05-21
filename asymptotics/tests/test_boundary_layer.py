"""
Tests for matched asymptotic expansions (boundary layer method).
"""

import pytest
import numpy as np
from sympy import exp, simplify, symbols, Symbol, E
from scipy.integrate import solve_bvp

from asymptotics import ODE
from asymptotics.methods.boundary_layer import BoundaryLayerHierarchy

x_sym = Symbol('x')
eps_sym = Symbol('eps')


class TestLayerAtLeft:
    """
    eps*u'' + u' + u = 0,  u(0)=0, u(1)=1
    Layer at x=0 (p=1>0).
    Outer: u_out = e^{1-x}
    Expansion: (1 - e^{(eps-1)x/eps}) * e^{1-x}
    """

    def setup_method(self):
        eq = ODE(
            "eps*u'' + u' + u",
            dependent='u', small_param='eps', independent='x',
            conditions=['u(0) = 0', 'u(1) = 1'],
        )
        self.sol = eq.expand_boundary_layer()
        self.eps = self.sol.small_param
        self.x   = self.sol.independent

    def test_returns_hierarchy(self):
        assert isinstance(self.sol, BoundaryLayerHierarchy)

    def test_layer_location(self):
        assert '0' in self.sol.layer_location

    def test_outer_at_far_bc(self):
        """Outer solution satisfies u(1)=1."""
        val = simplify(self.sol.outer.subs(x_sym, 1) - 1)
        assert val == 0

    def test_expansion_bcs(self):
        """Expansion satisfies u(0)=0 exactly; u(1)=1 up to O(eps)."""
        from sympy import lambdify
        comp = self.sol.expansion
        eps  = self.sol.small_param
        # u(0) = 0 exactly (inner BC)
        assert simplify(comp.subs(x_sym, 0)) == 0
        # u(1) = 1 + O(eps): check numerically for small eps
        comp_fn = lambdify(x_sym, comp.subs(eps, 0.01), 'numpy')
        assert abs(comp_fn(1) - 1) < 0.02

    def test_numerical_accuracy(self):
        """Expansion error should be O(eps)."""
        eps_val = 0.05
        from sympy import lambdify
        comp_fn = lambdify(x_sym, self.sol.expansion.subs(self.eps, eps_val), 'numpy')
        x_vals  = np.linspace(0, 1, 200)

        def bvp(x, y): return [y[1], (-y[1] - y[0]) / eps_val]
        def bc(ya, yb): return [ya[0], yb[0] - 1]
        y0 = np.zeros((2, x_vals.size)); y0[0] = x_vals
        num = solve_bvp(bvp, bc, x_vals, y0, tol=1e-10)
        u_exact = num.sol(x_vals)[0]

        max_err = np.max(np.abs(comp_fn(x_vals) - u_exact))
        assert max_err < 0.15   # O(eps) error


class TestLayerAtRight:
    """
    eps*u'' - u' + u = 0,  u(0)=0, u(1)=1
    Layer at x=1 (p=-1<0).
    Outer: u_out = 0
    Expansion: e^{(x-1)/eps}
    """

    def setup_method(self):
        eq = ODE(
            "eps*u'' - u' + u",
            dependent='u', small_param='eps', independent='x',
            conditions=['u(0) = 0', 'u(1) = 1'],
        )
        self.sol = eq.expand_boundary_layer()

    def test_layer_location(self):
        assert '1' in self.sol.layer_location

    def test_outer(self):
        """Outer solution is 0 for this problem."""
        assert simplify(self.sol.outer) == 0

    def test_expansion_bcs(self):
        comp = self.sol.expansion
        eps = self.sol.small_param
        # u(0): e^{-1/eps} ≈ 0 for small eps
        # u(1): e^0 = 1
        assert simplify(comp.subs(x_sym, 1) - 1) == 0

    def test_numerical_accuracy(self):
        eps_val = 0.05
        from sympy import lambdify
        x_vals  = np.linspace(0, 1, 200)
        comp_fn = lambdify(x_sym,
                           self.sol.expansion.subs(self.sol.small_param, eps_val),
                           'numpy')

        def bvp(x, y): return [y[1], (y[1] - y[0]) / eps_val]
        def bc(ya, yb): return [ya[0], yb[0] - 1]
        y0 = np.zeros((2, x_vals.size)); y0[0] = x_vals
        num = solve_bvp(bvp, bc, x_vals, y0, tol=1e-10)
        u_exact = num.sol(x_vals)[0]

        max_err = np.max(np.abs(comp_fn(x_vals) - u_exact))
        assert max_err < 0.1


class TestVariableCoefficients:
    """
    eps*u'' + (1+x)*u' - u = 0,  u(0)=1, u(1)=2
    Layer at x=0 (p(0)=1>0).
    Outer: u_out = x+1
    """

    def setup_method(self):
        eq = ODE(
            "eps*u'' + (1+x)*u' - u",
            dependent='u', small_param='eps', independent='x',
            conditions=['u(0) = 1', 'u(1) = 2'],
        )
        self.sol = eq.expand_boundary_layer()

    def test_layer_location(self):
        assert '0' in self.sol.layer_location

    def test_outer(self):
        """Outer solution: u_out = x+1."""
        assert simplify(self.sol.outer - (x_sym + 1)) == 0

    def test_outer_far_bc(self):
        """Outer satisfies u(1)=2."""
        assert simplify(self.sol.outer.subs(x_sym, 1) - 2) == 0


class TestErrorChecking:

    def test_ivp_raises(self):
        eq = ODE("eps*u'' + u' + u", dependent='u', small_param='eps', independent='x',
                 conditions=["u(0) = 0", "u'(0) = 1"])
        with pytest.raises(ValueError, match="BVP"):
            eq.expand_boundary_layer()

    def test_first_order_raises(self):
        # A 1st-order ODE with 2 BCs raises ConditionError (wrong BC count)
        # A 2nd-order ODE without eps*u'' raises when trying to expand
        from asymptotics.core.conditions import ConditionError
        with pytest.raises(ConditionError):
            ODE("u' + eps*u", dependent='u', small_param='eps', independent='x',
                conditions=["u(0) = 0", "u(1) = 1"])

    def test_no_eps_raises(self):
        from asymptotics.core.exceptions import NoSmallParameterError
        eq = ODE("u'' + u' + u", dependent='u', small_param='eps', independent='x',
                 conditions=["u(0) = 0", "u(1) = 1"])
        with pytest.raises(NoSmallParameterError):
            eq.expand_boundary_layer()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
