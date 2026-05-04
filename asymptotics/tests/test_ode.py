"""
Tests for ODE — condition parsing, validation, and solving.
"""

import pytest
from sympy import exp, cos, sin, symbols, simplify, Rational

from asymptotics import ODE
from asymptotics.core.conditions import ConditionError, parse_condition, ParsedCondition


t = symbols('t')


# ===========================================================================
# Condition parsing tests
# ===========================================================================

class TestConditionParsing:
    """parse_condition handles various string formats."""

    def test_value_at_zero(self):
        c = parse_condition("u(0) = 1", "u")
        assert c.var == "u"
        assert c.deriv_order == 0
        assert c.point == 0
        assert c.value == 1

    def test_first_derivative(self):
        c = parse_condition("u'(0) = 0", "u")
        assert c.deriv_order == 1
        assert c.point == 0
        assert c.value == 0

    def test_second_derivative(self):
        c = parse_condition("u''(0) = 2", "u")
        assert c.deriv_order == 2

    def test_symbolic_point(self):
        from sympy import pi
        c = parse_condition("u(pi) = 0", "u")
        assert c.point == pi

    def test_symbolic_value(self):
        from sympy import sqrt
        c = parse_condition("u(0) = sqrt(2)", "u")
        assert c.value == sqrt(2)

    def test_negative_value(self):
        c = parse_condition("u(1) = -1", "u")
        assert c.value == -1

    def test_wrong_variable_raises(self):
        with pytest.raises(ConditionError, match="refers to variable"):
            parse_condition("v(0) = 1", "u")

    def test_bad_format_raises(self):
        with pytest.raises(ConditionError, match="Cannot parse"):
            parse_condition("u[0] = 1", "u")


# ===========================================================================
# Condition validation tests
# ===========================================================================

class TestConditionValidation:
    """parse_and_validate_conditions catches all error cases."""

    def test_wrong_count_too_many(self):
        with pytest.raises(ConditionError, match="3 condition"):
            ODE(
                "u'' + u",
                dependent="u", small_param="eps", independent="t",
                conditions=["u(0) = 1", "u'(0) = 0", "u(1) = 0"],
            )

    def test_wrong_count_too_few(self):
        with pytest.raises(ConditionError, match="1 condition"):
            ODE(
                "u'' + u",
                dependent="u", small_param="eps", independent="t",
                conditions=["u(0) = 1"],
            )

    def test_conflicting_conditions(self):
        with pytest.raises(ConditionError, match="Conflicting"):
            ODE(
                "u'' + u",
                dependent="u", small_param="eps", independent="t",
                conditions=["u(0) = 1", "u(0) = 2"],
            )

    def test_three_distinct_points(self):
        with pytest.raises(ConditionError, match="3 distinct"):
            ODE(
                "u'' + u",
                dependent="u", small_param="eps", independent="t",
                conditions=["u(0) = 0", "u(1) = 1"],  # this is fine
            )
            # force 3 points by using a special case
            from asymptotics.core.conditions import parse_and_validate_conditions
            parse_and_validate_conditions(
                ["u(0) = 0", "u(1) = 1", "u(2) = 0"],
                "u", 3
            )

    def test_wrong_variable_in_condition(self):
        with pytest.raises(ConditionError, match="refers to variable"):
            ODE(
                "u' + u",
                dependent="u", small_param="eps", independent="t",
                conditions=["v(0) = 1"],
            )

    def test_no_derivative_raises(self):
        with pytest.raises(ValueError, match="No derivatives"):
            ODE(
                "u + eps*u",
                dependent="u", small_param="eps", independent="t",
                conditions=["u(0) = 1"],
            )

    def test_ivp_detected(self):
        eq = ODE(
            "u'' + u",
            dependent="u", small_param="eps", independent="t",
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        assert eq.problem_type == "ivp"

    def test_bvp_detected(self):
        eq = ODE(
            "u'' + u",
            dependent="u", small_param="eps", independent="t",
            conditions=["u(0) = 0", "u(1) = 1"],
        )
        assert eq.problem_type == "bvp"


# ===========================================================================
# Solver tests — IVP
# ===========================================================================

class TestFirstOrderIVP:
    """
    u' + u + eps*u^2 = 0,  u(0) = 1

    Exact solution: u = 1/(1 + (e^t - 1)*eps)
    Expand in eps: u ≈ e^{-t} + eps*e^{-t}(e^{-t} - 1) + ...
    """

    def setup_method(self):
        eq = ODE(
            "u' + u + eps*u**2",
            dependent="u", small_param="eps", independent="t",
            conditions=["u(0) = 1"],
        )
        self.sol = eq.expand_regular(order=2)
        self.eps = self.sol.small_param

    def test_order0_satisfies_ic(self):
        """u_0(0) should equal 1."""
        val = self.sol[0].particular_solution.subs(t, 0)
        assert simplify(val - 1) == 0

    def test_order1_satisfies_ic(self):
        """u_1(0) should equal 0 (homogeneous)."""
        val = self.sol[1].particular_solution.subs(t, 0)
        assert simplify(val) == 0

    def test_order2_satisfies_ic(self):
        """u_2(0) should equal 0 (homogeneous)."""
        val = self.sol[2].particular_solution.subs(t, 0)
        assert simplify(val) == 0

    def test_leading_order_solution(self):
        """u_0 = e^{-t}"""
        assert simplify(self.sol[0].particular_solution - exp(-t)) == 0

    def test_composite_at_zero(self):
        """Composite at t=0 should equal 1 for any eps."""
        val = self.sol.composite.subs(t, 0)
        assert simplify(val - 1) == 0

    def test_hierarchy_length(self):
        assert len(self.sol) == 3

    def test_no_secular_terms(self):
        """First-order IVP should not produce secular terms."""
        assert not any(e.secular for e in self.sol.entries)


class TestSecondOrderIVP:
    """
    u'' + u + eps*u^3 = 0,  u(0)=1, u'(0)=0  (Duffing oscillator)

    u_0 = cos(t) — secular terms appear at O(eps).
    """

    def setup_method(self):
        eq = ODE(
            "u'' + u + eps*u**3",
            dependent="u", small_param="eps", independent="t",
            conditions=["u(0) = 1", "u'(0) = 0"],
        )
        self.sol = eq.expand_regular(order=1)

    def test_leading_order(self):
        """u_0 = cos(t)"""
        assert simplify(self.sol[0].particular_solution - cos(t)) == 0

    def test_order0_ics(self):
        u0 = self.sol[0].particular_solution
        assert simplify(u0.subs(t, 0) - 1) == 0
        assert simplify(u0.diff(t).subs(t, 0)) == 0

    def test_secular_detected(self):
        """Duffing oscillator produces secular terms at O(eps)."""
        assert self.sol[1].secular is True

    def test_problem_type(self):
        assert self.sol._problem_type == "ivp"


# ===========================================================================
# Solver tests — BVP
# ===========================================================================

class TestBVP:
    """
    u'' + eps*u = 0,  u(0)=0, u(1)=1

    Exact unperturbed: u_0 = t  (satisfies u''=0, u(0)=0, u(1)=1)
    """

    def setup_method(self):
        eq = ODE(
            "u'' + eps*u",
            dependent="u", small_param="eps", independent="t",
            conditions=["u(0) = 0", "u(1) = 1"],
        )
        self.sol = eq.expand_regular(order=2)
        self.eps = self.sol.small_param

    def test_problem_type(self):
        assert self.sol._problem_type == "bvp"

    def test_leading_order(self):
        """u_0 = t"""
        assert simplify(self.sol[0].particular_solution - t) == 0

    def test_order0_bcs(self):
        u0 = self.sol[0].particular_solution
        assert simplify(u0.subs(t, 0)) == 0
        assert simplify(u0.subs(t, 1) - 1) == 0

    def test_order1_bcs(self):
        """Higher-order solutions satisfy homogeneous BCs."""
        u1 = self.sol[1].particular_solution
        assert simplify(u1.subs(t, 0)) == 0
        assert simplify(u1.subs(t, 1)) == 0

    def test_order2_bcs(self):
        u2 = self.sol[2].particular_solution
        assert simplify(u2.subs(t, 0)) == 0
        assert simplify(u2.subs(t, 1)) == 0

    def test_no_secular_terms(self):
        assert not any(e.secular for e in self.sol.entries)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
