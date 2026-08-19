"""
Tests for the step-by-step (stepwise) expansion API, focusing on the
set_solution() residual check that verifies a supplied solution against the
order-k equation.
"""

import pytest
from asymptotics import ODE


def _duffing_stepwise():
    eq = ODE("u'' + u + eps*u**3", dependent="u", small_param="eps",
             independent="t", conditions=["u(0)=1", "u'(0)=0"])
    return eq.begin_expansion(order=1)


class TestSetSolutionResidualCheck:
    def test_residual_zero_for_true_solution(self):
        sol = _duffing_stepwise()
        # cos(t) solves the leading equation u0'' + u0 = 0
        assert sol[0].residual("cos(t)") == 0

    def test_residual_nonzero_for_wrong_solution(self):
        sol = _duffing_stepwise()
        # t does NOT solve u0'' + u0 = 0  (residual = t)
        assert sol[0].residual("t") != 0

    def test_set_solution_accepts_true_solution(self):
        sol = _duffing_stepwise()
        sol[0].set_solution("cos(t)")          # must not raise
        assert sol[0].is_solved

    def test_set_solution_rejects_wrong_solution(self):
        sol = _duffing_stepwise()
        with pytest.raises(ValueError, match="does not satisfy"):
            sol[0].set_solution("t")

    def test_check_false_bypasses_verification(self):
        sol = _duffing_stepwise()
        sol[0].set_solution("t", check=False)  # explicitly bypass
        assert sol[0].is_solved


class TestPorousEigenvalueConsistency:
    """The porous-channel example: F0 = sin(pi y/2) only solves the leading
    equation when the leading eigenvalue lambda0 = -pi^2/4 is present."""

    def test_leading_eigenvalue_makes_F0_a_solution(self):
        eq = ODE("eps*F''' + F*F'' - F'**2 + pi**2/4", dependent="F",
                 small_param="eps", independent="y",
                 conditions=["F(0)=0", "F'(1)=0", "F(1)=1"])
        sol = eq.begin_expansion(order=1)
        assert sol[0].residual("sin(pi*y/2)") == 0
        sol[0].set_solution("sin(pi*y/2)")     # must not raise
        assert sol[0].is_solved

    def test_missing_eigenvalue_is_rejected(self):
        # Without lambda0, F0 = sin(pi y/2) leaves residual -pi^2/4.
        eq = ODE("eps*F''' + F*F'' - F'**2", dependent="F",
                 small_param="eps", independent="y",
                 conditions=["F(0)=0", "F'(1)=0", "F(1)=1"])
        sol = eq.begin_expansion(order=1)
        assert sol[0].residual("sin(pi*y/2)") != 0
        with pytest.raises(ValueError):
            sol[0].set_solution("sin(pi*y/2)")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
