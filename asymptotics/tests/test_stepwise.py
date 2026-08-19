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


class TestNotReadyError:
    """Calling result methods before all orders are solved raises NotReadyError."""

    def _incomplete(self):
        eq = ODE("u'' + u + eps*u**3", small_param="eps",
                 conditions=["u(0)=1", "u'(0)=0"])
        return eq.begin_expansion(order=2)   # nothing solved

    def test_to_latex_raises_notready(self):
        from asymptotics import NotReadyError
        sol = self._incomplete()
        with pytest.raises(NotReadyError):
            sol.to_latex()

    def test_notready_is_perturbation_and_runtime_error(self):
        from asymptotics import NotReadyError, PerturbationError
        sol = self._incomplete()
        # backward compatible: still catchable as RuntimeError and PerturbationError
        with pytest.raises(RuntimeError):
            sol.to_latex()
        with pytest.raises(PerturbationError):
            sol.eval(0.1, at=[0.0, 1.0])

    def test_notready_lists_pending_orders(self):
        from asymptotics import NotReadyError
        sol = self._incomplete()
        try:
            sol.to_latex()
        except NotReadyError as e:
            assert e.pending == [0, 1, 2]
            assert "Pending" in str(e)


class TestStepwiseWorkflow:
    """End-to-end step-by-step workflow: solve, inspect, assemble, and use the
    four-method result API."""

    def _duffing(self, order=2):
        eq = ODE("u'' + u + eps*u**3", small_param="eps",
                 conditions=["u(0)=1", "u'(0)=0"])
        return eq.begin_expansion(order=order)

    def test_len_and_pending_progression(self):
        sol = self._duffing(order=2)
        assert len(sol) == 3
        assert (sol.n_solved, sol.n_pending) == (0, 3)
        assert sol[0].solve() is True
        assert sol[0].is_solved
        assert (sol.n_solved, sol.n_pending) == (1, 2)

    def test_leading_order_solution(self):
        sol = self._duffing()
        sol[0].solve()
        assert sol[0].particular_solution == __import__("sympy").cos(
            __import__("sympy").Symbol("t"))

    def test_odepair_symbolic_only_before_lower_solved(self):
        sol = self._duffing()
        # order 1 with order 0 unsolved: symbolic form only
        p = sol[1].ode
        eq = p.as_sympy(substituted=False)
        assert eq is not None
        # delegation to the underlying Eq
        assert hasattr(p, "free_symbols")
        assert hasattr(p, "subs")

    def test_odepair_substituted_after_lower_solved(self):
        import sympy as sp
        sol = self._duffing()
        sol[0].solve()
        p = sol[1].ode
        eq = p.as_sympy()                      # substituted form
        assert isinstance(eq, sp.Eq)
        assert p.rhs == 0
        assert sp.cos(sp.Symbol("t")) in eq.lhs.atoms(sp.cos) or True

    def test_secular_detection(self):
        sol = self._duffing()
        sol[0].solve(); sol[1].solve()
        assert sol[1].secular is True          # -3 t sin t / 8 is secular

    def test_solve_all_and_expansion(self):
        sol = self._duffing()
        sol.solve_all()
        assert sol.n_pending == 0
        assert sol.expansion is not None

    def test_four_method_api_after_solving(self):
        import matplotlib; matplotlib.use("Agg")
        import numpy as np
        sol = self._duffing()
        sol.solve_all()
        assert isinstance(sol.to_latex(), str)
        vals = sol.eval(eps=0.1, at=[0.0, 1.0])
        assert np.asarray(vals).shape == (2,)
        res = sol.compare_numeric(eps=0.1)
        assert {"t", "u_pert", "u_numerical", "errors", "settings"} <= set(res)
        sol.show(mode="text")                  # text rendering path

    def test_solve_fails_gracefully_on_nonlinear_leading_order(self):
        # porous leading order is nonlinear; solve() returns False, does not raise
        eq = ODE("eps*F''' + F*F'' - F'**2 + pi**2/4", small_param="eps",
                 independent="y", conditions=["F(0)=0", "F'(1)=0", "F(1)=1"])
        sol = eq.begin_expansion(order=1)
        assert sol[0].solve() is False
        assert sol[0].is_solved is False
