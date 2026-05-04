"""
Tests for asymptotics error checking.
All inputs use the string-based API.
"""

import pytest
from sympy import symbols, cos
from asymptotics import (
    AlgebraicEquation,
    NoSmallParameterError,
    NoLeadingOrderSolutionError,
    OnlyComplexRootsError,
)

eps = symbols('epsilon')


class TestNoSmallParameter:
    """Equation does not contain eps."""

    def test_pure_polynomial(self):
        eq = AlgebraicEquation("x**3 - 1", dependent="x", small_param="eps")
        with pytest.raises(NoSmallParameterError):
            eq.expand_regular(order=2)

    def test_constant_equation(self):
        eq = AlgebraicEquation("x - 1", dependent="x", small_param="eps")
        with pytest.raises(NoSmallParameterError):
            eq.expand_regular(order=2)

    def test_error_message_mentions_param(self):
        eq = AlgebraicEquation("x**2 - 4", dependent="x", small_param="eps")
        with pytest.raises(NoSmallParameterError) as exc:
            eq.expand_regular(order=2)
        assert "eps" in str(exc.value)

    def test_error_message_shows_equation(self):
        eq = AlgebraicEquation("x**2 - 4", dependent="x", small_param="eps")
        with pytest.raises(NoSmallParameterError) as exc:
            eq.expand_regular(order=2)
        assert "x" in str(exc.value)


class TestNoLeadingOrderSolution:
    """SymPy cannot solve the O(1) equation symbolically."""

    def test_cosine_fixed_point(self):
        eq = AlgebraicEquation("x - cos(x) - eps", dependent="x", small_param="eps")
        with pytest.raises(NoLeadingOrderSolutionError) as exc:
            eq.expand_regular(order=2)
        msg = str(exc.value)
        assert "leading-order" in msg or "O(1)" in msg

    def test_error_message_has_hints(self):
        eq = AlgebraicEquation("x - cos(x) - eps", dependent="x", small_param="eps")
        with pytest.raises(NoLeadingOrderSolutionError) as exc:
            eq.expand_regular(order=2)
        assert "root_hint" in str(exc.value)


class TestOnlyComplexRoots:
    """O(1) equation has only complex roots."""

    def test_x_squared_plus_one(self):
        eq = AlgebraicEquation("x**2 + 1 + eps*x", dependent="x", small_param="eps")
        with pytest.raises(OnlyComplexRootsError) as exc:
            eq.expand_regular(order=2)
        assert "complex" in str(exc.value)

    def test_error_message_shows_roots(self):
        eq = AlgebraicEquation("x**2 + 1 + eps*x", dependent="x", small_param="eps")
        with pytest.raises(OnlyComplexRootsError) as exc:
            eq.expand_regular(order=2)
        assert exc.value.roots

    def test_with_root_hint_succeeds(self):
        import sympy as sp
        eq = AlgebraicEquation(
            "x**2 + 1 + eps*x",
            dependent   = "x",
            small_param = "eps",
            root_hint   = sp.I,
        )
        h = eq.expand_regular(order=2)
        assert h[0].solution == sp.I


class TestStringParseBadInput:
    """Bad string inputs raise ValueError with helpful messages."""

    def test_caret_power_raises(self):
        """x^3 is not valid — must use x**3."""
        with pytest.raises(ValueError, match="Could not parse"):
            AlgebraicEquation("x^3 + eps*x - 1", dependent="x", small_param="eps")

    def test_error_message_has_tips(self):
        with pytest.raises(ValueError) as exc:
            AlgebraicEquation("x^3 + eps*x - 1", dependent="x", small_param="eps")
        assert "**" in str(exc.value)


class TestValidEquationsStillWork:
    """Regression — good equations still pass."""

    def test_cubic(self):
        from sympy import Rational
        eq = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
        h  = eq.expand_regular(order=3)
        assert h[0].solution == 1
        assert h[1].solution == Rational(-1, 3)
        assert h[2].solution == 0
        assert h[3].solution == Rational(1, 81)

    def test_log_equation(self):
        from sympy import E
        eq = AlgebraicEquation("log(x) + eps*x - 1", dependent="x", small_param="eps")
        h  = eq.expand_regular(order=2)
        assert h[0].solution == E


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
