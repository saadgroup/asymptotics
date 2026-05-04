"""
Tests for regular perturbation — algebraic problems.

All inputs are strings — no symbols() needed.
We verify against known exact solutions.
"""

import pytest
from sympy import sqrt, Rational, series, simplify, symbols, E

from asymptotics import AlgebraicEquation

# We use h.small_param for verification so the symbol name always matches


class TestCubic:
    """x**3 + eps*x - 1 = 0  — hand-verified results."""

    def setup_method(self):
        eq   = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
        self.h = eq.expand_regular(order=3)

    def test_order0(self):
        assert self.h[0].solution == 1

    def test_order1(self):
        assert self.h[1].solution == Rational(-1, 3)

    def test_order2(self):
        assert self.h[2].solution == 0

    def test_order3(self):
        assert self.h[3].solution == Rational(1, 81)

    def test_composite(self):
        e = self.h.small_param
        diff = simplify(self.h.composite - (1 - e/3 + e**3/81))
        assert diff == 0

    def test_hierarchy_length(self):
        assert len(self.h) == 4

    def test_solutions_dict(self):
        vals = list(self.h.solutions.values())
        assert vals[0] == 1
        assert vals[1] == Rational(-1, 3)

    def test_equations_stored(self):
        for entry in self.h.entries:
            assert entry.equation is not None

    def test_collected_stored(self):
        assert len(self.h.collected) == 4


class TestQuadratic:
    """x**2 + eps*x - 1 = 0 — verified against exact series."""

    def setup_method(self):
        eq   = AlgebraicEquation("x**2 + eps*x - 1", dependent="x", small_param="eps")
        self.h = eq.expand_regular(order=3)

    def test_order0(self):
        assert self.h[0].solution == 1

    def test_matches_exact_series(self):
        e            = self.h.small_param
        exact        = (-e + sqrt(e**2 + 4)) / 2
        exact_series = series(exact, e, 0, 4).removeO()
        diff = simplify(self.h.composite - exact_series)
        assert diff == 0

    def test_hierarchy_length(self):
        assert len(self.h) == 4


class TestQuadraticNegativeRoot:
    """Follow the negative root using root_hint."""

    def setup_method(self):
        eq   = AlgebraicEquation(
            "x**2 + eps*x - 1",
            dependent   = "x",
            small_param = "eps",
            root_hint   = -1,
        )
        self.h = eq.expand_regular(order=2)

    def test_order0_negative_root(self):
        assert self.h[0].solution == -1

    def test_matches_negative_exact_series(self):
        e            = self.h.small_param
        exact        = (-e - sqrt(e**2 + 4)) / 2
        exact_series = series(exact, e, 0, 3).removeO()
        diff = simplify(self.h.composite - exact_series)
        assert diff == 0


class TestLinear:
    """x - 1 + eps = 0 — exact solution terminates at order 1."""

    def setup_method(self):
        eq   = AlgebraicEquation("x - 1 + eps", dependent="x", small_param="eps")
        self.h = eq.expand_regular(order=2)

    def test_order0(self):
        assert self.h[0].solution == 1

    def test_order1(self):
        assert self.h[1].solution == -1

    def test_order2_zero(self):
        assert self.h[2].solution == 0

    def test_composite(self):
        e    = self.h.small_param
        diff = simplify(self.h.composite - (1 - e))
        assert diff == 0


class TestTranscendental:
    """x*log(x) - eps = 0 — transcendental, x0=1."""

    def setup_method(self):
        eq   = AlgebraicEquation("x*log(x) - eps", dependent="x", small_param="eps")
        self.h = eq.expand_regular(order=2)

    def test_order0(self):
        assert self.h[0].solution == 1

    def test_hierarchy_length(self):
        assert len(self.h) == 3


class TestBookkeeping:
    """Intermediate objects are accessible."""

    def setup_method(self):
        eq   = AlgebraicEquation("x**3 + eps*x - 1", dependent="x", small_param="eps")
        self.h = eq.expand_regular(order=2)

    def test_substituted_equation_exists(self):
        assert self.h.substituted_equation is not None

    def test_collected_has_right_keys(self):
        assert set(self.h.collected.keys()) == {0, 1, 2}

    def test_each_entry_has_symbol(self):
        for e in self.h.entries:
            assert str(e.symbol).startswith('x_')

    def test_solutions_property(self):
        sols = self.h.solutions
        assert all(str(k).startswith('x_') for k in sols.keys())


class TestStringParsing:
    """Verify the string parsing handles various equation forms."""

    def test_trig_equation(self):
        eq = AlgebraicEquation("tan(x) - 1 - eps*x**2", dependent="x", small_param="eps")
        h  = eq.expand_regular(order=2)
        from sympy import pi
        assert simplify(h[0].solution - pi/4) == 0

    def test_exp_equation(self):
        eq = AlgebraicEquation("x - exp(-eps*x)", dependent="x", small_param="eps")
        h  = eq.expand_regular(order=2)
        assert h[0].solution == 1

    def test_log_equation(self):
        eq = AlgebraicEquation("log(x) + eps*x - 1", dependent="x", small_param="eps")
        h  = eq.expand_regular(order=2)
        assert h[0].solution == E

    def test_bad_equation_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            AlgebraicEquation("x^3 + eps*x - 1", dependent="x", small_param="eps")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
