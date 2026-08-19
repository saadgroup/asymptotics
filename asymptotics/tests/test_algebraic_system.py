"""
Tests for regular perturbation — coupled algebraic systems.
"""

import pytest
from sympy import Rational, simplify, symbols

from asymptotics import AlgebraicSystem, SystemHierarchy
from asymptotics.core.exceptions import NoSmallParameterError, OnlyComplexRootsError

eps = symbols('eps')


class TestSymmetricSystem:
    """
    x**2 + eps*y - 1 = 0
    y**2 + eps*x - 1 = 0

    By symmetry x(eps) = y(eps).
    Verified by hand: x0=1, x1=-1/2, x2=1/8, x3=0
    """

    def setup_method(self):
        sys = AlgebraicSystem(
            equations   = ["x**2 + eps*y - 1", "y**2 + eps*x - 1"],
            dependents  = ["x", "y"],
            small_param = "eps",
        )
        self.sol = sys.expand_regular(order=3)

    def test_returns_system_hierarchy(self):
        assert isinstance(self.sol, SystemHierarchy)

    def test_variables(self):
        assert self.sol.variables == ["x", "y"]

    def test_x0(self):
        assert self.sol["x"][0].solution == 1

    def test_y0(self):
        assert self.sol["y"][0].solution == 1

    def test_x1(self):
        assert self.sol["x"][1].solution == Rational(-1, 2)

    def test_y1(self):
        assert self.sol["y"][1].solution == Rational(-1, 2)

    def test_x2(self):
        assert self.sol["x"][2].solution == Rational(1, 8)

    def test_y2(self):
        assert self.sol["y"][2].solution == Rational(1, 8)

    def test_x3(self):
        assert self.sol["x"][3].solution == 0

    def test_symmetry(self):
        """By symmetry x and y expansions must be equal."""
        diff = simplify(self.sol["x"].expansion - self.sol["y"].expansion)
        assert diff == 0

    def test_expansions_correct(self):
        e = self.sol.small_param
        expected = 1 - e/2 + e**2/8
        for var in ["x", "y"]:
            diff = simplify(self.sol[var].expansion - expected)
            assert diff == 0

    def test_hierarchy_length(self):
        assert len(self.sol["x"]) == 4
        assert len(self.sol["y"]) == 4

    def test_getitem_bad_key(self):
        with pytest.raises(KeyError):
            _ = self.sol["z"]


class TestAsymmetricSystem:
    """
    x + eps*y**2 - 1 = 0
    y + eps*x    - 2 = 0

    At eps=0: x0=1, y0=2  (trivially)
    """

    def setup_method(self):
        sys = AlgebraicSystem(
            equations   = ["x + eps*y**2 - 1", "y + eps*x - 2"],
            dependents  = ["x", "y"],
            small_param = "eps",
        )
        self.sol = sys.expand_regular(order=2)

    def test_x0(self):
        assert self.sol["x"][0].solution == 1

    def test_y0(self):
        assert self.sol["y"][0].solution == 2

    def test_x1(self):
        """x + eps*y^2 - 1 = 0: at O(eps), x1 + y0^2 = 0 => x1 = -4"""
        assert self.sol["x"][1].solution == -4

    def test_y1(self):
        """y + eps*x - 2 = 0: at O(eps), y1 + x0 = 0 => y1 = -1"""
        assert self.sol["y"][1].solution == -1


class TestThreeVariables:
    """Three-variable system to verify n > 2 works."""

    def setup_method(self):
        sys = AlgebraicSystem(
            equations   = [
                "x + eps*y - 1",
                "y + eps*z - 2",
                "z + eps*x - 3",
            ],
            dependents  = ["x", "y", "z"],
            small_param = "eps",
        )
        self.sol = sys.expand_regular(order=2)

    def test_leading_order(self):
        assert self.sol["x"][0].solution == 1
        assert self.sol["y"][0].solution == 2
        assert self.sol["z"][0].solution == 3

    def test_variables_list(self):
        assert self.sol.variables == ["x", "y", "z"]

    def test_hierarchy_length(self):
        assert len(self.sol["x"]) == 3


class TestRootHint:
    """root_hint selects a specific solution branch."""

    def test_negative_branch(self):
        """x^2 + eps*y - 1 = 0,  y^2 + eps*x - 1 = 0 has branch x0=y0=-1."""
        sys = AlgebraicSystem(
            equations   = ["x**2 + eps*y - 1", "y**2 + eps*x - 1"],
            dependents  = ["x", "y"],
            small_param = "eps",
            root_hint   = {"x": -1, "y": -1},
        )
        sol = sys.expand_regular(order=2)
        assert sol["x"][0].solution == -1
        assert sol["y"][0].solution == -1


class TestErrorChecking:
    """Error handling for systems."""

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="must match"):
            AlgebraicSystem(
                equations   = ["x + eps - 1", "y + eps - 2"],
                dependents  = ["x"],
                small_param = "eps",
            )

    def test_no_small_param(self):
        sys = AlgebraicSystem(
            equations   = ["x - 1", "y - 2"],
            dependents  = ["x", "y"],
            small_param = "eps",
        )
        with pytest.raises(NoSmallParameterError):
            sys.expand_regular(order=2)

    def test_bad_syntax(self):
        with pytest.raises(ValueError, match="Could not parse"):
            AlgebraicSystem(
                equations   = ["x^2 + eps - 1", "y + eps - 1"],
                dependents  = ["x", "y"],
                small_param = "eps",
            )


class TestSystemToLatex:
    """LaTeX export for a coupled algebraic system (regression: this used to
    raise TypeError because the to_latex dispatcher had no SystemHierarchy
    branch)."""

    def _sol(self, order=3, small_param="eps"):
        p = small_param
        sys = AlgebraicSystem(
            equations   = [f"x**2 + {p}*y - 1", f"y**2 + {p}*x - 1"],
            dependents  = ["x", "y"],
            small_param = p,
        )
        return sys.expand_regular(order=order)

    def test_to_latex_runs_and_returns_string(self):
        src = self._sol().to_latex()
        assert isinstance(src, str)
        assert src.startswith('%')
        # one expansion block per variable
        assert r'x(\varepsilon)' in src
        assert r'y(\varepsilon)' in src
        assert r'\begin{align}' in src

    def test_to_latex_show_orders(self):
        src = self._sol().to_latex(show_orders=True)
        assert 'x_{0}' in src and 'x_{1}' in src
        assert 'y_{0}' in src and 'y_{1}' in src

    def test_to_latex_renders_varepsilon_for_any_symbol(self):
        # a non-standard small-parameter name must still render as \varepsilon
        src = self._sol(order=2, small_param="delta").to_latex()
        assert r'\varepsilon' in src
        assert 'delta' not in src

    def test_to_latex_environment(self):
        src = self._sol().to_latex(environment='gather')
        assert r'\begin{gather}' in src


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
