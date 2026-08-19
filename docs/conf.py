# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# Make the package importable both locally and on Read the Docs.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "asymptotics"
author = "Tony Saad"
copyright = "2026, Tony Saad"

try:
    from asymptotics import __version__ as release
except Exception:  # pragma: no cover - fallback if import fails during build
    release = "0.2.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Keep the build resilient: never fail because a heavy dependency is slow to
# import. (All runtime deps are installed on RTD via pyproject.)
autodoc_mock_imports = []

# -- Napoleon (NumPy-style docstrings) --------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_use_param = True

# -- MyST (Markdown) ---------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",   # $...$ and $$...$$ math in Markdown pages
    "amsmath",      # \begin{align} ... in Markdown pages
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "asymptotics"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
