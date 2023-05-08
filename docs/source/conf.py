# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

# import importlib.metadata

import sphinx_rtd_theme  # type: ignore

sys.path.insert(0, os.path.abspath(".."))


project = "tradeforce"
# Get current version from pyproject.toml
# version = importlib.metadata.version(project)
# TODO: quick workaround for readthedocs
version = "0.0.1"
copyright = "2023, cyclux"
author = "cyclux"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

templates_path = ["_templates"]
exclude_patterns: list = []

html_static_path = ["_static"]
