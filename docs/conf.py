# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime
from pathlib import Path
# from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../src'))

# # Mock dependencies that may not be available during doc build
# class Mock(MagicMock):
#     @classmethod
#     def __getattr__(cls, name):
#         return MagicMock()

# MOCK_MODULES = ['numpy', 'numpy.typing', 'open3d', 'open3d.geometry', 
#                 'open3d.utility', 'open3d.visualization']
# sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# -- Project information from pyproject.toml ---------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Read project metadata from pyproject.toml
try:
    import tomli
except ModuleNotFoundError:
    # tomli is built into Python 3.11+ as tomllib
    import tomllib as tomli

pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
with open(pyproject_path, 'rb') as f:
    pyproject_data = tomli.load(f)

project_metadata = pyproject_data['project']
project = project_metadata['name'].title()  # 'registration' -> 'Registration'
release = project_metadata['version']
version = release  # Short version

# Extract description and author if available
description = project_metadata.get('description', '')
author = ''

# Dynamic copyright with current year
copyright = f'{datetime.now().year}, {author}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "sphinx_autodoc_typehints",
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
]

mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary settings
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}
