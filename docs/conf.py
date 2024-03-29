# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from lXtractor.__about__ import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lXtractor'
copyright = '2023, iReveguk'
author = 'iReveguk'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['ref.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

# To include __init__ docs
autoclass_content = 'class'

autodoc_member_order = 'groupwise'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented_params'

autodoc_default_options = {
    'member-order': 'groupwise',
    'special-members': '__init__, __call__',
    'undoc-members': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

html_static_path = ['_static']

# Required theme setup
html_theme = "sphinx_rtd_theme"

# Set link name generated in the top bar.
html_title = 'lXtractor'

html_show_sourcelink = True
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    'nav_title': 'lXtractor documentation',
    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',
    # Set the color and the accent color
    'color_primary': 'blue',
    'color_accent': 'cyan',
    'html_minify': False,
    'html_prettify': True,
    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/edikedik/lXtractor/',
    'repo_name': 'lXtractor',
    'repo_type': 'github',
    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 3,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,
}

todo_include_todos = True
