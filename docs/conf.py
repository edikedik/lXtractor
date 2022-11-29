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
copyright = '2022, iReveguk'
author = 'iReveguk'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    # 'sphinx.ext.napoleon',
    'nbsphinx'
]

# To include __init__ docs
autoclass_content = 'class'

autodoc_member_order = 'groupwise'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented_params'

autodoc_default_options = {
    # 'members': 'var1, var2',
    'member-order': 'groupwise',
    'special-members': '__init__, __call__',
    'undoc-members': True,
    # 'exclude-members': '__weakref__'
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'readable'
# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_static_path = ['_static']

# Required theme setup
html_theme = 'sphinx_material'

# Set link name generated in the top bar.
html_title = 'lXtractor'

# Material theme options (see theme.conf for more information)
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'lXtractor',

    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    # 'base_url': 'https://project.github.io/project',

    # Set the color and the accent color
    'color_primary': 'blue',
    'color_accent': 'light-green',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/edikedik/lXtractor/',
    'repo_name': 'lXtractor',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 4,
    # If False, expand all TOC entries
    'globaltoc_collapse': False,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': False,
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
