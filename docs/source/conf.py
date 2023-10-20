import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# Project Information

project = 'pymead'
copyright = '2022-2023, Matthew G. Lauer'
author = 'Matthew G. Lauer'

release = '2.0'
version = '2.0.0'

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autosectionlabel'
]

autodoc_mock_imports = ['PyQt5']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

templates_path = ['_templates']

html_theme = 'pydata_sphinx_theme'

html_context = {
    'display_github': True,
    'github_user': 'mlau154',
    'github_repo': 'pymead',
    'github_version': 'master',
}

add_module_names = False

numfig = True

navigation_depth = 2  # For the table of contents

html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

html_logo = "_static/pymead-logo.png"

autosectionlabel_prefix_document = False
