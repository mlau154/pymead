import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# Project Information

project = 'pymead'
copyright = '2022, Matthew G. Lauer'
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
]

autodoc_mock_imports = ['PyQt5']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

templates_path = ['_templates']

html_theme = 'pydata_sphinx_theme'
