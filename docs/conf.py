import os
import sys
sys.path.insert(0, os.path.abspath('../pymead'))
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
    'sphinx.ext.automodule',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

html_theme = 'sphinx_rtd_theme'
