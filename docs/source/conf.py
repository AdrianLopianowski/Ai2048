import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = '2048 ai project'
copyright = '2026, Jakub Studziński, Aleksander Bojko, Sergiusz Aniśko, Adrian Łopianowski'
author = 'Jakub Studziński, Aleksander Bojko, Sergiusz Aniśko, Adrian Łopianowski'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      
    'sphinx.ext.viewcode',     
    'sphinx.ext.napoleon',     
]

templates_path = ['_templates']

exclude_patterns = []

language = 'pl'


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']