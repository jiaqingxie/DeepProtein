# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepProtein'
copyright = '2024, jiaqingxie'
author = 'jiaqingxie'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']

source_parsers = {
'.md' : 'recommonmark.parser.CommonMarkParser'
}
source_suffix = ['.rst', '.md']

exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"

html_theme_options = {
   'collapse_navigation': False,
   'display_version': True,
   'logo_only': False,
}

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# html_logo = '../build/html/_images/deeppurpose_pp_logo.png'
html_context = {
    'css_files': [
  	   'https://fonts.googleapis.com/css?family=Raleway',
       '../build/html/_static/css/deepprotein_docs_theme.css'
    ],
}




html_static_path = ['_static']


# The master toctree document.
master_doc = 'index'