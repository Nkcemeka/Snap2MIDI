# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'Snap2MIDI'
copyright = '2025, Chukwuemeka L. Nkama'
author = 'Chukwuemeka L. Nkama'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "numpydoc",
    'sphinx.ext.napoleon',
]

# Disable numpydoc auto-toctree for class members to avoid stub conflicts
numpydoc_class_members_toctree = False

templates_path = ['_templates']
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}


# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}
html_sidebars = {
    '**': [
        'globaltoc.html',  # shows the full, global TOC from the master toctree
        'localtoc.html',   # also shows the local toctree (optional)
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}
