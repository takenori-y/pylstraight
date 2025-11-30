# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pylstraight"
copyright = "2025, Takenori Yoshimura"
author = "Takenori Yoshimura"
exec(open(f"../../{project}/version.py").read())
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",
]
templates_path = []
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_with_keys": False,
    "logo": {
        "text": project,
    },
    "switcher": {
        "json_url": f"https://takenori-y.github.io/{project}/switcher.json",
        "version_match": release,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": f"https://github.com/takenori-y/{project}/",
            "icon": "fab fa-github-square",
        },
        {
            "name": "PyPI",
            "url": f"https://pypi.org/project/{project}/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_start": ["navbar-logo", "version-switcher"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}
html_static_path = []
html_show_sourcelink = False
