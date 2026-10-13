from pathlib import Path
import tomllib

metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
project = "GLANCE"
author = "ylt"
copyright = "2026, ylt"
release = metadata["project"]["version"]
extensions = ["sphinx.ext.mathjax", "sphinx_copybutton", "myst_nb", "sphinx_design"]
master_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "build", "**/.ipynb_checkpoints", "**/*.orig"]
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 3
pygments_style = "tango"
pygments_dark_style = "monokai"
html_theme = "furo"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["css/site.css"]
html_show_sphinx = False
html_show_sourcelink = False
html_context = {"github_repository_url": "https://github.com/yilitai/GLANCE"}
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "img/logo.svg",
    "dark_logo": "img/logo.svg",
    "top_of_page_buttons": ["view"],
    "light_css_variables": {
        "color-brand-primary": "#5c7c9b",
        "color-brand-content": "#5c7c9b",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}
