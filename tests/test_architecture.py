"""Dependency rule of the hexagon: the core never imports the outside.

domain       -> only the standard library and the data stack (pandas/numpy/scipy)
application  -> domain and the standard library / pandas, never frameworks or adapters
adapters     -> may import application and domain, never the HTTP layer
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"

FORBIDDEN = {
    "domain": (
        "fastapi",
        "pydantic",
        "joblib",
        "app.api",
        "app.adapters",
        "app.application",
        "app.core",
        "app.main",
        "app.bootstrap",
    ),
    "application": (
        "fastapi",
        "pydantic",
        "joblib",
        "app.api",
        "app.adapters",
        "app.core",
        "app.main",
        "app.bootstrap",
    ),
    "adapters": ("fastapi", "app.api", "app.main", "app.bootstrap"),
}


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _violations(layer: str):
    found = []
    for path in sorted((APP / layer).rglob("*.py")):
        for module in _imports(path):
            if any(
                module == bad or module.startswith(bad + ".")
                for bad in FORBIDDEN[layer]
            ):
                found.append(f"{path.relative_to(APP)} imports {module}")
    return found


def test_domain_is_framework_free():
    assert _violations("domain") == []


def test_application_does_not_know_frameworks_or_adapters():
    assert _violations("application") == []


def test_adapters_do_not_reach_into_the_http_layer():
    assert _violations("adapters") == []
