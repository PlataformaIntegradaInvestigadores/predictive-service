import pytest
from fastapi.testclient import TestClient

from app.main import OPENAPI_TAGS, app

client = TestClient(app)
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


@pytest.fixture(scope="module")
def schema():
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    return response.json()


def _operations(schema):
    for path, item in schema["paths"].items():
        for method, operation in item.items():
            if method in HTTP_METHODS:
                yield path, method, operation


def test_info_describes_the_service(schema):
    info = schema["info"]
    assert info["version"] == "1.0.0"
    assert info["summary"]
    assert "analítica predictiva" in info["description"]


def test_every_operation_is_documented(schema):
    for path, method, operation in _operations(schema):
        label = f"{method.upper()} {path}"
        assert operation.get("summary"), f"{label} sin summary"
        assert operation.get("description"), f"{label} sin descripción"
        assert operation.get("tags"), f"{label} sin tag"


def test_every_operation_declares_a_json_response_schema(schema):
    for path, method, operation in _operations(schema):
        content = operation["responses"]["200"]["content"]["application/json"]
        assert content.get("schema") not in (None, {}), f"{method.upper()} {path}"


def test_tags_are_declared_with_a_description(schema):
    declared = {tag["name"] for tag in schema["tags"]}
    assert declared == {tag["name"] for tag in OPENAPI_TAGS}
    assert all(tag["description"] for tag in schema["tags"])
    used = {tag for _, _, op in _operations(schema) for tag in op["tags"]}
    assert used <= declared


def test_every_schema_property_has_a_description(schema):
    for name, definition in schema["components"]["schemas"].items():
        if name in {"HTTPValidationError", "ValidationError"}:
            continue
        for prop, spec in definition.get("properties", {}).items():
            assert spec.get("description"), f"{name}.{prop} sin descripción"
