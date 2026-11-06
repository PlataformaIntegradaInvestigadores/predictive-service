from fastapi.testclient import TestClient

from app.core.config import Settings, settings
from app.main import app

client = TestClient(app)


def _preflight(origin):
    return client.options(
        "/api/v1/affiliations",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


def test_every_configured_origin_is_allowed():
    for origin in settings.BACKEND_CORS_ORIGINS:
        response = _preflight(origin)
        assert response.headers.get("access-control-allow-origin") == origin


def test_the_local_front_is_allowed_by_default():
    # The default, not the value a local .env may override.
    defaults = Settings.model_fields["BACKEND_CORS_ORIGINS"].default

    assert "http://localhost:4200" in defaults


def test_an_unlisted_origin_is_rejected():
    response = _preflight("https://evil.example.com")

    assert "access-control-allow-origin" not in response.headers
