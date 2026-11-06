"""The real wiring: adapters + use cases built from the files in the repo."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.bootstrap import build_prediction_service, build_recommendation_service
from app.main import lifespan

pytestmark = pytest.mark.real_assets


def test_prediction_service_is_built_from_the_pickled_model():
    service = build_prediction_service()

    assert len(service.get_all_affiliations()) > 1000
    assert service.get_model_details()["total_affiliations"] == len(
        service.get_all_affiliations()
    )


def test_recommendation_service_is_built_from_the_configured_data():
    service = build_recommendation_service()

    assert len(service.articles) == 200


def test_lifespan_wires_both_services_with_the_real_assets(monkeypatch):
    monkeypatch.setattr("app.main.settings.GRS_PRECOMPUTE_ON_STARTUP", False)
    test_app = FastAPI(lifespan=lifespan)

    with TestClient(test_app):
        assert test_app.state.prediction_service is not None
        assert test_app.state.recommendation_service is not None
        assert test_app.state.recommendation_service_error is None
