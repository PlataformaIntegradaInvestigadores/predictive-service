import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (run with -m integration)",
    )
    config.addinivalue_line(
        "markers",
        "real_assets: lifespan builds the real model and CSV data (slow, no fakes)",
    )


@pytest.fixture(autouse=True)
def light_lifespan(request):
    """Every TestClient runs the app lifespan; keep it from loading the 2.7k
    affiliation model and building the GRS on each test. Tests that exercise the
    real wiring opt out with @pytest.mark.real_assets."""
    if request.node.get_closest_marker("real_assets"):
        yield
        return

    from tests.fakes import make_prediction_service

    with (
        patch("app.main.build_prediction_service", make_prediction_service),
        patch("app.main.build_recommendation_service", return_value=MagicMock()),
    ):
        yield


data_dir = tempfile.mkdtemp(prefix="centinela-test-data-")
os.environ["CENTINELA_DATA_PATH"] = data_dir
# Each TestClient runs the lifespan; do not start the minutes-long warm-up in tests.
os.environ["GRS_PRECOMPUTE_ON_STARTUP"] = "false"

pd.DataFrame(
    {
        "scopus_id": [f"a{i}" for i in range(200)],
        "publication_date": ["2020-01-01"] * 200,
    }
).to_csv(os.path.join(data_dir, "articulos_ecuador_CLEAN.csv"), index=False)

pd.DataFrame(
    {
        "scopus_id": [f"a{i}" for i in range(200)],
        "authid": [f"u{i % 10 + 1}" for i in range(200)],
    }
).to_csv(os.path.join(data_dir, "autor_articulo_CLEAN.csv"), index=False)

pd.DataFrame(
    {
        "article_id": [f"a{i}" for i in range(2000)],
        "topic": [f"t{i % 20 + 1}" for i in range(2000)],
    }
).to_csv(os.path.join(data_dir, "topic_article_CLEAN.csv"), index=False)
