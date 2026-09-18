import os
import tempfile

import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (run with -m integration)",
    )


data_dir = tempfile.mkdtemp(prefix="centinela-test-data-")
os.environ["CENTINELA_DATA_PATH"] = data_dir

pd.DataFrame(
    {"scopus_id": [f"a{i}" for i in range(200)], "publication_date": ["2020-01-01"] * 200}
).to_csv(os.path.join(data_dir, "articulos_ecuador_CLEAN.csv"), index=False)

pd.DataFrame(
    {"scopus_id": [f"a{i}" for i in range(200)], "authid": [f"u{i % 10 + 1}" for i in range(200)]}
).to_csv(os.path.join(data_dir, "autor_articulo_CLEAN.csv"), index=False)

pd.DataFrame(
    {"article_id": [f"a{i}" for i in range(2000)], "topic": [f"t{i % 20 + 1}" for i in range(2000)]}
).to_csv(os.path.join(data_dir, "topic_article_CLEAN.csv"), index=False)
