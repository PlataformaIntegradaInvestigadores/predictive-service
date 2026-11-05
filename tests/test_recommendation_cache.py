"""Memoization of per-group recommendations and the startup warm-up.

Scoring every topic for a group is pure Python (~0.2 s per group, 500+ groups),
so /groups and /metrics used to take tens of seconds to minutes on every call.
"""

import threading
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.grs.grs_production import GroupRecommendationSystem
from app.api.v1.endpoints import analytics
from app.main import _warm_up, lifespan
from app.services.recommendation_service import RecommendationService


def _grs() -> GroupRecommendationSystem:
    groups = {
        0: {
            "members": ["u1", "u2"],
            "n_members": 2,
            "papers_total": 5,
            "topics_p1": {"t1"},
            "topics_p2": {"t2"},
        },
        1: {
            "members": ["u3", "u4"],
            "n_members": 2,
            "papers_total": 4,
            "topics_p1": {"t2"},
            "topics_p2": {"t3"},
        },
    }
    return GroupRecommendationSystem(
        groups,
        {"t1", "t2", "t3", "t4", "t5"},
        {"t1": 10, "t2": 5, "t3": 2, "t4": 1, "t5": 1},
        {"t1": 5, "t2": 3},
        {"t1": 5, "t2": 2, "t3": 2, "t4": 1, "t5": 1},
    )


def test_second_call_is_served_from_cache():
    grs = _grs()

    with patch.object(
        grs, "_compute_recommendations", wraps=grs._compute_recommendations
    ) as compute:
        first = grs.recommend_for_group(0, k=3)
        second = grs.recommend_for_group(0, k=3)

    assert compute.call_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_cache_returns_independent_copies():
    grs = _grs()

    first = grs.recommend_for_group(0, k=3)
    first["score"] = -1.0
    first.drop(first.index, inplace=True)

    again = grs.recommend_for_group(0, k=3)

    assert len(again) > 0
    assert (again["score"] > 0).all()


def test_cache_key_includes_group_k_and_weights():
    grs = _grs()

    with patch.object(
        grs, "_compute_recommendations", wraps=grs._compute_recommendations
    ) as compute:
        grs.recommend_for_group(0, k=3)
        grs.recommend_for_group(1, k=3)
        grs.recommend_for_group(0, k=2)
        grs.recommend_for_group(0, k=3, alpha=0.5)
        grs.recommend_for_group(0, k=3)

    assert compute.call_count == 4


def test_generate_all_recommendations_reuses_the_cache():
    grs = _grs()
    grs.generate_all_recommendations(k=3)

    with patch.object(
        grs, "_compute_recommendations", wraps=grs._compute_recommendations
    ) as compute:
        again = grs.generate_all_recommendations(k=3)
        single = grs.recommend_for_group(0, k=3)

    assert compute.call_count == 0
    assert set(again["group_id"]) == {0, 1}
    assert len(single) > 0


def test_cached_result_is_identical_to_uncached_computation():
    grs = _grs()

    cached = grs.recommend_for_group(0, k=3)
    direct = grs._compute_recommendations(0, k=3)

    pd.testing.assert_frame_equal(cached, direct)


def test_concurrent_calls_compute_once():
    grs = _grs()
    start = threading.Barrier(4)
    results = []

    with patch.object(
        grs, "_compute_recommendations", wraps=grs._compute_recommendations
    ) as compute:

        def worker():
            start.wait()
            results.append(grs.recommend_for_group(0, k=3))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert compute.call_count == 1
    assert len(results) == 4


def test_warm_up_computes_metrics_for_the_requested_k():
    with (
        patch.object(RecommendationService, "_load_data"),
        patch.object(RecommendationService, "_initialize_grs"),
    ):
        service = RecommendationService.__new__(RecommendationService)
    service.groups = {0: {}, 1: {}}
    service.get_metrics = MagicMock()

    service.warm_up(k=7)

    service.get_metrics.assert_called_once_with(k=7)


def test_warm_up_helper_swallows_errors():
    service = MagicMock()
    service.warm_up.side_effect = RuntimeError("boom")

    _warm_up(service)

    service.warm_up.assert_called_once()


def test_lifespan_starts_warm_up_thread_when_enabled():
    service = MagicMock()
    started = threading.Event()
    service.warm_up.side_effect = lambda k: started.set()

    with (
        patch("app.main.RecommendationService", return_value=service),
        patch("app.main.settings.GRS_PRECOMPUTE_ON_STARTUP", True),
        patch("app.main.settings.GRS_PRECOMPUTE_K", 4),
    ):
        with TestClient(FastAPI(lifespan=lifespan)):
            assert started.wait(timeout=5)

    service.warm_up.assert_called_once_with(k=4)


def test_lifespan_skips_warm_up_when_disabled():
    service = MagicMock()

    with (
        patch("app.main.RecommendationService", return_value=service),
        patch("app.main.settings.GRS_PRECOMPUTE_ON_STARTUP", False),
    ):
        with TestClient(FastAPI(lifespan=lifespan)):
            pass

    service.warm_up.assert_not_called()


def test_warm_up_helper_warms_ranking_before_recommendations():
    order = []
    service = MagicMock()
    service.warm_up.side_effect = lambda k: order.append("grs")

    with patch.object(analytics, "prediction_service") as prediction:
        prediction.check_assets_loaded.return_value = True
        prediction.get_ranking.side_effect = lambda: order.append("ranking")

        _warm_up(service)

    assert order == ["ranking", "grs"]


def test_warm_up_helper_skips_ranking_when_assets_are_missing():
    service = MagicMock()

    with patch.object(analytics, "prediction_service") as prediction:
        prediction.check_assets_loaded.return_value = False

        _warm_up(service)

    prediction.get_ranking.assert_not_called()
    service.warm_up.assert_called_once()


def test_ranking_failure_does_not_block_recommendation_warm_up():
    service = MagicMock()

    with patch.object(analytics, "prediction_service") as prediction:
        prediction.check_assets_loaded.return_value = True
        prediction.get_ranking.side_effect = RuntimeError("boom")

        _warm_up(service)

    service.warm_up.assert_called_once()
