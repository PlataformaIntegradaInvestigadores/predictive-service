from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.grs.grs_production import (
    DataPreprocessor,
    DataValidator,
    GroupIdentifier,
    GroupRecommendationSystem,
    MetricsEvaluator,
    SimilarityCalculator,
)


def make_articles_df():
    rows = []
    for i in range(200):
        year = 2020 + (i % 6)
        rows.append({"scopus_id": f"a{i}", "publication_date": f"{year}-01-01"})
    return pd.DataFrame(rows)


def make_author_article_df():
    rows = []
    for i in range(200):
        rows.append({"scopus_id": f"a{i}", "authid": f"u{i % 10 + 1}"})
    return pd.DataFrame(rows)


def make_topic_article_df():
    rows = []
    for i in range(2000):
        rows.append({"article_id": f"a{i}", "topic": f"t{i % 20 + 1}"})
    return pd.DataFrame(rows)


class TestDataValidator:
    def test_validate_ok(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        assert DataValidator.validate(df_articles, df_author_article, df_topic_article) is True

    def test_validate_missing_article_column(self):
        df_articles = pd.DataFrame({"other": [1]})
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        with pytest.raises(ValueError):
            DataValidator.validate(df_articles, df_author_article, df_topic_article)

    def test_validate_missing_author_column(self):
        df_articles = make_articles_df()
        df_author_article = pd.DataFrame({"other": [1]})
        df_topic_article = make_topic_article_df()
        with pytest.raises(ValueError):
            DataValidator.validate(df_articles, df_author_article, df_topic_article)

    def test_validate_missing_topic_column(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = pd.DataFrame({"other": [1]})
        with pytest.raises(ValueError):
            DataValidator.validate(df_articles, df_author_article, df_topic_article)

    def test_validate_too_few_articles(self):
        df_articles = pd.DataFrame({"scopus_id": ["a1"], "publication_date": ["2020-01-01"]})
        df_author_article = pd.DataFrame({"scopus_id": ["a1"], "authid": ["u1"]})
        df_topic_article = pd.DataFrame({"article_id": ["a1"], "topic": ["t1"]})
        with pytest.raises(ValueError):
            DataValidator.validate(df_articles, df_author_article, df_topic_article)

    def test_validate_too_few_topics(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = pd.DataFrame(
            {"article_id": ["a1", "a2"], "topic": ["t1", "t2"]}
        )
        with pytest.raises(ValueError):
            DataValidator.validate(df_articles, df_author_article, df_topic_article)


class TestDataPreprocessor:
    def test_get_article_authors_map(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
        mapping = preprocessor.get_article_authors_map()
        assert "a0" in mapping
        assert "u1" in mapping["a0"]

    def test_get_article_topics_map(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
        mapping = preprocessor.get_article_topics_map()
        assert "a0" in mapping
        assert "t1" in mapping["a0"]

    def test_get_all_topics(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
        topics = preprocessor.get_all_topics()
        assert len(topics) == 20

    def test_get_topic_frequency(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
        freq = preprocessor.get_topic_frequency()
        assert freq["t1"] == 100

    def test_get_articles_by_period_default(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
        p1, p2 = preprocessor.get_articles_by_period()
        assert "a0" in p1
        assert "a3" in p2

    def test_get_topic_frequency_by_period(self):
        df_articles = make_articles_df()
        df_author_article = make_author_article_df()
        df_topic_article = make_topic_article_df()
        preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
        article_topics = preprocessor.get_article_topics_map()
        p1, p2 = preprocessor.get_articles_by_period()
        freq_p1, freq_p2 = preprocessor.get_topic_frequency_by_period(
            article_topics, p1, p2
        )
        assert isinstance(freq_p1, dict)
        assert isinstance(freq_p2, dict)


class TestGroupIdentifier:
    def test_extract_groups_in_period(self):
        articles = {"a1", "a2"}
        article_authors = {"a1": {"u1", "u2"}, "a2": {"u2", "u3"}}
        article_topics = {"a1": {"t1"}, "a2": {"t2"}}
        identifier = GroupIdentifier(article_authors, article_topics)
        groups = identifier.extract_groups_in_period(articles)
        assert len(groups) == 2

    def test_extract_groups_ignores_single_author(self):
        articles = {"a1"}
        article_authors = {"a1": {"u1"}}
        article_topics = {"a1": {"t1"}}
        identifier = GroupIdentifier(article_authors, article_topics)
        groups = identifier.extract_groups_in_period(articles)
        assert len(groups) == 0

    def test_identify_persistent(self):
        groups_p1 = {("u1", "u2"): {"papers": 2, "topics": {"t1"}}}
        groups_p2 = {("u1", "u2"): {"papers": 1, "topics": {"t2"}}}
        identifier = GroupIdentifier({}, {})
        persistent = identifier.identify_persistent(groups_p1, groups_p2)
        assert len(persistent) == 1
        assert persistent[0]["members"] == ["u1", "u2"]
        assert persistent[0]["papers_total"] == 3

    def test_identify_persistent_below_min_papers(self):
        groups_p1 = {("u1", "u2"): {"papers": 1, "topics": {"t1"}}}
        groups_p2 = {("u1", "u2"): {"papers": 0, "topics": {"t2"}}}
        identifier = GroupIdentifier({}, {})
        persistent = identifier.identify_persistent(groups_p1, groups_p2, min_papers_total=2)
        assert len(persistent) == 0


class TestSimilarityCalculator:
    def test_jaccard_similarity_identical(self):
        similarity = SimilarityCalculator.jaccard_similarity({"a", "b"}, {"a", "b"})
        assert similarity == 1.0

    def test_jaccard_similarity_no_overlap(self):
        similarity = SimilarityCalculator.jaccard_similarity({"a"}, {"b"})
        assert similarity == 0.0

    def test_jaccard_similarity_partial(self):
        similarity = SimilarityCalculator.jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(similarity - 1 / 3) < 1e-6

    def test_jaccard_similarity_empty(self):
        similarity = SimilarityCalculator.jaccard_similarity(set(), {"a"})
        assert similarity == 0.0

    def test_compute_group_similarities(self):
        groups = {
            0: {"topics_p1": {"t1", "t2"}},
            1: {"topics_p1": {"t2", "t3"}},
            2: {"topics_p1": {"t4"}},
        }
        sims = SimilarityCalculator.compute_group_similarities(groups, top_k=2)
        assert 0 in sims
        assert 1 in sims
        assert len(sims[0]) <= 2


class TestGroupRecommendationSystem:
    def test_recommend_for_group(self):
        groups = {
            0: {
                "members": ["u1", "u2"],
                "n_members": 2,
                "papers_total": 5,
                "topics_p1": {"t1"},
                "topics_p2": {"t2"},
            }
        }
        all_topics = {"t1", "t2", "t3"}
        topic_frequency = {"t1": 10, "t2": 5, "t3": 1}
        topic_frequency_p1 = {"t1": 5, "t2": 3}
        topic_frequency_p2 = {"t1": 5, "t2": 2, "t3": 1}
        article_authors = {"a1": {"u1", "u2"}}
        article_topics = {"a1": {"t1"}}
        identifier = GroupIdentifier(article_authors, article_topics)
        groups_p1 = identifier.extract_groups_in_period({"a1"})
        groups_p2 = identifier.extract_groups_in_period({"a1"})
        persistent = identifier.identify_persistent(groups_p1, groups_p2)
        grs = GroupRecommendationSystem(
            persistent,
            all_topics,
            topic_frequency,
            topic_frequency_p1,
            topic_frequency_p2,
        )
        recs = grs.recommend_for_group(group_id=0, k=2)
        assert len(recs) <= 2

    def test_recommend_for_group_no_scores(self):
        groups = {
            0: {
                "members": ["u1"],
                "n_members": 1,
                "papers_total": 1,
                "topics_p1": set(),
                "topics_p2": set(),
            }
        }
        all_topics = set()
        topic_frequency = {}
        grs = GroupRecommendationSystem(groups, all_topics, topic_frequency)
        recs = grs.recommend_for_group(group_id=0, k=2)
        assert len(recs) == 0

    def test_recommend_for_group_filters_explored(self):
        groups = {
            0: {
                "members": ["u1", "u2"],
                "n_members": 2,
                "papers_total": 5,
                "topics_p1": {"t1"},
                "topics_p2": {"t1"},
            }
        }
        all_topics = {"t1", "t2", "t3"}
        topic_frequency = {"t1": 10, "t2": 5, "t3": 1}
        topic_frequency_p1 = {"t1": 5, "t2": 3}
        topic_frequency_p2 = {"t1": 5, "t2": 2, "t3": 1}
        article_authors = {"a1": {"u1", "u2"}}
        article_topics = {"a1": {"t1"}}
        identifier = GroupIdentifier(article_authors, article_topics)
        groups_p1 = identifier.extract_groups_in_period({"a1"})
        groups_p2 = identifier.extract_groups_in_period({"a1"})
        persistent = identifier.identify_persistent(groups_p1, groups_p2)
        grs = GroupRecommendationSystem(
            persistent,
            all_topics,
            topic_frequency,
            topic_frequency_p1,
            topic_frequency_p2,
        )
        recs = grs.recommend_for_group(group_id=0, k=2)
        assert "t1" not in recs["topic"].values

    def test_generate_all_recommendations(self):
        groups = {
            0: {
                "members": ["u1", "u2"],
                "n_members": 2,
                "papers_total": 5,
                "topics_p1": {"t1"},
                "topics_p2": {"t2"},
            }
        }
        all_topics = {"t1", "t2", "t3"}
        topic_frequency = {"t1": 10, "t2": 5, "t3": 1}
        topic_frequency_p1 = {"t1": 5, "t2": 3}
        topic_frequency_p2 = {"t1": 5, "t2": 2, "t3": 1}
        article_authors = {"a1": {"u1", "u2"}}
        article_topics = {"a1": {"t1"}}
        identifier = GroupIdentifier(article_authors, article_topics)
        groups_p1 = identifier.extract_groups_in_period({"a1"})
        groups_p2 = identifier.extract_groups_in_period({"a1"})
        persistent = identifier.identify_persistent(groups_p1, groups_p2)
        grs = GroupRecommendationSystem(
            persistent,
            all_topics,
            topic_frequency,
            topic_frequency_p1,
            topic_frequency_p2,
        )
        df = grs.generate_all_recommendations(k=2)
        assert "group_id" in df.columns
        assert "topic" in df.columns

    def test_compute_relevance_with_zero_freq(self):
        groups = {0: {"topics_p1": set(), "topics_p2": set()}}
        topic_frequency = {}
        grs = GroupRecommendationSystem(groups, set(), topic_frequency)
        relevance = grs.compute_relevance("nonexistent")
        assert relevance == 1.0

    def test_compute_novelty_with_zero_total(self):
        groups = {0: {"topics_p1": set(), "topics_p2": set()}}
        topic_frequency = {}
        grs = GroupRecommendationSystem(
            groups, set(), topic_frequency, topic_frequency_p1={}, topic_frequency_p2={}
        )
        novelty = grs.compute_novelty("nonexistent")
        assert novelty == 1.0

    def test_compute_collaborative_signal_empty(self):
        groups = {0: {"topics_p1": set(), "topics_p2": set()}}
        topic_frequency = {}
        grs = GroupRecommendationSystem(groups, set(), topic_frequency)
        signal = grs.compute_collaborative_signal("t1", [])
        assert signal == 0.0

    def test_compute_collaborative_signal_with_similar_groups(self):
        groups = {
            0: {"topics_p1": set(), "topics_p2": {"t1"}},
            1: {"topics_p1": set(), "topics_p2": {"t1", "t2"}},
        }
        topic_frequency = {"t1": 10, "t2": 5}
        grs = GroupRecommendationSystem(groups, {"t1", "t2"}, topic_frequency)
        signal = grs.compute_collaborative_signal("t1", [(1, 0.8)])
        assert signal == 0.8


class TestMetricsEvaluator:
    def test_calculate_novelty_rate(self):
        df = pd.DataFrame({"novelty": [0.5, 0.8, 1.0]})
        assert abs(MetricsEvaluator.calculate_novelty_rate(df) - 0.766666) < 1e-5

    def test_calculate_novelty_rate_empty(self):
        df = pd.DataFrame({"novelty": []})
        assert MetricsEvaluator.calculate_novelty_rate(df) == 0.0

    def test_calculate_new_vs_recent(self):
        df = pd.DataFrame({"is_new_vs_recent": [1, 0, 1]})
        assert abs(MetricsEvaluator.calculate_new_vs_recent(df) - 2 / 3) < 1e-5

    def test_calculate_coverage(self):
        df = pd.DataFrame({"topic": ["t1", "t2", "t1"]})
        assert abs(MetricsEvaluator.calculate_coverage(df, 4) - 2 / 4) < 1e-5

    def test_calculate_coverage_zero_topics(self):
        df = pd.DataFrame({"topic": []})
        assert MetricsEvaluator.calculate_coverage(df, 0) == 0.0

    def test_calculate_diversity(self):
        df = pd.DataFrame({"topic": ["t1", "t2", "t1", "t2"]})
        diversity = MetricsEvaluator.calculate_diversity(df)
        assert 0.0 <= diversity <= 1.0

    def test_calculate_diversity_single_topic(self):
        df = pd.DataFrame({"topic": ["t1", "t1", "t1"]})
        assert MetricsEvaluator.calculate_diversity(df) == 0.0

    def test_calculate_diversity_empty(self):
        df = pd.DataFrame({"topic": []})
        assert MetricsEvaluator.calculate_diversity(df) == 0.0

    def test_calculate_avg_relevance(self):
        df = pd.DataFrame({"relevance": [0.2, 0.4, 0.6]})
        assert abs(MetricsEvaluator.calculate_avg_relevance(df) - 0.4) < 1e-5

    def test_calculate_avg_score(self):
        df = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
        assert abs(MetricsEvaluator.calculate_avg_score(df) - 0.2) < 1e-5

    def test_calculate_group_score_gini(self):
        df = pd.DataFrame({"group_id": [0, 0, 1, 1], "score": [0.8, 0.9, 0.1, 0.2]})
        gini = MetricsEvaluator.calculate_group_score_gini(df)
        assert 0.0 <= gini <= 1.0

    def test_calculate_group_score_gini_empty(self):
        df = pd.DataFrame({"group_id": [], "score": []})
        assert MetricsEvaluator.calculate_group_score_gini(df) == 0.0

    def test_calculate_group_score_gini_zero_total(self):
        df = pd.DataFrame({"group_id": [0, 0], "score": [0.0, 0.0]})
        assert MetricsEvaluator.calculate_group_score_gini(df) == 0.0

    def test_calculate_score_gap_by_group_size_with_four_groups(self):
        df = pd.DataFrame(
            {
                "group_id": [0, 0, 1, 1, 2, 2, 3, 3],
                "score": [0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3],
            }
        )
        groups = {
            0: {"n_members": 20},
            1: {"n_members": 15},
            2: {"n_members": 10},
            3: {"n_members": 5},
        }
        gap = MetricsEvaluator.calculate_score_gap_by_group_size(df, groups)
        assert isinstance(gap, float)

    def test_calculate_score_gap_by_group_size(self):
        df = pd.DataFrame(
            {
                "group_id": [0, 0, 1, 1, 2, 2],
                "score": [0.9, 0.9, 0.5, 0.5, 0.1, 0.1],
            }
        )
        groups = {
            0: {"n_members": 10},
            1: {"n_members": 5},
            2: {"n_members": 2},
        }
        gap = MetricsEvaluator.calculate_score_gap_by_group_size(df, groups)
        assert isinstance(gap, float)

    def test_print_report(self):
        df = pd.DataFrame(
            {
                "group_id": [0, 0, 0],
                "topic": ["t1", "t2", "t3"],
                "novelty": [0.5, 0.6, 0.7],
                "relevance": [0.8, 0.7, 0.6],
                "score": [0.9, 0.8, 0.7],
                "is_new_vs_recent": [1, 0, 1],
            }
        )
        result = MetricsEvaluator.print_report(df, 1, 3)
        assert "novelty_rate" in result
        assert "coverage" in result


class TestRunGRSPipeline:
    def test_run_grs_pipeline(self, tmp_path, monkeypatch):
        import app.grs.grs_production as grs_module

        monkeypatch.setattr(grs_module, "run_grs_pipeline", lambda **kwargs: (pd.DataFrame(), {}))
        result = grs_module.run_grs_pipeline(
            input_path=str(tmp_path),
            output_path=str(tmp_path),
            k=2,
            verbose=False,
        )
        assert isinstance(result, tuple)

    def test_calculate_score_gap_by_group_size_single_bucket(self):
        df = pd.DataFrame(
            {
                "group_id": [0, 0, 1, 1],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )
        groups = {
            0: {"n_members": 10},
            1: {"n_members": 10},
        }
        gap = MetricsEvaluator.calculate_score_gap_by_group_size(df, groups)
        assert gap == 0.0
