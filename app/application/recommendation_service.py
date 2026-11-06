from __future__ import annotations

import logging
import threading
import time
from typing import Any, NamedTuple

import pandas as pd

from app.application.ports import ArticleRepository
from app.domain.errors import GroupNotFound
from app.domain.grs import (
    DataPreprocessor,
    DataValidator,
    GroupIdentifier,
    GroupRecommendationSystem,
    MetricsEvaluator,
)

logger = logging.getLogger(__name__)


class GrsPeriods(NamedTuple):
    """Años (inclusive) de los dos periodos con que se detectan grupos persistentes."""

    period1_start: int
    period1_end: int
    period2_start: int
    period2_end: int


class RecommendationService:
    """Casos de uso de recomendación de tópicos a grupos de investigación."""

    def __init__(self, repository: ArticleRepository, periods: GrsPeriods) -> None:
        self._periods = periods
        self._metrics_k: int | None = None
        self._metrics_df: pd.DataFrame | None = None
        self._metrics_lock = threading.Lock()

        self._load_data(repository)
        self._initialize_grs()

    def _load_data(self, repository: ArticleRepository) -> None:
        dataset = repository.load()

        self.articles = dataset.articles
        self.author_article = dataset.author_article
        self.topic_article = dataset.topic_article

        DataValidator.validate(
            self.articles,
            self.author_article,
            self.topic_article,
        )

        logger.info("Artículos cargados: %d", len(self.articles))
        logger.info("Relaciones autor-artículo: %d", len(self.author_article))
        logger.info("Relaciones tópico-artículo: %d", len(self.topic_article))

    def _initialize_grs(self) -> None:
        preprocessor = DataPreprocessor(
            self.articles,
            self.author_article,
            self.topic_article,
        )

        article_authors = preprocessor.get_article_authors_map()
        article_topics = preprocessor.get_article_topics_map()

        self.all_topics = preprocessor.get_all_topics()
        topic_frequency = preprocessor.get_topic_frequency()

        articles_p1, articles_p2 = preprocessor.get_articles_by_period(
            **self._periods._asdict()
        )

        topic_frequency_p1, topic_frequency_p2 = (
            preprocessor.get_topic_frequency_by_period(
                article_topics, articles_p1, articles_p2
            )
        )

        identifier = GroupIdentifier(article_authors, article_topics)

        groups_p1 = identifier.extract_groups_in_period(articles_p1)

        groups_p2 = identifier.extract_groups_in_period(articles_p2)

        self.groups = identifier.identify_persistent(groups_p1, groups_p2)

        self.grs = GroupRecommendationSystem(
            self.groups,
            self.all_topics,
            topic_frequency,
            topic_frequency_p1,
            topic_frequency_p2,
        )

        logger.info("Grupos persistentes: %d", len(self.groups))
        logger.info("Tópicos únicos: %d", len(self.all_topics))

    def get_group_recommendations(
        self,
        group_id: int,
        k: int = 10,
    ) -> dict[str, Any]:
        if group_id not in self.groups:
            raise GroupNotFound(f"No existe el grupo con id {group_id}")

        recommendations_df = self.grs.recommend_for_group(
            group_id=group_id,
            k=k,
        )

        recommendations = recommendations_df.reset_index(drop=True).to_dict(
            orient="records"
        )

        for rank, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            recommendation["rank"] = rank

        group = self.groups[group_id]

        return {
            "group_id": group_id,
            "members": group["members"],
            "n_members": group["n_members"],
            "papers_total": group["papers_total"],
            "recommendations": recommendations,
            "count": len(recommendations),
        }

    def find_group_by_members(self, scopus_ids: list[str]) -> int | None:
        normalized = {str(sid) for sid in scopus_ids}

        for group_id, group in self.groups.items():
            members = {str(member) for member in group["members"]}

            if normalized & members:
                return group_id

        return None

    def get_all_recommendations(
        self,
        k: int = 10,
        limit: int = 50,
    ) -> dict[str, Any]:
        groups_data = []

        for group_id in list(self.groups.keys())[:limit]:
            groups_data.append(
                self.get_group_recommendations(
                    group_id=group_id,
                    k=k,
                )
            )

        return {
            "total_groups": len(self.groups),
            "displayed_groups": len(groups_data),
            "groups": groups_data,
        }

    def warm_up(self, k: int = 10) -> None:
        """Calcula (y deja en caché) las recomendaciones de todos los grupos.

        Es lo más caro del servicio: sin esto, la primera petición a /groups o
        /metrics paga el cálculo completo."""
        started = time.monotonic()
        logger.info(
            "Precalculando recomendaciones (k=%d, %d grupos)...", k, len(self.groups)
        )
        self.get_metrics(k=k)
        logger.info(
            "Recomendaciones precalculadas en %.1f s", time.monotonic() - started
        )

    def get_metrics(self, k: int = 10) -> dict[str, Any]:
        # Un solo cálculo a la vez: dos peticiones con distinto k no deben
        # pisarse el resultado, y el DataFrame se lee dentro del lock.
        with self._metrics_lock:
            if self._metrics_k != k:
                logger.info("Generando recomendaciones para métricas (k=%d)...", k)

                self._metrics_df = self.grs.generate_all_recommendations(k=k)
                self._metrics_k = k

                logger.info(
                    "Recomendaciones generadas para métricas: %d",
                    len(self._metrics_df),
                )

            recommendations_df = self._metrics_df

        return {
            "groups_persistent": len(self.groups),
            "topics_unique": len(self.all_topics),
            "novelty_rate": float(
                MetricsEvaluator.calculate_novelty_rate(recommendations_df) * 100
            ),
            "new_vs_recent": float(
                MetricsEvaluator.calculate_new_vs_recent(recommendations_df) * 100
            ),
            "coverage": float(
                MetricsEvaluator.calculate_coverage(
                    recommendations_df,
                    len(self.all_topics),
                )
                * 100
            ),
            "diversity": float(
                MetricsEvaluator.calculate_diversity(recommendations_df) * 100
            ),
            "avg_relevance": float(
                MetricsEvaluator.calculate_avg_relevance(recommendations_df)
            ),
            "avg_score": float(
                MetricsEvaluator.calculate_avg_score(recommendations_df)
            ),
            "fairness_gini": float(
                MetricsEvaluator.calculate_group_score_gini(recommendations_df)
            ),
            "fairness_score_gap_by_group_size": float(
                MetricsEvaluator.calculate_score_gap_by_group_size(
                    recommendations_df,
                    self.groups,
                )
            ),
            "n_recommendations": len(recommendations_df),
        }
