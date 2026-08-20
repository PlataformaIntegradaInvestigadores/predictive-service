from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import settings
from app.grs.grs_production import (
    DataPreprocessor,
    DataValidator,
    GroupIdentifier,
    GroupRecommendationSystem,
    MetricsEvaluator,
)

logger = logging.getLogger(__name__)


class RecommendationService:
    """Adaptador entre FastAPI y el sistema GRS real."""

    def __init__(self, data_path: str | None = None) -> None:
        configured_path = data_path or os.getenv("CENTINELA_DATA_PATH")

        if not configured_path:
            raise RuntimeError("No se configuró CENTINELA_DATA_PATH.")

        self.data_path = Path(configured_path).resolve()

        if not self.data_path.is_dir():
            raise FileNotFoundError(f"La carpeta de datos no existe: {self.data_path}")

        self._load_data()
        self._initialize_grs()

    def _load_data(self) -> None:
        articles_path = self.data_path / "articulos_ecuador_CLEAN.csv"
        author_article_path = self.data_path / "autor_articulo_CLEAN.csv"
        topic_article_path = self.data_path / "topic_article_CLEAN.csv"

        required_files = [
            articles_path,
            author_article_path,
            topic_article_path,
        ]

        missing_files = [str(path) for path in required_files if not path.is_file()]

        if missing_files:
            raise FileNotFoundError(
                "Faltan archivos requeridos:\n" + "\n".join(missing_files)
            )

        self.articles = pd.read_csv(articles_path)
        self.author_article = pd.read_csv(author_article_path)
        self.topic_article = pd.read_csv(topic_article_path)

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
            period1_start=settings.GRS_PERIOD1_START,
            period1_end=settings.GRS_PERIOD1_END,
            period2_start=settings.GRS_PERIOD2_START,
            period2_end=settings.GRS_PERIOD2_END,
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
            raise KeyError(f"No existe el grupo con id {group_id}")

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

    def get_metrics(self, k: int = 10) -> dict[str, Any]:
        if getattr(self, "_metrics_k", None) != k:
            logger.info("Generando recomendaciones para métricas (k=%d)...", k)

            self.recommendations_df = self.grs.generate_all_recommendations(k=k)
            self._metrics_k = k

            logger.info(
                "Recomendaciones generadas para métricas: %d",
                len(self.recommendations_df),
            )

        recommendations_df = self.recommendations_df

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
