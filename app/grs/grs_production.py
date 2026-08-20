"""
═══════════════════════════════════════════════════════════════════════════════
SISTEMA DE RECOMENDACIÓN GRUPAL - VERSIÓN PRODUCCIÓN
Centinela V2.2
Autor: Andrés Salvatore Suárez Herrera
Directora: PhD. Gabriela Suntaxi

Este módulo implementa el pipeline completo GRS:
1. Carga y validación de datos
2. Identificación de grupos persistentes
3. Cálculo de similitudes
4. Generación de recomendaciones
5. Evaluación con métricas
═══════════════════════════════════════════════════════════════════════════════
"""

import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 1: Validación de datos
# ═══════════════════════════════════════════════════════════════════════════════


class DataValidator:
    """Valida que los datos de entrada cumplen requisitos mínimos"""

    @staticmethod
    def validate(df_articles, df_author_article, df_topic_article):
        """
        Valida estructura de datos.

        Raises:
            ValueError: Si los datos no cumplen requisitos
        """
        errors = []

        # Validar articles
        required_articles = ["scopus_id", "publication_date"]
        for col in required_articles:
            if col not in df_articles.columns:
                errors.append(f" articulos_ecuador.csv: falta columna '{col}'")

        # Validar autor_articulo
        required_aa = ["scopus_id", "authid"]
        for col in required_aa:
            if col not in df_author_article.columns:
                errors.append(f" autor_articulo.csv: falta columna '{col}'")

        # Validar topic_article
        required_ta = ["article_id", "topic"]
        for col in required_ta:
            if col not in df_topic_article.columns:
                errors.append(f" topic_article.csv: falta columna '{col}'")

        # Validar tamaños
        if len(df_articles) < 100:
            errors.append(f"  Pocas artículos: {len(df_articles)} (esperado ≥100)")

        if len(df_topic_article) < 1000:
            errors.append(f"  Pocos tópicos: {len(df_topic_article)} (esperado ≥1000)")

        if errors:
            raise ValueError("\n".join(errors))

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 2: Preparación de datos
# ═══════════════════════════════════════════════════════════════════════════════


class DataPreprocessor:
    """Prepara datos para el modelo"""

    def __init__(self, df_articles, df_author_article, df_topic_article):
        """Inicializa con datos brutos"""
        self.df_articles = df_articles.copy()
        self.df_author_article = df_author_article
        self.df_topic_article = df_topic_article

        # Agregar año
        self.df_articles["year"] = pd.to_datetime(
            self.df_articles["publication_date"]
        ).dt.year

    def get_article_authors_map(self) -> dict[str, set]:
        """Mapea artículo -> conjunto de autores"""
        return (
            self.df_author_article.set_index("scopus_id")["authid"]
            .groupby(level=0)
            .apply(set)
            .to_dict()
        )

    def get_article_topics_map(self) -> dict[str, set]:
        """Mapea artículo -> conjunto de tópicos"""
        return (
            self.df_topic_article.set_index("article_id")["topic"]
            .groupby(level=0)
            .apply(set)
            .to_dict()
        )

    def get_all_topics(self) -> set[str]:
        """Retorna conjunto de todos los tópicos únicos"""
        return set(self.df_topic_article["topic"].unique())

    def get_topic_frequency(self) -> dict[str, int]:
        """Calcula frecuencia de cada tópico"""
        return self.df_topic_article["topic"].value_counts().to_dict()

    def get_topic_frequency_by_period(
        self, article_topics_map: dict[str, set], articles_p1: set, articles_p2: set
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Calcula la frecuencia de cada tópico por separado en cada período,
        para medir qué tan emergente es un tópico (novedad temporal).

        Returns:
            (topic_frequency_p1, topic_frequency_p2)
        """
        freq_p1 = defaultdict(int)
        freq_p2 = defaultdict(int)

        for aid in articles_p1:
            for topic in article_topics_map.get(aid, set()):
                freq_p1[topic] += 1

        for aid in articles_p2:
            for topic in article_topics_map.get(aid, set()):
                freq_p2[topic] += 1

        return dict(freq_p1), dict(freq_p2)

    def get_articles_by_period(
        self, period1_start=2020, period1_end=2022, period2_start=2023, period2_end=2025
    ) -> tuple[set, set]:
        """
        Divide artículos en dos períodos temporales.

        Returns:
            (articles_period1, articles_period2)
        """
        p1 = set(
            self.df_articles[
                (self.df_articles["year"] >= period1_start)
                & (self.df_articles["year"] <= period1_end)
            ]["scopus_id"]
        )

        p2 = set(
            self.df_articles[
                (self.df_articles["year"] >= period2_start)
                & (self.df_articles["year"] <= period2_end)
            ]["scopus_id"]
        )

        return p1, p2


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 3: Identificación de grupos
# ═══════════════════════════════════════════════════════════════════════════════


class GroupIdentifier:
    """Identifica grupos persistentes de coautores"""

    def __init__(self, article_authors_map: dict, article_topics_map: dict):
        """
        Inicializa con mapeos de artículos.

        Args:
            article_authors_map: Dict[article_id -> Set[author_ids]]
            article_topics_map: Dict[article_id -> Set[topics]]
        """
        self.article_authors = article_authors_map
        self.article_topics = article_topics_map

    def extract_groups_in_period(self, articles: set) -> dict:
        """
        Extrae grupos (equipos de coautores) de un período.

        Args:
            articles: Set de article_ids en el período

        Returns:
            Dict[team_tuple -> {papers, topics}]
        """
        groups = defaultdict(lambda: {"papers": 0, "topics": set()})

        for aid in articles:
            if aid in self.article_authors:
                authors = self.article_authors[aid]

                # Mínimo 2 autores para considerarlo "grupo"
                if len(authors) >= 2:
                    team_key = tuple(sorted(authors))
                    groups[team_key]["papers"] += 1

                    if aid in self.article_topics:
                        groups[team_key]["topics"].update(self.article_topics[aid])

        return groups

    def identify_persistent(
        self, groups_p1: dict, groups_p2: dict, min_papers_total: int = 2
    ) -> dict:
        """
        Identifica grupos persistentes (presentes en ambos períodos).

        Args:
            groups_p1: Grupos del período 1
            groups_p2: Grupos del período 2
            min_papers_total: Mínimo de papers para ser válido

        Returns:
            Dict[group_id -> {members, papers_p1, papers_p2, topics_p1, topics_p2}]
        """
        persistent = {}
        gid = 0

        for team_key in groups_p1.keys():
            if team_key in groups_p2:
                n_total = groups_p1[team_key]["papers"] + groups_p2[team_key]["papers"]

                if n_total >= min_papers_total:
                    persistent[gid] = {
                        "members": list(team_key),
                        "n_members": len(team_key),
                        "papers_p1": groups_p1[team_key]["papers"],
                        "papers_p2": groups_p2[team_key]["papers"],
                        "papers_total": n_total,
                        "topics_p1": groups_p1[team_key]["topics"],
                        "topics_p2": groups_p2[team_key]["topics"],
                    }
                    gid += 1

        return persistent


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 4: Cálculo de similitudes
# ═══════════════════════════════════════════════════════════════════════════════


class SimilarityCalculator:
    """Calcula similitudes entre grupos"""

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Calcula similitud Jaccard entre dos conjuntos.

        J(A,B) = |A ∩ B| / |A ∪ B|

        Returns:
            float entre 0 y 1
        """
        if len(set1) == 0 or len(set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def compute_group_similarities(groups: dict, top_k: int = 5) -> dict:
        """
        Calcula similitud Jaccard entre todos los pares de grupos.

        Args:
            groups: Dict de grupos persistentes
            top_k: Mantener solo los k grupos más similares

        Returns:
            Dict[group_id -> [(similar_group_id, similarity_score)]]
        """
        similarities = defaultdict(list)

        for g1_id in groups.keys():
            topics_g1 = groups[g1_id]["topics_p1"]

            for g2_id in groups.keys():
                if g1_id != g2_id:
                    topics_g2 = groups[g2_id]["topics_p1"]

                    jaccard = SimilarityCalculator.jaccard_similarity(
                        topics_g1, topics_g2
                    )

                    if jaccard > 0:
                        similarities[g1_id].append((g2_id, jaccard))

        # Mantener top-k
        for gid in similarities.keys():
            similarities[gid] = sorted(
                similarities[gid], key=lambda x: x[1], reverse=True
            )[:top_k]

        return similarities


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 5: Sistema GRS
# ═══════════════════════════════════════════════════════════════════════════════


class GroupRecommendationSystem:
    """Algoritmo GRS híbrido"""

    def __init__(
        self,
        groups: dict,
        all_topics: set,
        topic_frequency: dict,
        topic_frequency_p1: dict = None,
        topic_frequency_p2: dict = None,
    ):
        """
        Inicializa GRS.

        Args:
            groups: Dict de grupos persistentes
            all_topics: Set de todos los tópicos
            topic_frequency: Dict[topic -> frecuencia]
            topic_frequency_p1: Dict[topic -> frecuencia en período 1]
            topic_frequency_p2: Dict[topic -> frecuencia en período 2]
        """
        self.groups = groups
        self.all_topics = all_topics
        self.topic_frequency = topic_frequency
        self.topic_frequency_p1 = topic_frequency_p1 or {}
        self.topic_frequency_p2 = topic_frequency_p2 or {}
        self.max_freq = max(topic_frequency.values()) if topic_frequency else 1
        self.similarities = SimilarityCalculator.compute_group_similarities(groups)

    def compute_relevance(self, topic: str) -> float:
        """
        Calcula relevancia de un tópico (rareza global).

        relevance = 1 - (frecuencia_normalizada)
        → Tópicos raros son más relevantes

        Args:
            topic: Nombre del tópico

        Returns:
            float entre 0 y 1
        """
        freq = self.topic_frequency.get(topic, 0)
        return 1.0 - (freq / self.max_freq)

    def compute_novelty(self, topic: str) -> float:
        """
        Calcula novelty de un tópico como su emergencia temporal: qué
        proporción de sus apariciones en el corpus caen en el período
        reciente frente al total.

        novelty = freq_p2 / (freq_p1 + freq_p2)
        → 1.0 si el tópico es enteramente nuevo/emergente en el período 2
        → 0.0 si el tópico ya no se investiga desde el período 1
        → valores intermedios para tópicos activos en ambos períodos

        (El filtro de "ya explorado por este grupo" es un paso previo y
        aparte en recommend_for_group; esta función mide novedad del tópico
        en el corpus completo, no respecto a un grupo en particular.)

        Args:
            topic: Nombre del tópico

        Returns:
            float entre 0 y 1
        """
        freq_p1 = self.topic_frequency_p1.get(topic, 0)
        freq_p2 = self.topic_frequency_p2.get(topic, 0)
        total = freq_p1 + freq_p2

        if total == 0:
            return 1.0

        return freq_p2 / total

    def compute_collaborative_signal(
        self, topic: str, similar_groups_data: list
    ) -> float:
        """
        Calcula señal colaborativa (qué exploran grupos similares).

        Args:
            topic: Nombre del tópico
            similar_groups_data: List[(group_id, similarity_score)]

        Returns:
            float entre 0 y 1
        """
        signal = 0.0

        for similar_gid, sim_score in similar_groups_data:
            if topic in self.groups[similar_gid]["topics_p2"]:
                signal += sim_score

        return signal / len(similar_groups_data) if similar_groups_data else 0.0

    def recommend_for_group(
        self,
        group_id: int,
        k: int = 10,
        alpha: float = 0.3,
        beta: float = 0.2,
        gamma: float = 0.5,
    ) -> pd.DataFrame:
        """
        Genera recomendaciones para un grupo específico.

        Score = (α × Relevancia + β × Colaborativo + γ × Novedad) / (α + β + γ)

        Args:
            group_id: ID del grupo
            k: Número de recomendaciones
            alpha: Peso de relevancia
            beta: Peso colaborativo
            gamma: Peso de novedad

        Returns:
            DataFrame con columnas: topic, novelty, relevance,
            collaborative_signal, score
        """
        group = self.groups[group_id]
        # Incluye ambos períodos: no repetir ni lo histórico ni lo ya
        # investigado recientemente por el grupo.
        explored_topics = group["topics_p1"] | group["topics_p2"]

        # Obtener grupos similares
        similar_groups = self.similarities.get(group_id, [])

        # Scoring
        scores = []
        for topic in self.all_topics:
            # No recomendar lo ya explorado
            if topic in explored_topics:
                continue

            # Componentes
            novelty = self.compute_novelty(topic)
            relevance = self.compute_relevance(topic)
            collaborative = self.compute_collaborative_signal(topic, similar_groups)

            # Score combinado
            total_weight = alpha + beta + gamma
            score = (
                alpha * relevance + beta * collaborative + gamma * novelty
            ) / total_weight

            if score > 0:
                scores.append(
                    {
                        "topic": topic,
                        "novelty": novelty,
                        "relevance": relevance,
                        "collaborative_signal": collaborative,
                        "score": score,
                    }
                )

        # Top-k
        df = pd.DataFrame(scores)
        if len(df) == 0:
            return pd.DataFrame()

        return df.nlargest(k, "score")

    def generate_all_recommendations(
        self, k: int = 10, alpha: float = 0.3, beta: float = 0.2, gamma: float = 0.5
    ) -> pd.DataFrame:
        """
        Genera recomendaciones para TODOS los grupos.

        Returns:
            DataFrame con columnas: group_id, rank, topic, novelty, relevance,
                                   collaborative_signal, score, is_new_vs_recent
        """
        all_recs = []

        for gid in self.groups.keys():
            recs = self.recommend_for_group(
                gid, k=k, alpha=alpha, beta=beta, gamma=gamma
            )

            for rank, (_, rec) in enumerate(recs.iterrows(), 1):
                # ¿Es nuevo vs período reciente?
                is_new = 1 if rec["topic"] not in self.groups[gid]["topics_p2"] else 0

                all_recs.append(
                    {
                        "group_id": gid,
                        "n_members": self.groups[gid]["n_members"],
                        "papers_total": self.groups[gid]["papers_total"],
                        "rank": rank,
                        "topic": rec["topic"],
                        "novelty": rec["novelty"],
                        "relevance": rec["relevance"],
                        "collaborative_signal": rec["collaborative_signal"],
                        "score": rec["score"],
                        "is_new_vs_recent": is_new,
                    }
                )

        return pd.DataFrame(all_recs)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE 6: Métricas de evaluación
# ═══════════════════════════════════════════════════════════════════════════════


class MetricsEvaluator:
    """Calcula métricas de evaluación del modelo"""

    @staticmethod
    def calculate_novelty_rate(df_recs: pd.DataFrame) -> float:
        """Proporción de recomendaciones nuevas"""
        return df_recs["novelty"].mean() if len(df_recs) > 0 else 0.0

    @staticmethod
    def calculate_new_vs_recent(df_recs: pd.DataFrame) -> float:
        """Proporción no explorada en período reciente"""
        return df_recs["is_new_vs_recent"].mean() if len(df_recs) > 0 else 0.0

    @staticmethod
    def calculate_coverage(df_recs: pd.DataFrame, total_topics: int) -> float:
        """Proporción de tópicos cubiertos"""
        unique = df_recs["topic"].nunique() if len(df_recs) > 0 else 0
        return unique / total_topics if total_topics > 0 else 0.0

    @staticmethod
    def calculate_diversity(df_recs: pd.DataFrame) -> float:
        """Entropía normalizada de distribución de tópicos"""
        from scipy.stats import entropy

        if len(df_recs) == 0:
            return 0.0

        topic_dist = df_recs["topic"].value_counts(normalize=True)
        if len(topic_dist) <= 1:
            return 0.0

        ent = entropy(topic_dist)
        max_ent = np.log(len(topic_dist))

        return ent / max_ent if max_ent > 0 else 0.0

    @staticmethod
    def calculate_avg_relevance(df_recs: pd.DataFrame) -> float:
        """Relevancia promedio de recomendaciones"""
        return df_recs["relevance"].mean() if len(df_recs) > 0 else 0.0

    @staticmethod
    def calculate_avg_score(df_recs: pd.DataFrame) -> float:
        """Score promedio de recomendaciones"""
        return df_recs["score"].mean() if len(df_recs) > 0 else 0.0

    @staticmethod
    def calculate_group_score_gini(df_recs: pd.DataFrame) -> float:
        """
        Coeficiente de Gini del score promedio de recomendación entre
        grupos. 0 = equidad perfecta (todos los grupos reciben
        recomendaciones de calidad similar), 1 = inequidad máxima
        (unos pocos grupos concentran las mejores recomendaciones).
        """
        if len(df_recs) == 0:
            return 0.0

        group_scores = np.sort(df_recs.groupby("group_id")["score"].mean().to_numpy())
        n = len(group_scores)
        total = group_scores.sum()

        if n <= 1 or total == 0:
            return 0.0

        index = np.arange(1, n + 1)
        return (2 * np.sum(index * group_scores) - (n + 1) * total) / (n * total)

    @staticmethod
    def calculate_score_gap_by_group_size(df_recs: pd.DataFrame, groups: dict) -> float:
        """
        Diferencia de score promedio entre el cuartil de grupos más
        grandes (por n_members) y el cuartil de grupos más pequeños.
        Un gap alto sugiere que el modelo favorece sistemáticamente a
        grupos grandes/establecidos por sobre grupos nuevos o pequeños.
        """
        if len(df_recs) == 0:
            return 0.0

        group_avg = df_recs.groupby("group_id")["score"].mean().reset_index()
        group_avg["n_members"] = group_avg["group_id"].map(
            lambda gid: groups[gid]["n_members"]
        )

        if group_avg["n_members"].nunique() < 4:
            return 0.0

        group_avg["size_bucket"] = pd.qcut(
            group_avg["n_members"], q=4, duplicates="drop"
        )
        bucket_means = group_avg.groupby("size_bucket", observed=True)["score"].mean()

        if len(bucket_means) < 2:
            return 0.0

        return float(bucket_means.iloc[-1] - bucket_means.iloc[0])

    @staticmethod
    def print_report(df_recs: pd.DataFrame, n_groups: int, n_topics: int):
        """Imprime reporte completo de métricas"""

        novelty = MetricsEvaluator.calculate_novelty_rate(df_recs)
        new_vs_recent = MetricsEvaluator.calculate_new_vs_recent(df_recs)
        coverage = MetricsEvaluator.calculate_coverage(df_recs, n_topics)
        diversity = MetricsEvaluator.calculate_diversity(df_recs)
        avg_rel = MetricsEvaluator.calculate_avg_relevance(df_recs)
        avg_score = MetricsEvaluator.calculate_avg_score(df_recs)

        print("\n" + "=" * 80)
        print(" REPORTE DE EVALUACIÓN - SISTEMA GRS")
        print("=" * 80)

        print("\nDATOS:")
        print(f"  • Grupos persistentes: {n_groups}")
        print(f"  • Recomendaciones generadas: {len(df_recs):,}")
        print(f"  • Tópicos únicos recomendados: {df_recs['topic'].nunique():,}")

        print("\nMÉTRICAS DE NOVEDAD:")
        print(f"  • Novelty Rate: {novelty:.0%}")
        print(f"  • No explorados 2023-25: {new_vs_recent:.1%}")

        print("\nMÉTRICAS DE CALIDAD:")
        print(f"  • Relevancia promedio: {avg_rel:.3f}")
        print(f"  • Score promedio: {avg_score:.3f}")
        print(f"  • Diversity (normalizada): {diversity:.1%}")
        print(f"  • Coverage: {coverage:.2%}")

        print("\n" + "=" * 80)

        return {
            "novelty_rate": novelty,
            "new_vs_recent": new_vs_recent,
            "coverage": coverage,
            "diversity": diversity,
            "avg_relevance": avg_rel,
            "avg_score": avg_score,
            "n_groups": n_groups,
            "n_recommendations": len(df_recs),
            "n_topics_unique": df_recs["topic"].nunique(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: Pipeline completo
# ═══════════════════════════════════════════════════════════════════════════════


def run_grs_pipeline(
    input_path: str,
    output_path: str,
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.5,
    k: int = 10,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el pipeline GRS completo.

    Args:
        input_path: Ruta a CSVs de entrada
        output_path: Ruta para guardar resultados
        alpha, beta, gamma: Pesos del modelo
        k: Top-k recomendaciones
        verbose: Imprimir progreso

    Returns:
        (df_recommendations, metrics_dict)
    """

    if verbose:
        print("=" * 80)
        print(" EJECUTANDO PIPELINE GRS")
        print("=" * 80)

    # PASO 1: Cargar datos
    if verbose:
        print("\n[1/6] Cargando datos...")

    df_articles = pd.read_csv(f"{input_path}articulos_ecuador.csv")
    df_author_article = pd.read_csv(f"{input_path}autor_articulo.csv")
    df_topic_article = pd.read_csv(f"{input_path}topic_article.csv")

    if verbose:
        print(f"  ✓ {len(df_articles):,} artículos")
        print(f"  ✓ {len(df_topic_article):,} relaciones tópico-artículo")

    # PASO 2: Validar datos
    if verbose:
        print("\n[2/6] Validando datos...")

    DataValidator.validate(df_articles, df_author_article, df_topic_article)

    if verbose:
        print("   Datos válidos")

    # PASO 3: Preparar datos
    if verbose:
        print("\n[3/6] Preparando datos...")

    preprocessor = DataPreprocessor(df_articles, df_author_article, df_topic_article)
    article_authors = preprocessor.get_article_authors_map()
    article_topics = preprocessor.get_article_topics_map()
    all_topics = preprocessor.get_all_topics()
    topic_frequency = preprocessor.get_topic_frequency()
    articles_p1, articles_p2 = preprocessor.get_articles_by_period()
    topic_frequency_p1, topic_frequency_p2 = preprocessor.get_topic_frequency_by_period(
        article_topics, articles_p1, articles_p2
    )

    if verbose:
        print(f"  ✓ {len(all_topics):,} tópicos únicos")
        print(f"  ✓ Período 1: {len(articles_p1):,} artículos")
        print(f"  ✓ Período 2: {len(articles_p2):,} artículos")

    # PASO 4: Identificar grupos
    if verbose:
        print("\n[4/6] Identificando grupos persistentes...")

    identifier = GroupIdentifier(article_authors, article_topics)
    groups_p1 = identifier.extract_groups_in_period(articles_p1)
    groups_p2 = identifier.extract_groups_in_period(articles_p2)
    groups = identifier.identify_persistent(groups_p1, groups_p2)

    if verbose:
        print(f"  ✓ {len(groups)} grupos persistentes")

    # PASO 5: Generar recomendaciones
    if verbose:
        print("\n[5/6] Generando recomendaciones...")

    grs = GroupRecommendationSystem(
        groups,
        all_topics,
        topic_frequency,
        topic_frequency_p1,
        topic_frequency_p2,
    )
    df_recs = grs.generate_all_recommendations(k=k, alpha=alpha, beta=beta, gamma=gamma)

    if verbose:
        print(f"  ✓ {len(df_recs):,} recomendaciones")

    # PASO 6: Evaluar y guardar
    if verbose:
        print("\n[6/6] Evaluando resultados...")

    metrics = MetricsEvaluator.print_report(df_recs, len(groups), len(all_topics))

    # Guardar resultados
    df_recs.to_csv(f"{output_path}GRS_RECOMENDACIONES.csv", index=False)
    df_recs[df_recs["rank"] <= 5].to_csv(f"{output_path}GRS_TOP5.csv", index=False)

    if verbose:
        print(f"\n Resultados guardados en {output_path}")

    return df_recs, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sistema GRS - Producción")
    parser.add_argument("--input", type=str, default="/mnt/user-data/uploads/")
    parser.add_argument("--output", type=str, default="/mnt/user-data/outputs/")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    df_recs, metrics = run_grs_pipeline(
        input_path=args.input,
        output_path=args.output,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        k=args.k,
    )
