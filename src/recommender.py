from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ContentItem:
    item_id: str
    title: str
    topic: str
    difficulty: float
    freshness: float
    popularity: float
    embedding: Tuple[float, ...]


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    interests: Tuple[str, ...]
    preferred_difficulty: float
    embedding: Tuple[float, ...]


@dataclass(frozen=True)
class Recommendation:
    item: ContentItem
    cosine_similarity: float
    business_score: float
    final_score: float
    explanation: str


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_norm(vector: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product(left, right) / (left_norm * right_norm)


def topic_match_bonus(user: UserProfile, item: ContentItem) -> float:
    return 0.08 if item.topic in user.interests else 0.0


def difficulty_penalty(user: UserProfile, item: ContentItem) -> float:
    gap = abs(user.preferred_difficulty - item.difficulty)
    return min(gap * 0.06, 0.18)


def business_score(user: UserProfile, item: ContentItem) -> float:
    freshness_weight = 0.15 * item.freshness
    popularity_weight = 0.12 * item.popularity
    topic_bonus = topic_match_bonus(user, item)
    penalty = difficulty_penalty(user, item)
    return freshness_weight + popularity_weight + topic_bonus - penalty


def final_ranking_score(user: UserProfile, item: ContentItem) -> Tuple[float, float, float]:
    similarity = cosine_similarity(user.embedding, item.embedding)
    business = business_score(user, item)
    final_score = (0.75 * similarity) + (0.25 * business)
    return similarity, business, final_score


def explain_recommendation(user: UserProfile, item: ContentItem, similarity: float, business: float, final_score: float) -> str:
    reasons = [f"embedding similarity={similarity:.3f}"]
    if item.topic in user.interests:
        reasons.append(f"topic match on '{item.topic}'")
    reasons.append(f"freshness={item.freshness:.2f}")
    reasons.append(f"popularity={item.popularity:.2f}")
    reasons.append(f"difficulty gap={abs(user.preferred_difficulty - item.difficulty):.2f}")
    reasons.append(f"business score={business:.3f}")
    reasons.append(f"final score={final_score:.3f}")
    return ", ".join(reasons)


def rank_items(user: UserProfile, items: Iterable[ContentItem], limit: int = 5) -> List[Recommendation]:
    recommendations: List[Recommendation] = []
    for item in items:
        similarity, business, final_score = final_ranking_score(user, item)
        recommendations.append(
            Recommendation(
                item=item,
                cosine_similarity=similarity,
                business_score=business,
                final_score=final_score,
                explanation=explain_recommendation(user, item, similarity, business, final_score),
            )
        )

    recommendations.sort(
        key=lambda rec: (rec.final_score, rec.cosine_similarity, rec.business_score),
        reverse=True,
    )
    return recommendations[:limit]


SAMPLE_ITEMS: Tuple[ContentItem, ...] = (
    ContentItem("i1", "LLM System Design", "ml", 0.85, 0.90, 0.95, (0.92, 0.81, 0.24, 0.10)),
    ContentItem("i2", "Python for Data Interviews", "python", 0.45, 0.70, 0.88, (0.86, 0.65, 0.22, 0.20)),
    ContentItem("i3", "Neural Ranking Basics", "search", 0.72, 0.83, 0.74, (0.80, 0.76, 0.35, 0.18)),
    ContentItem("i4", "SQL Crash Course", "sql", 0.30, 0.55, 0.80, (0.48, 0.34, 0.12, 0.06)),
    ContentItem("i5", "Embedding Evaluation in Production", "ml", 0.88, 0.95, 0.69, (0.94, 0.84, 0.31, 0.14)),
    ContentItem("i6", "Trending Recommender Case Study", "recsys", 0.78, 0.98, 0.91, (0.89, 0.79, 0.28, 0.16)),
)

SAMPLE_USER = UserProfile(
    user_id="u1",
    interests=("ml", "python", "recsys"),
    preferred_difficulty=0.72,
    embedding=(0.90, 0.78, 0.26, 0.12),
)


def build_demo_output() -> List[Dict[str, str]]:
    ranked = rank_items(SAMPLE_USER, SAMPLE_ITEMS, limit=5)
    return [
        {
            "rank": str(index),
            "item_id": recommendation.item.item_id,
            "title": recommendation.item.title,
            "score": f"{recommendation.final_score:.3f}",
            "why": recommendation.explanation,
        }
        for index, recommendation in enumerate(ranked, start=1)
    ]


if __name__ == "__main__":
    for row in build_demo_output():
        print(f"#{row['rank']} {row['title']} ({row['score']}): {row['why']}")
