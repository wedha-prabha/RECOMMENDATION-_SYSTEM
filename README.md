# AI Recommendation System (Python)

A compact, interview-friendly recommendation system project in Python that demonstrates:

- **Embeddings** for semantic user-item matching.
- **Explicit ranking logic** beyond pure cosine similarity.
- A **real ML product framing** for content or course recommendation.
- A **trending interview topic**: recommendation systems + representation learning.

## Why this is a strong interview project

This project gives you a clean story to discuss:

1. **Candidate generation / retrieval** using embeddings.
2. **Ranking** using a weighted score that combines relevance and business/product features.
3. **Explainability** by breaking the score into similarity, freshness, popularity, and difficulty fit.
4. **Production thinking** around evaluation, cold start, and feature tradeoffs.

## Ranking formula

For each item, the system computes:

- `cosine_similarity(user_embedding, item_embedding)`
- `business_score = freshness + popularity + topic_bonus - difficulty_penalty`
- `final_score = 0.75 * similarity + 0.25 * business_score`

The weights are easy to explain in interviews:

- **75% semantic relevance** from embeddings.
- **25% product value** from freshness, popularity, and user-fit adjustments.

## Project structure

- `src/recommender.py` — core recommendation logic and sample data.
- `tests/test_recommender.py` — basic checks for similarity, ranking, and explanations.

## Run the demo

```bash
python3 src/recommender.py
```

## Run tests

```bash
python3 -m pytest
```

## Example discussion points for interviews

- Why embeddings are better than keyword-only matching.
- How to balance relevance with business metrics.
- How to evaluate ranking quality with CTR, watch time, conversion, or NDCG.
- How to handle cold-start users/items.
- How to extend this into a two-stage recommender: retrieval + reranking.
