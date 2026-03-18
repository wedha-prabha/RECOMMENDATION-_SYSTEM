from src.recommender import SAMPLE_ITEMS, SAMPLE_USER, build_demo_output, cosine_similarity, rank_items


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert round(cosine_similarity((1.0, 2.0), (1.0, 2.0)), 5) == 1.0


def test_rank_items_returns_expected_top_pick() -> None:
    ranked = rank_items(SAMPLE_USER, SAMPLE_ITEMS, limit=3)
    assert ranked[0].item.title == "Trending Recommender Case Study"
    assert ranked[0].final_score >= ranked[1].final_score


def test_demo_output_contains_explanations() -> None:
    rows = build_demo_output()
    assert rows[0]["rank"] == "1"
    assert "embedding similarity" in rows[0]["why"]
