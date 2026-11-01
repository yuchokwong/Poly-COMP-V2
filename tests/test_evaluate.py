import json
from types import SimpleNamespace

from tsr.evaluate import (
    EvalQuery,
    _nonliteral_metaphors,
    average_precision,
    bootstrap_ci,
    build_eval_query,
    check_leakage,
    draft_eval_set,
    evaluate_queries,
    holm_bonferroni,
    load_queries,
    ndcg_at_k,
    paired_bootstrap_test,
    per_query_values,
    precision_at_k,
    recall_at_k,
    recall_at_r,
    reciprocal_rank,
    reviewed_eval_set,
    save_qrels,
    save_queries,
    win_tie_loss,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k(retrieved, relevant, 0) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3
    assert recall_at_k(retrieved, relevant, 1) == 1 / 3
    assert recall_at_k(retrieved, set(), 4) == 0.0


def test_average_precision():
    assert average_precision(["a", "b", "c"], {"a", "c"}) == (1.0 + 2 / 3) / 2
    assert average_precision(["b", "c", "a"], {"a"}) == 1 / 3
    assert average_precision(["b", "c"], {"a"}) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["b", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["b", "c"], {"a"}) == 0.0


def test_ndcg_at_k_binary():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 2) - (1 / 1.584962500721156)) < 1e-9
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k(["a"], set(), 1) == 0.0


def test_ndcg_at_k_graded_prefers_high_grade_first():
    grades = {"a": 2, "b": 1}
    best = ndcg_at_k(["a", "b"], {"a", "b"}, 2, grades)
    swapped = ndcg_at_k(["b", "a"], {"a", "b"}, 2, grades)
    assert abs(best - 1.0) < 1e-9
    assert 0.0 < swapped < best


def test_recall_at_r():
    assert recall_at_r(["a", "c", "b"], {"a", "c"}) == 1.0
    assert recall_at_r(["a", "b", "c"], {"a", "c"}) == 0.5
    assert recall_at_r(["a"], set()) == 0.0


def test_evaluate_queries_perfect_retriever():
    queries = [
        EvalQuery("q1", "q1", ["1"]),
        EvalQuery("q2", "q2", ["2"], category="intent"),
    ]
    gallery = {"q1": ["1", "9", "8"], "q2": ["2", "9", "8"]}

    def retriever(query, k):
        return gallery[query][:k]

    report = evaluate_queries(retriever, queries, top_k=[1, 3])
    metrics = report["metrics"]
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["mAP"] == 1.0
    assert metrics["MRR"] == 1.0
    assert metrics["misses"] == 0
    assert report["per_category"]["visual"]["mAP"] == 1.0
    assert report["per_category"]["intent"]["mAP"] == 1.0


def test_evaluate_queries_misses():
    queries = [EvalQuery("q1", "q1", ["1"])]

    def retriever(query, k):
        return ["x", "y"]

    report = evaluate_queries(retriever, queries, top_k=[1, 2])
    assert report["metrics"]["precision@1"] == 0.0
    assert report["metrics"]["mAP"] == 0.0
    assert report["metrics"]["misses"] == 1
    assert report["metrics"]["median_rank"] is None


def test_evaluate_queries_per_split_and_ndcg():
    queries = [
        EvalQuery("q1", "q1", ["1"], metadata={"split": "val"}),
        EvalQuery("q2", "q2", ["2"], metadata={"split": "train"}),
    ]
    gallery = {"q1": ["1", "9"], "q2": ["9", "2"]}

    def retriever(query, k):
        return gallery[query][:k]

    report = evaluate_queries(retriever, queries, top_k=[1, 2])
    assert report["per_split"]["val"]["mAP"] == 1.0
    assert report["per_split"]["train"]["mAP"] == 0.5
    assert report["per_split"]["val"]["num_queries"] == 1
    assert report["metrics"]["ndcg@1"] == 0.5
    assert report["metrics"]["recall@R"] == 0.5


def test_per_query_values_and_bootstrap_ci():
    queries = [EvalQuery("q1", "q1", ["1"]), EvalQuery("q2", "q2", ["2"])]
    gallery = {"q1": ["1", "x"], "q2": ["x", "2"]}

    def retriever(query, k):
        return gallery[query][:k]

    values = per_query_values(retriever, queries, top_k=[1, 2])
    assert values["precision@1"] == [1.0, 0.0]
    assert "mAP" in values and "recall@R" in values

    ci = bootstrap_ci(values, n_boot=200, seed=0)
    mean, lo, hi = ci["mAP"]
    assert abs(mean - 0.75) < 1e-9
    assert lo <= mean <= hi


def test_paired_bootstrap_test_detects_difference():
    baseline = [0.0] * 20
    better = [1.0] * 20
    delta, lo, hi, p = paired_bootstrap_test(baseline, better, n_boot=200, seed=0)
    assert delta == 1.0
    assert lo == 1.0 and hi == 1.0
    assert p <= 0.05

    same = [0.5] * 20
    assert paired_bootstrap_test(same, same, n_boot=100)[0] == 0.0
    assert paired_bootstrap_test([], [], n_boot=10)[3] == 1.0


def test_win_tie_loss_and_holm():
    assert win_tie_loss([1, 1, 0], [2, 1, 0]) == {"win": 1, "tie": 2, "loss": 0}
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04})
    assert adjusted["a"] == 0.02
    assert adjusted["b"] == 0.04


def test_query_and_qrels_roundtrip(tmp_path):
    queries = [
        EvalQuery(
            "q1",
            "a cat at dinner",
            ["memes_1"],
            category="metaphor",
            grades={"memes_1": 2, "memes_2": 1},
            metadata={"split": "test"},
        )
    ]
    path = save_queries(queries, tmp_path / "queries.json")
    loaded = load_queries(path)
    assert loaded == queries
    assert loaded[0].to_dict()["grades"] == {"memes_1": 2, "memes_2": 1}

    qrels = save_qrels(queries, tmp_path / "qrels.jsonl")
    rows = [json.loads(line) for line in qrels.read_text(encoding="utf-8").splitlines()]
    assert {(row["qid"], row["image_id"], row["grade"]) for row in rows} == {
        ("q1", "memes_1", 2),
        ("q1", "memes_2", 1),
    }


def test_check_leakage_flags_copied_caption_and_accepts_paraphrase():
    source = "a man in a suit sitting at a dinner table with a woman"
    copied = check_leakage(source, [source], n=8)
    assert copied["passed"] is False
    assert copied["max_ngram_overlap"] > 0

    paraphrase = check_leakage(
        "someone formal at a meal beside a lady",
        [source],
        all_docs=[source],
        n=8,
    )
    assert paraphrase["passed"] is True
    assert paraphrase["strict_overlap"] == 0


def _fake_record(index, *, split="test", caption="cat sits table", meaning="feeling fine today"):
    return SimpleNamespace(
        id=index,
        image_id=f"memes_{index}",
        split=split,
        img_captions=[caption],
        meme_captions=[meaning],
        metaphors=[{"metaphor": "fire", "meaning": "chaos"}],
    )


def test_draft_eval_set_is_graded_and_checks_leakage():
    records = [_fake_record(index) for index in range(6)]
    queries = draft_eval_set(records, per_category={"visual": 2, "intent": 2, "metaphor": 2})
    assert len(queries) == 6
    assert {query.category for query in queries} == {"visual", "intent", "metaphor"}
    for query in queries:
        assert query.grades
        assert max(query.grades.values()) == 2
        assert query.metadata["leakage"]["passed"] is True
        assert query.metadata["query_origin"] == "synthetic_paraphrase"


def test_draft_eval_set_draws_each_category_from_its_own_pool():
    records = [_fake_record(index, split="test") for index in range(12)]
    queries = draft_eval_set(
        records, per_category={"visual": 2, "intent": 2, "metaphor": 2}
    )
    assert len(queries) == 6
    # A category exhausting its candidates must not starve the next category.
    assert [query.category for query in queries].count("metaphor") == 2


def test_build_eval_query_rejects_verbatim_and_accepts_rewrite():
    source = "a cat sleeps on the sofa while its owner quietly watches television"
    record = _fake_record(0, caption=source)
    assert build_eval_query(record, source, "visual", [record]) is None

    rewritten = "a lazy pet dozing on furniture as someone silently views a screen"
    query = build_eval_query(record, rewritten, "visual", [record])
    assert query is not None
    assert query.grades == {"memes_0": 2}
    assert query.metadata["leakage"]["passed"] is True


def test_reviewed_eval_set_gates_leakage_and_missing_rewrites():
    from tsr.evaluate import reviewed_eval_set

    source_text = "a cat sleeps on the sofa while its owner quietly watches television"
    record = _fake_record(0, caption=source_text)
    sources = [
        {"qid": "q1", "image_id": "memes_0", "category": "visual"},
        {"qid": "q2", "image_id": "memes_0", "category": "visual"},
        {"qid": "q3", "image_id": "memes_404", "category": "visual"},
    ]
    rewrites = {
        "q1": "a lazy pet dozing on furniture as someone silently views a screen",
        "q2": source_text,
        "q3": "some other scene entirely",
    }
    queries, rejected = reviewed_eval_set(sources, rewrites, [record])
    assert [query.qid for query in queries] == ["q1"]
    assert {item["qid"]: item["reason"] for item in rejected} == {
        "q2": "leakage",
        "q3": "unknown_image",
    }


def test_scale_per_category_keeps_total_and_proportions():
    from tsr.evaluate import scale_per_category

    scaled = scale_per_category({"visual": 80, "intent": 80, "metaphor": 80}, 30)
    assert sum(scaled.values()) == 30
    assert max(scaled.values()) - min(scaled.values()) <= 1


def test_metaphor_grade_1_ignores_corpus_common_labels():
    from tsr.evaluate import build_eval_query

    records = [_fake_record(index) for index in range(6)]
    common = records[0]
    query = build_eval_query(
        common,
        "a meme about fire standing for chaos",
        "metaphor",
        records,
        metaphor_df={"fire": 500},
    )
    # "fire" is shared but appears everywhere, so nothing extra is graded 1.
    assert query is not None
    assert query.relevant == ["memes_0"]

    rare = build_eval_query(
        common,
        "a meme about fire standing for chaos",
        "metaphor",
        records,
        metaphor_df={"fire": 2},
    )
    assert rare is not None
    assert len(rare.relevant) == 6


def test_metaphor_grade_1_requires_the_same_meaning_and_ignores_literal_rows():
    from tsr.evaluate import build_eval_query

    def rec(index, meaning, label="having"):
        return SimpleNamespace(
            id=index,
            image_id=f"memes_{index}",
            split="test",
            img_captions=["a masked figure at a pizza shop"],
            meme_captions=["outdoing a pizza chain"],
            metaphors=[{"metaphor": label, "meaning": meaning}],
        )

    source = rec(0, "trying to be a more extreme version")
    literal = rec(1, "Meme poster")
    same = rec(2, "trying to be a more extreme version")
    different = rec(3, "having arguments")
    query = build_eval_query(
        source,
        "a masked figure challenging a pizza restaurant at its own game",
        "metaphor",
        [source, literal, same, different],
        metaphor_df={"having": 4},
    )
    assert query is not None
    # Same label + same meaning only; literal "Meme poster" rows never grade 1.
    assert query.grades == {"memes_0": 2, "memes_2": 1}


def test_leakage_rejects_a_short_annotation_embedded_verbatim():
    result = check_leakage(
        "find me a meme saying when the code finally compiles and then we can go home",
        ["when the code finally compiles"],
    )
    assert result["contained"] is True
    assert result["passed"] is False


def test_leakage_checks_the_whole_gallery_not_only_the_source():
    result = check_leakage(
        "find a meme about dogs flying over the city skyline tonight",
        ["a cat sitting on a mat"],
        all_docs=["dogs flying over the city skyline"],
    )
    assert result["gallery_contained"] is True
    assert result["strict_overlap"] == 0  # too short for an 8-gram
    assert result["passed"] is False


def test_retrieval_depth_is_not_capped_by_top_k():
    from tsr.evaluate import retrieval_depth

    queries = [EvalQuery("q1", "q1", [str(i) for i in range(120)])]
    assert retrieval_depth(queries, [1, 5, 10]) == 120

    depths: list[int] = []

    def retriever(query, k):
        depths.append(k)
        return [str(i) for i in range(k)]

    report = evaluate_queries(retriever, queries, top_k=[1, 10])
    assert depths == [120]
    # Every relevant image is retrievable, so recall@R must reach 1.0.
    assert report["metrics"]["recall@R"] == 1.0


def test_per_query_values_include_recall_at_k():
    queries = [EvalQuery("q1", "q1", ["1", "2"])]

    def retriever(query, k):
        return ["1", "x"]

    values = per_query_values(retriever, queries, top_k=[1, 10])
    assert values["recall@1"] == [0.5]
    assert values["recall@10"] == [0.5]


def test_drafting_never_uses_a_literal_placeholder_as_a_metaphor():
    import random
    from types import SimpleNamespace

    from tsr.evaluate import _draft_query, _eligible

    placeholder = SimpleNamespace(
        id=0,
        image_id="memes_0",
        split="test",
        img_captions=["a shirtless man pointing"],
        meme_captions=["people should hydrate"],
        metaphors=[{"metaphor": "A shirtless man", "meaning": "Meme poster"}],
    )
    assert _eligible(placeholder, "metaphor") is None

    real = SimpleNamespace(
        id=1,
        image_id="memes_1",
        split="test",
        img_captions=["a plane in the sky"],
        meme_captions=["flights are expensive"],
        metaphors=[{"metaphor": "a plane", "meaning": "the cost of travel"}],
    )
    assert _eligible(real, "metaphor") is real
    query = _draft_query(real, "metaphor", random.Random(0))
    assert "caption writer" not in query
    assert "cost" in query and "travel" in query


def test_check_leakage_rejects_a_five_token_copy_inside_a_longer_caption():
    """An 8-gram gate alone cannot see a 5-word phrase pasted from the caption."""
    source = "A boy is shocked, eyes and mouth wide open."
    result = check_leakage("a tight shot of a boy with his eyes and mouth wide open", [source])
    assert result["max_source_span"] >= 5
    assert result["passed"] is False

    # Four shared content words are normal descriptive overlap, not copying.
    ok = check_leakage("a stunned kid whose mouth hangs open", [source])
    assert ok["max_source_span"] < 5
    assert ok["passed"] is True


def test_literal_meaning_variants_are_not_metaphors():
    variants = ["Meme poster", "the meme poster", "poster", "The Poster",
                "not related to the meme context"]
    for meaning in variants:
        record = SimpleNamespace(
            metaphors=[{"metaphor": "a man", "meaning": meaning}],
            img_captions=[],
            meme_captions=[],
            title=None,
        )
        assert _nonliteral_metaphors(record) == [], meaning

    # "meme poster's boss" is an intended meaning, not the literal marker.
    record = SimpleNamespace(
        metaphors=[{"metaphor": "an angry character", "meaning": "meme poster's boss"}],
        img_captions=[],
        meme_captions=[],
        title=None,
    )
    assert _nonliteral_metaphors(record) == [
        {"metaphor": "an angry character", "meaning": "meme poster's boss"}
    ]


def test_reviewed_eval_set_rejects_non_test_and_stale_sources():
    source = "a cat sits on a table"
    record = _fake_record(1, split="trainval", caption=source, meaning="pet pride")
    reviewed = {
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
    }
    sources = [
        # Trainval sources leak the gallery's own annotations into the benchmark.
        {"qid": "q1", "image_id": "memes_1", "category": "visual"},
        # The draft was reviewed against a caption the gallery no longer has.
        {
            "qid": "q2",
            "image_id": "memes_2",
            "category": "visual",
            "img_captions": ["an older caption"],
        },
        # This one still matches (appending a caption keeps the reviewed prefix).
        {
            "qid": "q3",
            "image_id": "memes_3",
            "category": "visual",
            "img_captions": [source],
        },
    ]
    records = [
        record,
        _fake_record(2, split="test", caption="a dog sleeps", meaning="rest"),
        _fake_record(3, split="test", caption=source, meaning="pet pride"),
    ]
    queries, rejected = reviewed_eval_set(sources, reviewed, records)
    reasons = {item["qid"]: item["reason"] for item in rejected}
    assert reasons == {"q1": "not_test_split", "q2": "stale_source"}
    assert [query.qid for query in queries] == ["q3"]
