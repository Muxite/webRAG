"""Unit tests for scripts/generate_derivation_tasks.py -- the seeded derivation-task generator.

Synthetic corpora only (a handful of hand-built infobox-shaped documents); the real
``numeric22`` corpus is exercised once, lightly, in
``test_generate_end_to_end_on_the_real_corpus_produces_usable_tasks`` to prove the pipeline
actually runs against it, without making every other test depend on its size/content.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import generate_derivation_tasks as gen  # noqa: E402


def _doc(url, title, label_value_units):
    """Build a corpus document dict whose ``text`` is a synthetic infobox: each
    ``(label, value, unit)`` triple becomes three consecutive lines, the shape
    ``agent.app.quantity_index._scan_infobox`` reads."""
    lines = []
    for label, value, unit in label_value_units:
        lines.append(label)
        lines.append(value)
        lines.append(unit)
    return {"url": url, "title": title, "description": title, "text": "\n".join(lines)}


LAKE_A = _doc(
    "https://en.wikipedia.org/wiki/Lake_A", "Lake A",
    [("Max. depth", "1642", "m"), ("Average depth", "744", "m")],
)
LAKE_B = _doc(
    "https://en.wikipedia.org/wiki/Lake_B", "Lake B",
    [("Max. depth", "506", "m"), ("Average depth", "180", "m")],
)
TOWN_C = _doc(
    "https://en.wikipedia.org/wiki/Town_C", "Town C",
    [("Population", "48213", "people"), ("Established", "1943", "")],
)
TOWN_D = _doc(
    "https://en.wikipedia.org/wiki/Town_D", "Town D",
    [("Population", "91007", "people")],
)


def _write_corpus(tmp_path, docs):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    with (corpus_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc) + "\n")
    return str(corpus_dir)


# --------------------------------------------------------------------------------------------
# load_operands / dimension grouping
# --------------------------------------------------------------------------------------------
def test_load_operands_keeps_only_labelled_infobox_quantities():
    documents = [LAKE_A, TOWN_C]
    operands = gen.load_operands(documents)
    labels = {op.label for op in operands}
    assert "Max. depth" in labels and "Population" in labels
    # "Established 1943" has no unit token at all -> dropped (no canonical unit to pair on)
    assert "Established" not in labels


def test_group_by_dimension_buckets_by_exact_unit_never_across_units():
    documents = [LAKE_A, LAKE_B, TOWN_C]
    operands = gen.load_operands(documents)
    groups = gen.group_by_dimension(operands)
    assert ("", "m") in groups
    assert ("", "people") in groups
    m_labels_docs = {(operands[i].doc_index) for i in groups[("", "m")]}
    assert m_labels_docs == {documents.index(LAKE_A), documents.index(LAKE_B)}


# --------------------------------------------------------------------------------------------
# candidate_pairs: different documents only, no cross-unit pairs possible by construction
# --------------------------------------------------------------------------------------------
def test_candidate_pairs_never_pairs_two_operands_from_the_same_document():
    documents = [LAKE_A, LAKE_B]
    operands = gen.load_operands(documents)
    pairs = gen.candidate_pairs(operands, seed=1)
    assert pairs, "expected at least one cross-document pair in the 'm' bucket"
    for i, j in pairs:
        assert operands[i].doc_index != operands[j].doc_index


def test_candidate_pairs_only_pairs_within_one_dimension_bucket():
    documents = [LAKE_A, LAKE_B, TOWN_C, TOWN_D]
    operands = gen.load_operands(documents)
    pairs = gen.candidate_pairs(operands, seed=1)
    for i, j in pairs:
        assert operands[i].dimension == operands[j].dimension


# --------------------------------------------------------------------------------------------
# compute_derived / degeneracy
# --------------------------------------------------------------------------------------------
def test_compute_derived_difference_and_sum():
    assert gen.compute_derived(10.0, 3.0, "difference") == 7.0
    assert gen.compute_derived(10.0, 3.0, "sum") == 13.0


def test_compute_derived_ratio_is_always_larger_over_smaller():
    assert gen.compute_derived(3.0, 10.0, "ratio") == gen.compute_derived(10.0, 3.0, "ratio")


def test_compute_derived_quotient_respects_operand_order():
    assert gen.compute_derived(10.0, 4.0, "quotient") == 2.5
    assert gen.compute_derived(4.0, 10.0, "quotient") == 0.4


def test_equal_values_are_degenerate_regardless_of_operation():
    a = gen.Operand(0, "u1", "T1", "L", "5", "m", 5.0, ("", "m"), "infobox")
    b = gen.Operand(1, "u2", "T2", "L", "5", "m", 5.0, ("", "m"), "infobox")
    for op in gen.OPERATIONS:
        derived = gen.compute_derived(a.magnitude, b.magnitude, op)
        assert gen._is_degenerate(a, b, op, derived) == "equal_values"


def test_ratio_near_unity_is_degenerate():
    a = gen.Operand(0, "u1", "T1", "L", "100", "m", 100.0, ("", "m"), "infobox")
    b = gen.Operand(1, "u2", "T2", "L", "102", "m", 102.0, ("", "m"), "infobox")
    derived = gen.compute_derived(a.magnitude, b.magnitude, "ratio")
    assert gen._is_degenerate(a, b, "ratio", derived) == "ratio_near_unity"


def test_a_meaningfully_different_ratio_is_not_degenerate():
    a = gen.Operand(0, "u1", "T1", "L", "100", "m", 100.0, ("", "m"), "infobox")
    b = gen.Operand(1, "u2", "T2", "L", "250", "m", 250.0, ("", "m"), "infobox")
    derived = gen.compute_derived(a.magnitude, b.magnitude, "ratio")
    assert gen._is_degenerate(a, b, "ratio", derived) is None


# --------------------------------------------------------------------------------------------
# plausibility (exact-same-fact-type gate)
# --------------------------------------------------------------------------------------------
def test_identical_normalized_labels_are_plausible():
    a = gen.Operand(0, "u1", "T1", "Max. depth", "1", "m", 1.0, ("", "m"), "infobox")
    b = gen.Operand(1, "u2", "T2", "max.  depth", "1", "m", 2.0, ("", "m"), "infobox")
    assert gen._plausible_pair(a, b) is True


def test_different_labels_are_not_plausible_even_in_the_same_unit_and_rough_topic():
    """Same unit, both loosely 'size'-shaped, but NOT the same fact -- must be rejected. This is
    exactly the case an earlier coarse family heuristic wrongly accepted (a bridge's Height
    paired with an unrelated mountain range's Elevation); see _plausible_pair's docstring."""
    a = gen.Operand(0, "u1", "T1", "Height", "1", "ft", 1.0, ("", "ft"), "infobox")
    b = gen.Operand(1, "u2", "T2", "Elevation", "1", "ft", 2.0, ("", "ft"), "infobox")
    assert gen._plausible_pair(a, b) is False


def test_max_depth_does_not_pair_with_average_depth():
    """Same entity-kind and same rough topic, but a DIFFERENT fact (max vs average) -- rejected,
    matching the authored-suite convention of comparing the identical field on two entities."""
    a = gen.Operand(0, "u1", "T1", "Max. depth", "1", "m", 1.0, ("", "m"), "infobox")
    b = gen.Operand(1, "u2", "T2", "Average depth", "1", "m", 2.0, ("", "m"), "infobox")
    assert gen._plausible_pair(a, b) is False


# --------------------------------------------------------------------------------------------
# leak check
# --------------------------------------------------------------------------------------------
def test_leaked_derived_value_is_discarded():
    leak_doc = _doc("https://en.wikipedia.org/wiki/Leak", "Leak Page",
                     [("Gap", "898", "m")])   # 1642 - 744 = 898, printed verbatim here
    documents = [LAKE_A, LAKE_B, leak_doc]
    assert gen._is_leaked(898.0, documents) is True
    assert gen._is_leaked(12345.0, documents) is False


def test_generate_end_to_end_discards_a_leaked_pair(tmp_path, monkeypatch):
    """Pin both the operation and the candidate-pair selection (bypassing the dimension-bucket
    shuffle, which is orthogonal to what this test is pinning) so the pair under test -- Lake A
    max. depth (1642 m) vs Lake B max. depth (506 m), difference 1136 m -- is exactly the one
    ``generate`` evaluates, and a leak page prints that 1136 verbatim."""
    monkeypatch.setattr(gen, "OPERATIONS", ("difference",))
    leak_doc = _doc("https://en.wikipedia.org/wiki/Leak", "Leak Page",
                     [("Gap", "1136", "m")])   # Lake A max (1642) - Lake B max (506) = 1136
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B, leak_doc])
    documents = gen.load_corpus_documents(corpus_dir)
    operands = gen.load_operands(documents)
    lake_a_max = next(i for i, o in enumerate(operands)
                       if o.url == LAKE_A["url"] and o.label == "Max. depth")
    lake_b_max = next(i for i, o in enumerate(operands)
                       if o.url == LAKE_B["url"] and o.label == "Max. depth")
    monkeypatch.setattr(gen, "candidate_pairs", lambda *a, **k: [(lake_a_max, lake_b_max)])
    result = gen.generate(seed=7, corpus_dir=corpus_dir, n_target=10)
    assert result["tasks"] == []
    assert result["stats"]["discard_leaked"] == 1


# --------------------------------------------------------------------------------------------
# retrievability
# --------------------------------------------------------------------------------------------
def test_is_retrievable_true_when_operand_query_hits_its_own_document():
    documents = [LAKE_A, LAKE_B]
    index = gen.BM25Index([gen.CorpusDocument(url=d["url"], title=d["title"],
                                               description=d["description"], text=d["text"])
                            for d in documents])
    operand = gen.Operand(0, LAKE_A["url"], LAKE_A["title"], "Max. depth", "1642", "m",
                           1642.0, ("", "m"), "infobox")
    assert gen._is_retrievable(operand, index, k=5) is True


def test_is_retrievable_false_when_the_query_never_scores_the_operands_document():
    """A query built from the operand's own title/label must share at least one token with the
    indexed text (title + description + body) to score above zero under BM25; when the operand's
    document carries none of those query tokens anywhere, and OTHER documents outrank it for
    whatever overlap remains, it falls out of the top-k."""
    target = {"url": "https://en.wikipedia.org/wiki/Target", "title": "Target Peak",
              "description": "", "text": "Some unrelated body text about geology and erosion."}
    decoy_1 = {"url": "https://en.wikipedia.org/wiki/Decoy1", "title": "Decoy One",
               "description": "", "text": "Target Peak elevation summit ridge Target Peak"}
    decoy_2 = {"url": "https://en.wikipedia.org/wiki/Decoy2", "title": "Decoy Two",
               "description": "", "text": "Target Peak elevation summit ridge Target Peak"}
    index = gen.BM25Index([gen.CorpusDocument(url=d["url"], title=d["title"],
                                               description=d["description"], text=d["text"])
                            for d in (target, decoy_1, decoy_2)])
    operand = gen.Operand(0, target["url"], target["title"], "Elevation", "1", "m",
                           1.0, ("", "m"), "infobox")
    assert gen._is_retrievable(operand, index, k=1) is False


def test_generate_drops_a_pair_whose_page_is_not_retrievable(tmp_path, monkeypatch):
    """Wire :func:`generate` to a BM25 index built the same way, but make ``_is_retrievable``
    reject one specific url so the pipeline-level dropping behaviour (not BM25 dynamics, covered
    above) is what's under test."""
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B])
    blocked_url = LAKE_B["url"]
    real_is_retrievable = gen._is_retrievable

    def _patched(operand, index, k):
        if operand.url == blocked_url:
            return False
        return real_is_retrievable(operand, index, k)

    monkeypatch.setattr(gen, "_is_retrievable", _patched)
    result = gen.generate(seed=3, corpus_dir=corpus_dir, n_target=10, k=5)
    for task in result["tasks"]:
        assert task["operand_a"]["url"] != blocked_url
        assert task["operand_b"]["url"] != blocked_url
    assert result["stats"]["discard_not_retrievable"] >= 1


# --------------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------------
def test_generate_is_deterministic_for_a_fixed_seed(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B, TOWN_C, TOWN_D])
    r1 = gen.generate(seed=42, corpus_dir=corpus_dir, n_target=10)
    r2 = gen.generate(seed=42, corpus_dir=corpus_dir, n_target=10)
    assert r1 == r2


def test_write_outputs_is_byte_identical_across_runs(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B, TOWN_C, TOWN_D])
    result = gen.generate(seed=99, corpus_dir=corpus_dir, n_target=10)
    p1, r1 = gen.write_outputs(result, str(tmp_path / "out1"), seed=99)
    p2, r2 = gen.write_outputs(result, str(tmp_path / "out2"), seed=99)
    assert p1.read_bytes() == p2.read_bytes()
    assert r1.read_bytes() == r2.read_bytes()


def test_different_seeds_can_change_the_output():
    corpus_dir_docs = [LAKE_A, LAKE_B, TOWN_C, TOWN_D]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = _write_corpus(Path(tmp), corpus_dir_docs)
        r1 = gen.generate(seed=1, corpus_dir=corpus_dir, n_target=10)
        r2 = gen.generate(seed=2, corpus_dir=corpus_dir, n_target=10)
        # Not asserting inequality (a tiny corpus can coincidentally agree); just that both run
        # cleanly and stay internally deterministic against themselves.
        assert r1 == gen.generate(seed=1, corpus_dir=corpus_dir, n_target=10)
        assert r2 == gen.generate(seed=2, corpus_dir=corpus_dir, n_target=10)


# --------------------------------------------------------------------------------------------
# round-trip
# --------------------------------------------------------------------------------------------
def test_emitted_task_round_trips_through_jsonl(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B, TOWN_C, TOWN_D])
    result = gen.generate(seed=5, corpus_dir=corpus_dir, n_target=10)
    assert result["tasks"], "expected at least one accepted task from this fixture"
    tasks_path, _ = gen.write_outputs(result, str(tmp_path / "out"), seed=5)
    loaded = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()]
    assert loaded == result["tasks"]
    for task in loaded:
        recomputed = gen.compute_derived(
            task["operand_a"]["magnitude"], task["operand_b"]["magnitude"], task["operation"])
        assert abs(recomputed - task["expected_value"]) < 1e-6


# --------------------------------------------------------------------------------------------
# cross-unit / same-document guarantees on real generated output
# --------------------------------------------------------------------------------------------
def test_generate_never_emits_a_cross_unit_pair(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B, TOWN_C, TOWN_D])
    result = gen.generate(seed=11, corpus_dir=corpus_dir, n_target=10)
    for task in result["tasks"]:
        assert task["operand_a"]["unit_text"] == task["operand_b"]["unit_text"]


def test_generate_never_emits_a_same_document_pair(tmp_path):
    corpus_dir = _write_corpus(tmp_path, [LAKE_A, LAKE_B, TOWN_C, TOWN_D])
    result = gen.generate(seed=11, corpus_dir=corpus_dir, n_target=10)
    for task in result["tasks"]:
        assert task["operand_a"]["url"] != task["operand_b"]["url"]


# --------------------------------------------------------------------------------------------
# end-to-end on the real frozen corpus (light touch: small n, just prove it runs and passes
# every gate it claims to)
# --------------------------------------------------------------------------------------------
def test_generate_end_to_end_on_the_real_corpus_produces_usable_tasks():
    result = gen.generate(seed=gen.DEFAULT_SEED, corpus_dir=gen.DEFAULT_CORPUS_DIR, n_target=5)
    assert len(result["tasks"]) >= 1
    documents = gen.load_corpus_documents(gen.DEFAULT_CORPUS_DIR)
    for task in result["tasks"]:
        assert task["operand_a"]["unit_text"] == task["operand_b"]["unit_text"]
        assert task["operand_a"]["url"] != task["operand_b"]["url"]
        assert gen._is_leaked(task["expected_value"], documents) is False
