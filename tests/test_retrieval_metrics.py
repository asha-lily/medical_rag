"""
uv run --with pytest pytest tests/test_retrieval_metrics.py -v
"""

import logging

import pytest
from langchain_core.documents import Document

from rag_pipeline.ground_truth_data_set import Evidence, GroundTruthQuestion
from rag_pipeline.retrieval_metrics import (
    check_evidence_findable,
    contains_evidence,
    evaluate_question,
    evidence_ranks,
    normalise,
    recall_at_k,
    reciprocal_rank,
)

STOP = Evidence("paracetamol_PIL.pdf", "Stop taking the tablets")
RASH = Evidence("paracetamol_PIL.pdf", "skin rashes")
BRUISE = Evidence("paracetamol_PIL.pdf", "unexpected bruising")

DOCTOR = Evidence("paracetamol_PIL.pdf", "contact your doctor straight away")
LIVER = Evidence("paracetamol_PIL.pdf", "can cause serious liver damage")
MISSING = Evidence("paracetamol_PIL.pdf", "this sentence is not in the leaflet")


def chunk(text: str, source: str = "paracetamol_PIL.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source})


INDEX = [
    chunk("If you get a rash, contact your doctor straight away."),
    chunk("Taking too much paracetamol can cause serious liver damage."),
]


def question(id: str, *evidence: Evidence, category: str = "side_effects"):
    return GroundTruthQuestion(
        id=id,
        question="placeholder question",
        answerable=True,
        category=category,
        evidence=list(evidence),
    )


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("“Take with food”", '"takewithfood"'),
        ("‘Keep out of reach’", "'keepoutofreach'"),
    ],
    ids=["curly_double", "curly_single"],
)
def test_normalise_straightens_curly_quotes(input, expected_output):
    assert normalise(input) == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("breast\u2010feeding", "breast-feeding"),  # hyphen
        ("breast\u2011feeding", "breast-feeding"),  # non-breaking hyphen
        ("breast\u2012feeding", "breast-feeding"),  # figure dash
        ("5\u201310", "5-10"),  # en dash
        ("5\u201410", "5-10"),  # em dash
        ("\u22125", "-5"),  # minus sign
    ],
    ids=["hyphen", "non_breaking_hyphen", "figure_dash", "en_dash", "em_dash", "minus"],
)
def test_normalise_converts_dashes_to_hyphen(input, expected_output):
    assert normalise(input) == expected_output


def test_normalise_removes_soft_hyphens():
    assert normalise("pain\u00adkillers") == "painkillers"


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("\ufb01lm-coated", "film-coated"),  # "ﬁ" ligature
        ("\uff21\uff22\uff23", "abc"),  # fullwidth "ＡＢＣ"
    ],
    ids=["ligature", "fullwidth"],
)
def test_normalise_applies_nfkc(input, expected_output):
    assert normalise(input) == expected_output


@pytest.mark.parametrize(
    "input,expected_output",
    [
        ("take\tone\ttablet", "takeonetablet"),
        ("  Take\r\n ONE  ", "takeone"),
        ("500\u00a0mg", "500mg"),  # non-breaking space
    ],
    ids=["tabs", "crlf_and_padding", "nbsp"],
)
def test_normalise_removes_all_whitespace(input, expected_output):
    assert normalise(input) == expected_output


@pytest.mark.parametrize(
    "quote,chunk_text",
    [
        ("if you are breast-feeding", "if you arebreast-feeding"),
        (
            "planning to become pregnant or breast feeding",
            "If you pregnant, planning to become pregnant or breastfeeding",
        ),
    ],
    ids=["are_breast", "breast_feeding"],
)
def test_contains_evidence_ignores_spaces_dropped_by_parser(quote, chunk_text):
    # Real examples from the ibuprofen leaflet: the parser drops spaces.
    ev = Evidence("ibuprofen_PIL.pdf", quote)
    assert contains_evidence(chunk(chunk_text, "ibuprofen_PIL.pdf"), ev)


def test_contains_evidence_ignores_spaces_added_by_parser():
    ev = Evidence("ibuprofen_PIL.pdf", "combinations of different painkillers, you")
    parsed = "especially combinations of different painkillers , you may"
    assert contains_evidence(chunk(parsed, "ibuprofen_PIL.pdf"), ev)


@pytest.mark.parametrize(
    "chunk_text",
    [
        "Stop taking the tablets now",
        "Do not take more than 8 tablets",
    ],
    ids=["unrelated_text", "near_miss"],
)
def test_contains_evidence_false_when_quote_absent(chunk_text):
    ev = Evidence("paracetamol_PIL.pdf", "do not take more than 4 tablets")
    assert not contains_evidence(chunk(chunk_text), ev)


def test_contains_evidence_ignores_case():
    assert contains_evidence(chunk("STOP TAKING THE TABLETS"), STOP)


def test_contains_evidence_false_when_chunk_has_no_source():
    no_source = Document(page_content="Stop taking the tablets")
    assert not contains_evidence(no_source, STOP)


def test_contains_evidence_requires_same_source():
    text = "Stop taking the tablets now"
    assert contains_evidence(chunk(text), STOP)
    assert not contains_evidence(chunk(text, "ibuprofen_PIL.pdf"), STOP)


def test_evidence_ranks_uses_first_matching_chunk():
    # With chunk overlap, the same quote can appear in more than one chunk.
    retrieved = [
        chunk("Unrelated text"),
        chunk("skin rashes may occur"),
        chunk("skin rashes again (overlapping chunk)"),
    ]
    assert evidence_ranks(retrieved, [RASH]) == [2]


def test_evidence_ranks_none_when_evidence_not_retrieved():
    retrieved = [chunk("Unrelated text"), chunk("Stop taking the tablets")]
    assert evidence_ranks(retrieved, [STOP, BRUISE]) == [2, None]


def test_evidence_ranks_skips_chunks_from_other_sources():
    other_source = chunk("Stop taking the tablets", "ibuprofen_PIL.pdf")
    same_source = chunk("Stop taking the tablets")
    assert evidence_ranks([other_source, same_source], [STOP]) == [2]
    assert evidence_ranks([other_source], [STOP]) == [None]


def test_evidence_ranks_all_none_when_nothing_retrieved():
    assert evidence_ranks([], [STOP, RASH]) == [None, None]


def test_evidence_ranks_empty_when_no_evidence():
    assert evidence_ranks([chunk("Stop taking the tablets")], []) == []


def test_recall_counts_evidence_items_within_k():
    ranks = [3, 2, None]
    assert recall_at_k(ranks, 1) == 0
    assert recall_at_k(ranks, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranks, 5) == pytest.approx(2 / 3)


def test_recall_undefined_without_evidence():
    with pytest.raises(ValueError):
        recall_at_k([], 5)


def test_reciprocal_rank_uses_best_ranked_evidence():
    assert reciprocal_rank([3, 2, None]) == pytest.approx(1 / 2)
    assert reciprocal_rank([None, None]) == 0.0


def test_check_evidence_findable_empty_when_all_found():
    assert check_evidence_findable([question("q1", DOCTOR, LIVER)], INDEX) == set()


def test_check_evidence_findable_returns_only_failing_ids():
    questions = [
        question("q1", DOCTOR),
        question("q2", MISSING),
        question("q3", DOCTOR, MISSING),
        question("q4", LIVER),
    ]
    assert check_evidence_findable(questions, INDEX) == {"q2", "q3"}


def test_check_evidence_findable_logs_missing_quote(caplog):
    with caplog.at_level(logging.WARNING):
        check_evidence_findable([question("q1", MISSING)], INDEX)
    assert "q1: quote not found" in caplog.text


@pytest.mark.parametrize(
    "quote,should_warn",
    [
        ("Stop taking the tablet", True),  # 19 non-whitespace chars
        ("Stop taking the tablets", False),  # exactly 20
    ],
    ids=["19_chars", "20_chars"],
)
def test_check_evidence_findable_warns_on_short_quote(caplog, quote, should_warn):
    ev = Evidence("paracetamol_PIL.pdf", quote)
    index = [chunk("Stop taking the tablets now")]
    with caplog.at_level(logging.WARNING):
        unfindable = check_evidence_findable([question("q1", ev)], index)
    assert ("shorter than" in caplog.text) == should_warn
    # A warning only: the question is still evaluated.
    assert unfindable == set()


@pytest.mark.parametrize(
    "n_chunks,should_warn",
    [(2, False), (3, True)],
    ids=["2_matches", "3_matches"],
)
def test_check_evidence_findable_warns_on_many_matches(caplog, n_chunks, should_warn):
    index = [chunk("contact your doctor straight away")] * n_chunks
    with caplog.at_level(logging.WARNING):
        unfindable = check_evidence_findable([question("q1", DOCTOR)], index)
    assert ("found in 3 chunks" in caplog.text) == should_warn
    assert unfindable == set()


RETRIEVED = [
    chunk("Unrelated text"),
    chunk("More unrelated text"),
    chunk("Stop taking the tablets"),
    chunk("skin rashes may occur"),
]


def test_evaluate_question_row():
    # Inputs chosen so every metric value is different, so values written
    # under the wrong key, or computed with the wrong k, fail the test.
    row = evaluate_question(question("q1", STOP, RASH), RETRIEVED, (1, 3, 4))
    assert row == {
        "id": "q1",
        "category": "side_effects",
        "n_evidence": 2,
        "evidence_ranks": [3, 4],
        "reciprocal_rank": pytest.approx(1 / 3),
        "recall@1": 0.0,
        "recall@3": 0.5,
        "recall@4": 1.0,
    }
