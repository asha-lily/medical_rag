"""
uv run pytest tests/test_ground_truth_data_set.py
"""

import pytest
import yaml

from rag_pipeline.ground_truth_data_set import (
    CATEGORIES,
    Evidence,
    GroundTruthQuestion,
    load_ground_truth_set,
    validate,
)

ENTRY = {
    "id": "q001",
    "question": "What are the side effects of paracetamol?",
    "answerable": True,
    "category": "side_effects",
    "evidence": [{"source": "paracetamol_PIL.pdf", "quote": "skin rashes"}],
}


def write_yaml(tmp_path, entries):
    path = tmp_path / "ground_truth.yaml"
    path.write_text(yaml.safe_dump(entries))
    return path


def question(**overrides) -> GroundTruthQuestion:
    fields = dict(
        id="q001",
        question="What are the side effects of paracetamol?",
        answerable=True,
        medicine=["Paracetamol"],
        category="side_effects",
        evidence=[Evidence("paracetamol_PIL.pdf", "skin rashes")],
    )
    return GroundTruthQuestion(**(fields | overrides))


def test_duplicate_id_is_a_problem():
    # Given
    test_input = [question(), question()]
    expected_output = ["q001: duplicate id"]

    # When
    actual_output = validate(test_input)

    # Then
    assert actual_output == expected_output


def test_valid_question_has_no_problems():
    # Given
    test_input = [question()]
    expected_output = []

    # When
    problems = validate(test_input)

    # Then
    assert problems == expected_output


def test_unknown_category_is_a_problem():
    # Given
    test_input = [question(category="dosage")]

    # When
    problems = validate(test_input)

    # Then
    assert len(problems) == 1
    assert "q001: category must be one of" in problems[0]


def test_answerable_question_without_evidence_is_a_problem():
    # Given
    test_input = [question(evidence=[])]
    expected_output = ["q001: answerable question has no evidence"]

    # When
    problems = validate(test_input)

    # Then
    assert problems == expected_output


def test_unanswerable_question_with_evidence_is_a_problem():
    # Given
    test_input = [question(answerable=False)]
    expected_output = ["q001: unanswerable question should have no evidence"]

    # When
    problems = validate(test_input)

    # Then
    assert problems == expected_output


@pytest.mark.parametrize("quote", ["", "   ", "\n"])
def test_empty_quote_is_a_problem(quote):
    # Given
    test_input = [question(evidence=[Evidence("paracetamol_PIL.pdf", quote)])]
    expected_output = ["q001: evidence from paracetamol_PIL.pdf has an empty quote"]

    # When
    problems = validate(test_input)

    # Then
    assert problems == expected_output


def test_all_problems_are_reported_together():
    # Given
    test_input = [question(category="dosage", evidence=[])]
    expected_output = [
        f"q001: category must be one of {CATEGORIES}, got 'dosage'",
        "q001: answerable question has no evidence",
    ]

    # When
    problems = validate(test_input)

    # Then
    assert set(problems) == set(expected_output)


def test_load_raises_on_invalid_set(tmp_path):
    with pytest.raises(ValueError, match="q001: duplicate id"):
        load_ground_truth_set(write_yaml(tmp_path, [ENTRY, ENTRY]))


def test_load_empty_file_gives_empty_list(tmp_path):
    # Given
    path = tmp_path / "empty.yaml"
    path.write_text("")
    expected_output = []

    # When
    actual_output = load_ground_truth_set(path)

    # Then
    assert actual_output == expected_output
