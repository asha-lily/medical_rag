"""Load and validate the ground truth data set (data/ground_truth_dataset.yaml)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH_SET_PATH = _PROJECT_ROOT / "data" / "ground_truth_dataset.yaml"

RISK_LEVELS = ("high", "medium", "low")


@dataclass
class Evidence:
    """One fact a complete answer needs, located by an exact quote."""

    source: str
    quote: str
    essential: bool = False


@dataclass
class GroundTruthQuestion:
    id: str
    question: str
    risk: str
    answerable: bool
    medicine: str | None = None
    category: str | None = None
    evidence: list[Evidence] = field(default_factory=list)
    reference_answer: str = ""
    notes: str = ""


def _parse_entry(entry: dict) -> GroundTruthQuestion:
    evidence = [Evidence(**item) for item in entry.get("evidence") or []]
    return GroundTruthQuestion(
        id=entry["id"],
        question=entry["question"],
        risk=entry["risk"],
        answerable=entry["answerable"],
        medicine=entry.get("medicine"),
        category=entry.get("category"),
        evidence=evidence,
        reference_answer=(entry.get("reference_answer") or "").strip(),
        notes=(entry.get("notes") or "").strip(),
    )


def validate(questions: list[GroundTruthQuestion]) -> list[str]:
    """Return a list of problems with the ground truth set (empty if none)."""
    problems = []
    seen_ids = set()
    for q in questions:
        if q.id in seen_ids:
            problems.append(f"{q.id}: duplicate id")
        seen_ids.add(q.id)
        if q.risk not in RISK_LEVELS:
            problems.append(
                f"{q.id}: risk must be one of {RISK_LEVELS}, got {q.risk!r}"
            )
        if q.answerable and not q.evidence:
            problems.append(f"{q.id}: answerable question has no evidence")
        if not q.answerable and q.evidence:
            problems.append(f"{q.id}: unanswerable question should have no evidence")
    return problems


def is_placeholder(question: GroundTruthQuestion) -> bool:
    """True if any evidence quote is still a TODO placeholder."""
    return any(ev.quote.strip().upper().startswith("TODO") for ev in question.evidence)


def load_ground_truth_set(
    path: Path = DEFAULT_GROUND_TRUTH_SET_PATH,
) -> list[GroundTruthQuestion]:
    """Load the ground truth set, skipping unfinished (TODO) entries.

    Raises ValueError if the set has structural problems.
    """
    with open(path) as f:
        entries = yaml.safe_load(f) or []
    questions = [_parse_entry(entry) for entry in entries]
    problems = validate(questions)
    if problems:
        raise ValueError("Ground truth set has problems:\n  " + "\n  ".join(problems))

    placeholders = [q.id for q in questions if is_placeholder(q)]
    if placeholders:
        log.warning("Skipping unfinished entries: %s", ", ".join(placeholders))
    return [q for q in questions if not is_placeholder(q)]
