"""Tests for dataset loading, validation and lookup.

Invalid-dataset cases are written to temporary files so the real
data/prompts.json is never corrupted by a test run.
"""

import json
import re

import pytest

from app.dataset import (
    VALID_CATEGORIES,
    DatasetError,
    PromptDataset,
    init_dataset,
    get_prompt,
    load_dataset,
)
from app.dataset import DEFAULT_DATASET_PATH
from scripts.generate_dataset import DOC_QUESTION_SPECS, generate_records

EXPECTED_RECORD_COUNT = 750

# Every question, mapped to the clause substrings required to answer it.
QUESTION_REQUIREMENTS = {
    question: keys
    for specs in DOC_QUESTION_SPECS.values()
    for question, keys in specs
}

QUESTION_LINE = re.compile(r"^Question \d+: (.+)$", re.MULTILINE)


@pytest.fixture(scope="module")
def dataset() -> PromptDataset:
    """The real dataset, loaded from data/prompts.json."""
    return load_dataset()


def write_dataset(tmp_path, records) -> str:
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


# --- Loading the real dataset ---------------------------------------------


def test_dataset_loads(dataset: PromptDataset):
    assert len(dataset) == EXPECTED_RECORD_COUNT


def test_ids_are_unique(dataset: PromptDataset):
    ids = dataset.ids
    assert len(set(ids)) == len(ids)


def test_categories_are_valid(dataset: PromptDataset):
    categories = {record.category for record in dataset.records}
    assert categories <= VALID_CATEGORIES


def test_all_categories_present(dataset: PromptDataset):
    assert set(dataset.category_counts()) == VALID_CATEGORIES


def test_prompts_are_non_empty(dataset: PromptDataset):
    assert all(record.prompt.strip() for record in dataset.records)


def test_prompts_are_unique(dataset: PromptDataset):
    """A workload of repeated identical prompts would not exercise much."""
    prompts = [record.prompt for record in dataset.records]
    assert len(set(prompts)) == len(prompts)


def test_categories_are_balanced(dataset: PromptDataset):
    counts = dataset.category_counts()
    assert min(counts.values()) >= 200
    assert max(counts.values()) <= 300


def test_dataset_has_length_variation(dataset: PromptDataset):
    """Prompt size drives simulated prefill cost, so the spread must be real."""
    lengths = [len(record.prompt) for record in dataset.records]
    assert min(lengths) < 300
    assert max(lengths) > 1500


# --- Document-understanding answerability ----------------------------------
#
# Clauses used to be sampled independently of the questions, which produced
# prompts asking about sections the document did not contain. These tests pin
# that down: every question a prompt asks must have its supporting clause
# present in that same prompt.


def test_question_texts_are_unique_across_doc_types():
    """The question -> required clause map assumes question text is a key."""
    all_questions = [q for specs in DOC_QUESTION_SPECS.values() for q, _ in specs]
    assert len(set(all_questions)) == len(all_questions)


def test_every_document_question_is_answerable(dataset: PromptDataset):
    failures = []
    for record in dataset.records:
        if record.category != "document_understanding":
            continue
        for question in QUESTION_LINE.findall(record.prompt):
            assert question in QUESTION_REQUIREMENTS, (
                f"Prompt {record.id} asks an unmapped question: {question!r}"
            )
            for key in QUESTION_REQUIREMENTS[question]:
                if key not in record.prompt:
                    failures.append((record.id, question, key))
    assert not failures, f"Questions without supporting clauses: {failures[:5]}"


def test_known_regression_cases_are_covered(dataset: PromptDataset):
    """The two concrete failures found in the previous dataset.

    A spec prompt asking for submission size and retry count must contain
    FR-12 and FR-18; an SLA prompt asking about data after termination must
    contain Clause 5.1.
    """
    checked = {"spec": 0, "sla": 0}
    for record in dataset.records:
        if record.category != "document_understanding":
            continue
        questions = QUESTION_LINE.findall(record.prompt)
        for question in questions:
            if question.startswith("What is the maximum accepted submission size"):
                assert "Requirement FR-12" in record.prompt
                assert "Requirement FR-18" in record.prompt
                checked["spec"] += 1
            if question.startswith("What happens to customer data 120 days"):
                assert "Clause 5.1" in record.prompt
                checked["sla"] += 1
    # Both cases must actually occur in the dataset, or the test proves nothing.
    assert checked["spec"] > 0 and checked["sla"] > 0, checked


# --- Workload shape --------------------------------------------------------


def test_long_context_tail_exists(dataset: PromptDataset):
    """A slice of genuinely long prompts, for prefill/KV-cache pressure."""
    long_context = [r for r in dataset.records if len(r.prompt) > 4000]
    share = len(long_context) / len(dataset)
    assert 0.04 <= share <= 0.12, f"long-context share was {share:.1%}"
    assert max(len(r.prompt) for r in long_context) <= 12000


def test_dataset_file_uses_lf_line_endings():
    """Keeps the committed file byte-identical across operating systems."""
    raw = DEFAULT_DATASET_PATH.read_bytes()
    assert b"\r" not in raw


def test_generator_reproduces_checked_in_dataset_byte_for_byte():
    """The fixed seed and serializer settings define the committed artifact."""
    expected = (
        json.dumps(generate_records(), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    assert DEFAULT_DATASET_PATH.read_bytes() == expected


# --- Lookup ----------------------------------------------------------------


def test_lookup_valid_id(dataset: PromptDataset):
    first_id = dataset.ids[0]
    record = dataset.get(first_id)
    assert record is not None
    assert record.id == first_id
    assert record.prompt


def test_lookup_unknown_id_returns_none(dataset: PromptDataset):
    assert dataset.get(10**9) is None


def test_module_level_get_prompt_uses_loaded_dataset():
    init_dataset()
    record = get_prompt(1)
    assert record is not None
    assert record.id == 1
    assert get_prompt(10**9) is None


# --- Validation ------------------------------------------------------------


def test_duplicate_ids_are_rejected(tmp_path):
    path = write_dataset(
        tmp_path,
        [
            {"id": 1, "category": "summarization", "prompt": "First."},
            {"id": 1, "category": "customer_support", "prompt": "Second."},
        ],
    )
    with pytest.raises(DatasetError, match="Duplicate prompt id: 1"):
        load_dataset(path)


def test_missing_field_is_rejected(tmp_path):
    path = write_dataset(tmp_path, [{"id": 1, "category": "summarization"}])
    with pytest.raises(DatasetError, match="missing field 'prompt'"):
        load_dataset(path)


def test_unknown_category_is_rejected(tmp_path):
    path = write_dataset(
        tmp_path, [{"id": 1, "category": "translation", "prompt": "Hello."}]
    )
    with pytest.raises(DatasetError, match="unknown category"):
        load_dataset(path)


def test_empty_prompt_is_rejected(tmp_path):
    path = write_dataset(
        tmp_path, [{"id": 1, "category": "summarization", "prompt": "   "}]
    )
    with pytest.raises(DatasetError, match="empty or non-string prompt"):
        load_dataset(path)


def test_non_integer_id_is_rejected(tmp_path):
    path = write_dataset(
        tmp_path, [{"id": "1", "category": "summarization", "prompt": "Hello."}]
    )
    with pytest.raises(DatasetError, match="non-integer id"):
        load_dataset(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DatasetError, match="Dataset file not found"):
        load_dataset(tmp_path / "does_not_exist.json")


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(path)


def test_empty_dataset_is_rejected(tmp_path):
    path = write_dataset(tmp_path, [])
    with pytest.raises(DatasetError, match="contains no records"):
        load_dataset(path)
