"""Prompt dataset: loading, validation and in-memory lookup.

Responsibility is deliberately narrow - this module knows about the prompt
records and nothing else. It has no awareness of tokenization, scheduling,
GPUs or metrics.

The dataset is read from disk once at application startup and then served
from memory, keeping file I/O out of request timing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts.json"

VALID_CATEGORIES = frozenset(
    {"summarization", "document_understanding", "customer_support"}
)

REQUIRED_FIELDS = ("id", "category", "prompt")


class DatasetError(ValueError):
    """Raised when the dataset file is missing, malformed or inconsistent.

    Deliberately fatal: a simulator running against a half-valid workload
    would produce numbers nobody should trust, so we fail at startup rather
    than silently skipping bad records.
    """


@dataclass(frozen=True)
class PromptRecord:
    """One unit of simulated workload."""

    id: int
    category: str
    prompt: str


class PromptDataset:
    """An immutable, ID-indexed collection of prompt records."""

    def __init__(self, records: list[PromptRecord]) -> None:
        self._records = records
        self._by_id = {record.id: record for record in records}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[PromptRecord]:
        return list(self._records)

    @property
    def ids(self) -> list[int]:
        return [record.id for record in self._records]

    def get(self, prompt_id: int) -> PromptRecord | None:
        """Return the record, or None when the ID is unknown.

        Returning None rather than raising keeps the caller in control: the
        API layer maps an unknown ID to HTTP 404.
        """
        return self._by_id.get(prompt_id)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.category] = counts.get(record.category, 0) + 1
        return counts


def _validate_record(raw: object, index: int) -> PromptRecord:
    """Validate a single raw record and convert it to a PromptRecord."""
    if not isinstance(raw, dict):
        raise DatasetError(f"Record at index {index} is not a JSON object.")

    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise DatasetError(f"Record at index {index} is missing field '{field}'.")

    record_id = raw["id"]
    # bool is a subclass of int in Python, so it is excluded explicitly.
    if isinstance(record_id, bool) or not isinstance(record_id, int):
        raise DatasetError(
            f"Record at index {index} has a non-integer id: {record_id!r}."
        )

    category = raw["category"]
    if category not in VALID_CATEGORIES:
        raise DatasetError(
            f"Record id={record_id} has unknown category {category!r}. "
            f"Expected one of {sorted(VALID_CATEGORIES)}."
        )

    prompt = raw["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise DatasetError(f"Record id={record_id} has an empty or non-string prompt.")

    return PromptRecord(id=record_id, category=category, prompt=prompt)


def load_dataset(path: Path | str | None = None) -> PromptDataset:
    """Read, validate and index the dataset file.

    Raises DatasetError for anything that would make the workload unreliable:
    a missing file, invalid JSON, a malformed record or a duplicate ID.
    """
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.exists():
        raise DatasetError(f"Dataset file not found: {dataset_path}")

    try:
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Dataset file is not valid JSON: {exc}") from exc

    if not isinstance(raw_data, list):
        raise DatasetError("Dataset file must contain a JSON array of records.")
    if not raw_data:
        raise DatasetError("Dataset file contains no records.")

    records: list[PromptRecord] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(raw_data):
        record = _validate_record(raw, index)
        # Duplicate IDs would make lookup ambiguous, so they are rejected
        # rather than resolved by last-write-wins.
        if record.id in seen_ids:
            raise DatasetError(f"Duplicate prompt id: {record.id}")
        seen_ids.add(record.id)
        records.append(record)

    return PromptDataset(records)


# --------------------------------------------------------------------------
# Module-level dataset held for the lifetime of the process.
# --------------------------------------------------------------------------

_dataset: PromptDataset | None = None


def init_dataset(path: Path | str | None = None) -> PromptDataset:
    """Load the dataset into memory. Called once at application startup."""
    global _dataset
    _dataset = load_dataset(path)
    return _dataset


def get_dataset() -> PromptDataset:
    """Return the loaded dataset, or fail loudly if startup did not run."""
    if _dataset is None:
        raise DatasetError("Dataset has not been loaded. Call init_dataset() first.")
    return _dataset


def get_prompt(prompt_id: int) -> PromptRecord | None:
    """Look up a prompt by ID, or None if no such prompt exists.

    Returns the full record because completion sizing needs the category as
    well as the prompt text.
    """
    return get_dataset().get(prompt_id)
