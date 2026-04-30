"""Utilities for loading and validating NASA C-MAPSS text files.

Step 1 intentionally stays dependency-free. Later stages can convert these
tables to pandas/numpy objects after the raw file contract is verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


SENSOR_COLUMNS = tuple(f"s{i}" for i in range(1, 22))
COLUMNS = (
    "unit",
    "cycle",
    "op1",
    "op2",
    "op3",
    *SENSOR_COLUMNS,
)


@dataclass(frozen=True)
class CMapssTable:
    """In-memory representation of one C-MAPSS train/test file."""

    path: Path
    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def unit_ids(self) -> tuple[int, ...]:
        return tuple(sorted({int(row[0]) for row in self.rows}))

    @property
    def engine_count(self) -> int:
        return len(self.unit_ids)

    def cycles_by_unit(self) -> dict[int, list[int]]:
        cycles: dict[int, list[int]] = {}
        for row in self.rows:
            unit = int(row[0])
            cycle = int(row[1])
            cycles.setdefault(unit, []).append(cycle)
        return cycles

    def max_cycle_by_unit(self) -> dict[int, int]:
        return {unit: max(cycles) for unit, cycles in self.cycles_by_unit().items()}

    def cycle_length_summary(self) -> dict[str, float]:
        lengths = list(self.max_cycle_by_unit().values())
        return {
            "min": float(min(lengths)),
            "mean": float(mean(lengths)),
            "max": float(max(lengths)),
        }


def load_cmapss_table(path: str | Path) -> CMapssTable:
    """Load a C-MAPSS train/test text file with the expected 26 columns."""

    file_path = Path(path)
    rows: list[tuple[float, ...]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            values = tuple(float(value) for value in stripped.split())
            if len(values) != len(COLUMNS):
                raise ValueError(
                    f"{file_path} line {line_number} has {len(values)} columns; "
                    f"expected {len(COLUMNS)}."
                )
            rows.append(values)

    if not rows:
        raise ValueError(f"{file_path} did not contain any data rows.")

    return CMapssTable(path=file_path, columns=COLUMNS, rows=tuple(rows))


def load_rul_values(path: str | Path) -> tuple[int, ...]:
    """Load a C-MAPSS RUL file containing one integer RUL per test engine."""

    file_path = Path(path)
    values: list[int] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 1:
                raise ValueError(
                    f"{file_path} line {line_number} has {len(parts)} values; "
                    "expected exactly one RUL value."
                )
            values.append(int(float(parts[0])))

    if not values:
        raise ValueError(f"{file_path} did not contain any RUL values.")

    return tuple(values)


def _assert_consecutive(values: Iterable[int], label: str) -> None:
    ordered = list(values)
    expected = list(range(1, len(ordered) + 1))
    if ordered != expected:
        raise ValueError(f"{label} must be consecutive IDs/cycles starting at 1.")


def validate_cmapss_table(table: CMapssTable) -> None:
    """Validate structural invariants shared by train and test files."""

    if table.column_count != 26:
        raise ValueError(f"{table.path} has {table.column_count} columns; expected 26.")

    unit_ids = table.unit_ids
    _assert_consecutive(unit_ids, f"{table.path} unit IDs")

    for unit, cycles in table.cycles_by_unit().items():
        _assert_consecutive(cycles, f"{table.path} cycles for unit {unit}")


def validate_fd001_dataset(data_dir: str | Path) -> dict[str, object]:
    """Load and validate the FD001 split used as the first project milestone."""

    base = Path(data_dir)
    train = load_cmapss_table(base / "train_FD001.txt")
    test = load_cmapss_table(base / "test_FD001.txt")
    rul = load_rul_values(base / "RUL_FD001.txt")

    validate_cmapss_table(train)
    validate_cmapss_table(test)

    expected = {
        "train_rows": 20631,
        "test_rows": 13096,
        "train_engines": 100,
        "test_engines": 100,
        "rul_values": 100,
    }
    actual = {
        "train_rows": train.row_count,
        "test_rows": test.row_count,
        "train_engines": train.engine_count,
        "test_engines": test.engine_count,
        "rul_values": len(rul),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(f"FD001 validation failed: {mismatches}")

    return {
        **actual,
        "columns": COLUMNS,
        "train_cycle_lengths": train.cycle_length_summary(),
        "test_cycle_lengths": test.cycle_length_summary(),
        "rul_min": min(rul),
        "rul_mean": mean(rul),
        "rul_max": max(rul),
    }
