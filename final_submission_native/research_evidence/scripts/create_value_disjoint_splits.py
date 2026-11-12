#!/usr/bin/env python3
"""Create a reproducible value-disjoint dataset copy using only stdlib.

Run from /home/foxy/data with:
    python3 create_value_disjoint_splits.py

Inputs are data/{train,validation,test}.csv. New outputs are written to
data/value_disjoint/{train,validation,test}.csv. Existing outputs are never
overwritten. Every exact nonblank value is assigned as one group. Blank-value
rows are independent units and therefore cannot be value-disjoint by ID.

Assignment uses a seeded, largest-group-first greedy heuristic. Each unit is
placed in the split that most reduces a normalized squared-deviation score
against 70/15/15 row targets and per-class targets. This aims for approximate
size and class balance; it does not guarantee globally optimal stratification.
"""

import csv
import itertools
import random
import sys
from collections import defaultdict
from pathlib import Path


SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
INPUT_FILES = {name: Path(f"data/{name}.csv") for name in SPLIT_NAMES}
OUTPUT_DIR = Path("data/value_disjoint")
OUTPUT_FILES = {name: OUTPUT_DIR / f"{name}.csv" for name in SPLIT_NAMES}
REQUIRED_COLUMNS = {"code", "label", "value"}
VALID_LABELS = {"0", "1"}
FRACTIONS = (0.70, 0.15, 0.15)


def is_blank(value):
    return value is None or value.strip() == ""


def read_sources():
    """Validate inputs and collect row/group counts without storing row text."""
    headers = []
    baseline = {}
    group_counts = defaultdict(lambda: [0, 0])
    blank_units = []
    total_rows = 0

    for split_index, name in enumerate(SPLIT_NAMES):
        path = INPUT_FILES[name]
        rows = 0
        labels = {"0": 0, "1": 0}
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle, strict=True)
                file_headers = reader.fieldnames
                if file_headers is None:
                    raise ValueError(f"{path}: missing header row")
                if len(file_headers) != len(set(file_headers)):
                    raise ValueError(f"{path}: duplicate column names")
                missing = REQUIRED_COLUMNS - set(file_headers)
                if missing:
                    raise ValueError(
                        f"{path}: missing required columns: {', '.join(sorted(missing))}"
                    )
                for column in file_headers:
                    if column not in headers:
                        headers.append(column)

                for row_index, row in enumerate(reader):
                    rows += 1
                    if None in row or any(row.get(column) is None for column in file_headers):
                        raise ValueError(
                            f"{path}: malformed row width near physical line {reader.line_num}"
                        )
                    label = row["label"].strip()
                    if label not in VALID_LABELS:
                        raise ValueError(
                            f"{path}: invalid or blank label at data row {row_index + 1}"
                        )
                    labels[label] += 1
                    value = row["value"]
                    label_index = int(label)
                    if is_blank(value):
                        # Each blank-value row gets a unique key and remains
                        # an independent one-row assignment unit.
                        blank_units.append((("blank", split_index, row_index), label_index))
                    else:
                        group_counts[value][label_index] += 1

        except FileNotFoundError as error:
            raise ValueError(f"Required input file not found: {path}") from error
        except (csv.Error, UnicodeError, OSError, OverflowError) as error:
            raise ValueError(
                f"Could not read/parse {path} ({type(error).__name__})"
            ) from error

        baseline[name] = {"rows": rows, "labels": labels}
        total_rows += rows

    return headers, baseline, group_counts, blank_units, total_rows


def objective(counts, targets):
    """Score row totals and class totals against their split targets."""
    score = 0.0
    for split_index in range(3):
        target_total, target_zero, target_one = targets[split_index]
        actual_total, actual_zero, actual_one = counts[split_index]
        for actual, target in (
            (actual_total, target_total),
            (actual_zero, target_zero),
            (actual_one, target_one),
        ):
            denominator = max(target, 1.0)
            score += ((actual - target) / denominator) ** 2
    return score


def assign_units(group_counts, blank_units, baseline, total_rows):
    """Assign every nonblank-value group and each blank row to one split."""
    fractions = FRACTIONS
    targets = []
    total_zero = sum(baseline[name]["labels"]["0"] for name in SPLIT_NAMES)
    total_one = sum(baseline[name]["labels"]["1"] for name in SPLIT_NAMES)
    for fraction in fractions:
        targets.append((
            total_rows * fraction,
            total_zero * fraction,
            total_one * fraction,
        ))

    units = []
    for value, counts in group_counts.items():
        units.append((value, counts[0], counts[1], counts[0] + counts[1]))
    for key, label_index in blank_units:
        zero, one = (1, 0) if label_index == 0 else (0, 1)
        units.append((key, zero, one, 1))

    rng = random.Random(SEED)
    rng.shuffle(units)
    # Stable descending size sort preserves seeded random ordering for ties.
    units.sort(key=lambda unit: unit[3], reverse=True)

    # Track [total rows, label-0 rows, label-1 rows] for each split.
    assigned = [[0, 0, 0] for _ in SPLIT_NAMES]
    # Map each value group (or independent blank-value row) to its chosen split.
    assignments = {}
    for key, zero, one, size in units:
        # Keep this unit together and try placing it in each of the three splits.
        choices = []
        for split_index in range(3):
            # Temporarily add the unit's row and class counts to this split.
            assigned[split_index][0] += size
            assigned[split_index][1] += zero
            assigned[split_index][2] += one
            # Lower scores mean closer agreement with size and class targets
            # across all splits, not just the split being tried.
            score = objective(assigned, targets)
            # Undo the trial placement before evaluating the next split.
            assigned[split_index][0] -= size
            assigned[split_index][1] -= zero
            assigned[split_index][2] -= one
            # The seeded random number breaks equal-score ties reproducibly.
            choices.append((score, rng.random(), split_index))
        # Tuple comparison selects the lowest score, then the tie-break value.
        _, _, chosen = min(choices)
        # Commit the chosen placement to the running counts and lookup map.
        assigned[chosen][0] += size
        assigned[chosen][1] += zero
        assigned[chosen][2] += one
        assignments[key] = chosen

    return assignments, assigned


def write_outputs(headers, assignments):
    """Write every source row once to an exclusive new output file."""
    existing = [str(path) for path in OUTPUT_FILES.values() if path.exists()]
    if existing:
        raise ValueError(
            "Refusing to overwrite existing output file(s): " + ", ".join(existing)
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handles = {}
    writers = {}
    row_counts = {name: 0 for name in SPLIT_NAMES}
    class_counts = {name: {"0": 0, "1": 0} for name in SPLIT_NAMES}

    try:
        for name in SPLIT_NAMES:
            handle = OUTPUT_FILES[name].open(
                "x", newline="", encoding="utf-8"
            )
            handles[name] = handle
            writers[name] = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
            writers[name].writeheader()

        for split_index, source_name in enumerate(SPLIT_NAMES):
            with INPUT_FILES[source_name].open(
                "r", newline="", encoding="utf-8-sig"
            ) as source:
                reader = csv.DictReader(source, strict=True)
                for row_index, row in enumerate(reader):
                    if None in row or any(row.get(column) is None for column in reader.fieldnames):
                        raise ValueError(
                            f"Malformed row width in {INPUT_FILES[source_name]} "
                            f"near physical line {reader.line_num}"
                        )
                    value = row["value"]
                    key = (
                        ("blank", split_index, row_index)
                        if is_blank(value)
                        else value
                    )
                    destination = SPLIT_NAMES[assignments[key]]
                    # Union headers retain all source metadata, filling fields
                    # absent from a particular source schema with empty cells.
                    normalized_row = {column: row.get(column, "") for column in headers}
                    writers[destination].writerow(normalized_row)
                    row_counts[destination] += 1
                    class_counts[destination][row["label"].strip()] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return row_counts, class_counts


def audit_outputs(headers):
    """Return aggregate output overlap counts without displaying cell values."""
    codes = {name: set() for name in SPLIT_NAMES}
    values = {name: set() for name in SPLIT_NAMES}
    for name, path in OUTPUT_FILES.items():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != headers:
                raise ValueError(f"Output schema check failed for {path}")
            for row in reader:
                if None in row or any(row.get(column) is None for column in headers):
                    raise ValueError(f"Malformed output row in {path}")
                if not is_blank(row["code"]):
                    codes[name].add(row["code"])
                if not is_blank(row["value"]):
                    values[name].add(row["value"])

    overlaps = {}
    for first, second in itertools.combinations(SPLIT_NAMES, 2):
        overlaps[(first, second)] = (
            len(codes[first] & codes[second]),
            len(values[first] & values[second]),
        )
    return overlaps


def main():
    csv.field_size_limit(sys.maxsize)
    try:
        headers, baseline, groups, blank_units, total_rows = read_sources()
        if not total_rows:
            raise ValueError("The source CSVs contain no data rows.")
        print("BASELINE SOURCE SPLITS")
        for name in SPLIT_NAMES:
            print(
                f"{name}: rows={baseline[name]['rows']}, "
                f"labels={baseline[name]['labels']}"
            )

        assignments, assigned_counts = assign_units(
            groups, blank_units, baseline, total_rows
        )
        output_rows, output_labels = write_outputs(headers, assignments)
        overlaps = audit_outputs(headers)

        print("\nVALUE-DISJOINT OUTPUT SPLITS")
        for name in SPLIT_NAMES:
            print(
                f"{name}: rows={output_rows[name]}, "
                f"labels={output_labels[name]}"
            )
        print("\nASSIGNMENT DETAILS")
        print("Seed:", SEED)
        print("Unique nonblank value groups:", len(groups))
        print("Rows with blank value assigned independently:", len(blank_units))
        print("Rows preserved:", sum(output_rows.values()), "of", total_rows)
        print("Requested proportions: train=70%, validation=15%, test=15% (approximate)")
        print("\nOUTPUT OVERLAP AUDIT")
        for (first, second), (code_count, value_count) in overlaps.items():
            print(
                f"{first} vs {second}: shared exact code={code_count}, "
                f"shared nonblank values={value_count}"
            )
        print("\nBlank-value rows have no value key to group, so they are not covered by the value-disjoint guarantee.")
        print("No rows were discarded. Origin was retained as metadata, not used as a repository ID.")
        print("Outputs:", ", ".join(str(path) for path in OUTPUT_FILES.values()))

        if sum(output_rows.values()) != total_rows:
            raise ValueError("Output row total does not match source row total.")
        if any(value_count for _, value_count in overlaps.values()):
            raise ValueError("Output audit found a nonblank value shared across splits.")
        return 0
    except (ValueError, csv.Error, UnicodeError, OSError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
