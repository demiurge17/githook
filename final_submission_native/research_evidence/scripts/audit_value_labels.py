#!/usr/bin/env python3
"""Read-only, aggregate audit of value/label overlap across dataset splits.

Run from /home/foxy/data with:
    python3 audit_value_labels.py

Blank means a missing, empty, or whitespace-only cell. Blank values are
excluded from value calculations. Nonblank values are compared exactly as
stored (without trimming or case normalization). Labels are stripped and
must be exactly "0" or "1" after stripping. Invalid rows are counted and
excluded from label/value associations; any invalid row makes the command
exit nonzero. No cell contents or complete rows are printed.
"""

import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path


SPLIT_FILES = {
    "train": Path("data/train.csv"),
    "validation": Path("data/validation.csv"),
    "test": Path("data/test.csv"),
}
REQUIRED_COLUMNS = ("label", "value")
VALID_LABELS = {"0", "1"}


def is_blank(value):
    """Return whether a CSV cell is absent, empty, or whitespace-only."""
    return value is None or value.strip() == ""


def inspect_split(name, path):
    """Return aggregate counts and value-to-label sets for one CSV."""
    result = {
        "name": name,
        "path": path,
        "rows": 0,
        "columns": [],
        "value_labels": defaultdict(set),
        "invalid_label_rows": 0,
        "malformed_rows": 0,
        "errors": [],
        "complete": False,
        "valid_schema": False,
    }

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file, strict=True)
            headers = reader.fieldnames
            if headers is None:
                result["errors"].append("missing header row")
                return result

            result["columns"] = headers
            duplicate_headers = len(headers) != len(set(headers))
            if duplicate_headers:
                result["errors"].append("duplicate column names in header")
            missing = [column for column in REQUIRED_COLUMNS if column not in headers]
            result["missing_columns"] = missing
            result["valid_schema"] = not duplicate_headers and not missing

            if missing:
                result["errors"].append(
                    "missing required columns: " + ", ".join(missing)
                )
                return result

            for row in reader:
                result["rows"] += 1
                malformed = None in row or any(
                    row.get(column) is None for column in headers
                )
                if malformed:
                    result["malformed_rows"] += 1
                    continue

                label = row["label"].strip()
                if label not in VALID_LABELS:
                    result["invalid_label_rows"] += 1
                    continue

                value = row["value"]
                if not is_blank(value):
                    result["value_labels"][value].add(label)

            result["complete"] = True

    except FileNotFoundError:
        result["errors"].append("file not found")
    except (csv.Error, UnicodeError, OSError, OverflowError) as error:
        # Do not print exception text; it may include input data.
        result["errors"].append(f"read/parse failure ({type(error).__name__})")

    return result


def split_counts(value_labels):
    """Summarize unique values and their associated binary labels."""
    values_zero = sum("0" in labels for labels in value_labels.values())
    values_one = sum("1" in labels for labels in value_labels.values())
    both_labels = sum(
        "0" in labels and "1" in labels for labels in value_labels.values()
    )
    return len(value_labels), values_zero, values_one, both_labels


def pair_counts(left, right):
    """Summarize shared values using each side's set of observed labels."""
    shared = left.keys() & right.keys()
    both_positive = sum("1" in left[value] and "1" in right[value] for value in shared)
    both_negative = sum("0" in left[value] and "0" in right[value] for value in shared)
    opposite = sum(
        ("1" in left[value] and "0" in right[value])
        or ("0" in left[value] and "1" in right[value])
        for value in shared
    )
    any_positive = sum(
        "1" in left[value] or "1" in right[value] for value in shared
    )
    any_negative = sum(
        "0" in left[value] or "0" in right[value] for value in shared
    )
    return len(shared), both_positive, both_negative, opposite, any_positive, any_negative


def print_report(results):
    failed = False
    print("VALUE/LABEL OVERLAP AUDIT (aggregate counts only)")
    print("Blank values are excluded; nonblank values are compared exactly.")
    print("Labels are stripped and must be 0 or 1.")

    for result in results:
        print(f"\n{result['name'].upper()} — {result['path']}")
        if result["columns"]:
            print("Columns:", ", ".join(result["columns"]))
        else:
            print("Columns: unavailable")
        if result["complete"]:
            print("Rows:", result["rows"])
        else:
            print("Rows parsed before failure:", result["rows"])

        if result["valid_schema"]:
            unique, with_zero, with_one, both = split_counts(result["value_labels"])
            print("Nonblank unique values:", unique)
            print("Unique values associated with label 0:", with_zero)
            print("Unique values associated with label 1:", with_one)
            print("Values associated with both labels in this split:", both)
        else:
            print("Value/label counts: unavailable (invalid or missing schema)")

        print("Rows with blank/invalid labels:", result["invalid_label_rows"])
        print("Malformed-width rows:", result["malformed_rows"])

        if result["invalid_label_rows"] or result["malformed_rows"]:
            failed = True
            print("NOTE: invalid rows were counted and excluded from value associations.")
        if result["errors"]:
            failed = True
            for error in result["errors"]:
                print("ERROR:", error)

    print("\nPAIRWISE SHARED NONBLANK VALUES")
    print("Opposite-label count means a value has label 1 on one side and 0 on the other.")
    by_name = {result["name"]: result for result in results}
    for first, second in itertools.combinations(SPLIT_FILES, 2):
        left, right = by_name[first], by_name[second]
        if not left["complete"] or not right["complete"] or not left["valid_schema"] or not right["valid_schema"]:
            print(f"{first} vs {second}: unavailable (incomplete or invalid split)")
            continue
        shared, pos_both, neg_both, opposite, any_pos, any_neg = pair_counts(
            left["value_labels"], right["value_labels"]
        )
        print(f"{first} vs {second}:")
        print("  Shared values, any label:", shared)
        print("  Label 1 in both splits:", pos_both)
        print("  Label 0 in both splits:", neg_both)
        print("  Opposite labels across splits:", opposite)
        print("  Any positive occurrence across the pair:", any_pos)
        print("  Any negative occurrence across the pair:", any_neg)

    print("\nInterpretation: overlap identifies repeated annotated strings, not proof of leaked secrets.")
    print("Negative examples can also have values; origin is not treated as a repository ID.")
    print("No rows or splits were modified.")
    return failed


def main():
    # Raise the default CSV field limit so long snippets/contexts do not fail
    # merely because of the module's small default limit.
    csv.field_size_limit(sys.maxsize)
    results = [inspect_split(name, path) for name, path in SPLIT_FILES.items()]
    return 1 if print_report(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
