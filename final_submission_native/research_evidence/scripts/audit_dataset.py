#!/usr/bin/env python3
"""Read-only aggregate audit for the prepared dataset CSVs.

Run from /home/foxy/data with:
    python3 audit_dataset.py

For code/value comparisons, None, the empty string, and strings containing
only whitespace are treated as blank. Nonblank strings are compared exactly
as stored; they are not stripped, case-folded, or otherwise normalized.
Labels are stripped for distribution and validation (only 0 and 1 are valid).
The script never prints cell contents or complete rows.
"""

import csv
import itertools
import sys
from pathlib import Path


SPLIT_FILES = {
    "train": Path("data/train.csv"),
    "validation": Path("data/validation.csv"),
    "test": Path("data/test.csv"),
}
REQUIRED_COLUMNS = ("code", "label", "value")
VALID_LABELS = {"0", "1"}


def is_blank(value):
    """Return whether a CSV cell is absent, empty, or whitespace-only."""
    return value is None or value.strip() == ""


def raise_csv_field_limit():
    """Permit long code snippets, bounded by the platform's C integer size."""
    csv.field_size_limit(sys.maxsize)


def inspect_file(name, path):
    """Read one split and return only sets/counts needed for the audit."""
    result = {
        "name": name,
        "path": path,
        "rows": 0,
        "columns": [],
        "missing_required": [],
        "labels": {},
        "blank_labels": 0,
        "unexpected_labels": 0,
        "blank_code": 0,
        "duplicate_code_rows": 0,
        "codes": set(),
        "blank_value": 0,
        "values": set(),
        "duplicate_rows": 0,
        "errors": [],
        "usable": False,
        "complete": False,
    }

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file, strict=True)
            headers = reader.fieldnames
            if headers is None:
                result["errors"].append("missing header row")
                return result
            result["columns"] = headers

            duplicate_headers = sorted(
                header for header in set(headers) if headers.count(header) > 1
            )
            if duplicate_headers:
                result["errors"].append("duplicate column names in header")

            result["missing_required"] = [
                column for column in REQUIRED_COLUMNS if column not in headers
            ]

            if any(column is None or column == "" for column in headers):
                result["errors"].append("blank column name in header")

            result["usable"] = not duplicate_headers and not any(
                column is None or column == "" for column in headers
            )
            seen_rows = set()

            for row in reader:
                result["rows"] += 1
                # DictReader uses a None key for extra cells and None values
                # for missing cells. Count the row, but flag its malformed
                # width and exclude it from comparisons to avoid false results.
                malformed = None in row or any(
                    row.get(column) is None for column in headers
                )
                if malformed:
                    result["errors"].append(
                        f"malformed row width near physical line {reader.line_num}"
                    )
                    continue

                if result["usable"]:
                    exact_row = tuple(row[column] for column in headers)
                    if exact_row in seen_rows:
                        result["duplicate_rows"] += 1
                    else:
                        seen_rows.add(exact_row)

                if "label" in row:
                    label = row["label"].strip()
                    if label == "":
                        result["blank_labels"] += 1
                    else:
                        if label not in VALID_LABELS:
                            result["unexpected_labels"] += 1
                        else:
                            result["labels"][label] = result["labels"].get(label, 0) + 1

                if "code" in row:
                    code = row["code"]
                    if is_blank(code):
                        result["blank_code"] += 1
                    elif result["usable"]:
                        if code in result["codes"]:
                            result["duplicate_code_rows"] += 1
                        result["codes"].add(code)

                if "value" in row:
                    value = row["value"]
                    if is_blank(value):
                        result["blank_value"] += 1
                    elif result["usable"]:
                        result["values"].add(value)

        result["complete"] = True
    except FileNotFoundError:
        result["errors"].append("file not found")
    except (csv.Error, UnicodeError, OSError, OverflowError) as error:
        # Exception messages can contain input text; report only the class.
        result["errors"].append(f"read/parse failure ({type(error).__name__})")

    return result


def print_report(results):
    """Print aggregate-only summaries and return whether the audit is valid."""
    failed = False
    print("DATASET AUDIT (aggregate counts only)")
    print("Comparison rule: whitespace-only is blank; other strings compare exactly.")

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
        print("Required columns:", end=" ")
        print(", ".join(
            f"{column}={'yes' if column in result['columns'] else 'NO'}"
            for column in REQUIRED_COLUMNS
        ))
        print("Valid label counts:", dict(sorted(result["labels"].items())))
        print("Blank labels:", result["blank_labels"])
        print("Unexpected label rows (expected 0 or 1):", result["unexpected_labels"])
        print("Missing/empty code fields:", result["blank_code"])
        print("Duplicate code rows:", result["duplicate_code_rows"])
        print("Blank values:", result["blank_value"])
        print("Duplicate complete rows:", result["duplicate_rows"])

        if result["missing_required"]:
            failed = True
            print("ERROR: missing required columns:", ", ".join(result["missing_required"]))
        if result["errors"]:
            failed = True
            # Avoid verbose parser diagnostics that could include input data.
            for error in result["errors"]:
                print("ERROR:", error)

    print("\nPAIRWISE EXACT NON-EMPTY OVERLAP")
    by_name = {result["name"]: result for result in results}
    for first, second in itertools.combinations(SPLIT_FILES, 2):
        left, right = by_name[first], by_name[second]
        if not left["usable"] or not right["usable"]:
            print(f"{first} vs {second}: unavailable (invalid or missing split)")
            continue
        if "code" not in left["columns"] or "code" not in right["columns"]:
            print(f"{first} vs {second}: code overlap unavailable (column missing)")
        else:
            print(f"{first} vs {second}: shared code snippets = {len(left['codes'] & right['codes'])}")
        if "value" not in left["columns"] or "value" not in right["columns"]:
            print(f"{first} vs {second}: value overlap unavailable (column missing)")
        else:
            print(f"{first} vs {second}: shared values = {len(left['values'] & right['values'])}")

    return failed


def main():
    raise_csv_field_limit()
    results = [inspect_file(name, path) for name, path in SPLIT_FILES.items()]
    return 1 if print_report(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
