#!/usr/bin/env python3
"""Read-only audit of source and value-disjoint CSV splits.

Run from /home/foxy/data with:
    python3 audit_value_disjoint_splits.py

Only aggregate counts are printed. Blank means an absent, empty, or
whitespace-only cell. Nonblank code and value strings are compared exactly
as stored. Labels are stripped and expected to be 0 or 1. Complete rows are
compared through SHA-256 fingerprints over their ordered cell values so the
audit can verify source-row preservation without retaining or printing rows.
"""

import csv
import hashlib
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")
SOURCE_FILES = {name: Path(f"data/{name}.csv") for name in SPLIT_NAMES}
REVISED_FILES = {
    name: Path(f"data/value_disjoint/{name}.csv") for name in SPLIT_NAMES
}
REQUIRED_COLUMNS = {"code", "label", "value"}
VALID_LABELS = {"0", "1"}


def is_blank(value):
    return value is None or value.strip() == ""


def row_fingerprint(row, headers):
    """Hash ordered cells with length prefixes to avoid ambiguous joins."""
    digest = hashlib.sha256()
    for column in headers:
        cell = row[column].encode("utf-8")
        digest.update(len(cell).to_bytes(8, "big"))
        digest.update(cell)
    return digest.digest()


def new_stats(name, path):
    return {
        "name": name,
        "path": path,
        "headers": None,
        "rows": 0,
        "labels": Counter(),
        "invalid_labels": 0,
        "blank_code": 0,
        "blank_value": 0,
        "codes": set(),
        "duplicate_code_rows": 0,
        "values": set(),
        "value_labels": defaultdict(set),
        "row_fingerprints": Counter(),
        "errors": [],
        "complete": False,
    }


def scan_csv(name, path):
    """Collect audit aggregates for one CSV without exposing cell data."""
    stats = new_stats(name, path)
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames
            if headers is None:
                stats["errors"].append("missing header row")
                return stats
            stats["headers"] = headers
            if len(headers) != len(set(headers)):
                stats["errors"].append("duplicate column names")
                return stats
            missing = REQUIRED_COLUMNS - set(headers)
            if missing:
                stats["errors"].append(
                    "missing required columns: " + ", ".join(sorted(missing))
                )
                return stats

            seen_codes = set()
            for row in reader:
                stats["rows"] += 1
                if None in row or any(row.get(column) is None for column in headers):
                    stats["errors"].append(
                        f"malformed row width near physical line {reader.line_num}"
                    )
                    continue

                stats["row_fingerprints"][row_fingerprint(row, headers)] += 1
                label = row["label"].strip()
                if label not in VALID_LABELS:
                    stats["invalid_labels"] += 1
                else:
                    stats["labels"][label] += 1

                code = row["code"]
                if is_blank(code):
                    stats["blank_code"] += 1
                else:
                    if code in seen_codes:
                        stats["duplicate_code_rows"] += 1
                    seen_codes.add(code)
                    stats["codes"].add(code)

                value = row["value"]
                if is_blank(value):
                    stats["blank_value"] += 1
                else:
                    stats["values"].add(value)
                    if label in VALID_LABELS:
                        stats["value_labels"][value].add(label)

            stats["complete"] = True

    except FileNotFoundError:
        stats["errors"].append("file not found")
    except (csv.Error, UnicodeError, OSError, OverflowError) as error:
        # Exception text is omitted because it can contain raw CSV content.
        stats["errors"].append(f"read/parse failure ({type(error).__name__})")
    return stats


def sum_fingerprints(scans):
    combined = Counter()
    for scan in scans:
        combined.update(scan["row_fingerprints"])
    return combined


def percentage(count, total):
    return 100.0 * count / total if total else 0.0


def print_split_summary(title, scans):
    print(f"\n{title}")
    for scan in scans:
        positives = scan["labels"]["1"]
        negatives = scan["labels"]["0"]
        print(f"{scan['name']}: rows={scan['rows']}")
        print(
            f"  labels: 0 negative={negatives} "
            f"({percentage(negatives, scan['rows']):.2f}%), "
            f"1 positive={positives} "
            f"({percentage(positives, scan['rows']):.2f}%)"
        )
        print(
            f"  blank/invalid labels={scan['invalid_labels']}, "
            f"blank code={scan['blank_code']}, blank value={scan['blank_value']}"
        )
        print("  exact duplicate nonblank code rows:", scan["duplicate_code_rows"])
        print("  values associated with both labels:", sum(
            "0" in labels and "1" in labels
            for labels in scan["value_labels"].values()
        ))


def print_pairwise(title, scans):
    print(f"\n{title}")
    failed = False
    for left, right in itertools.combinations(scans, 2):
        if not left["complete"] or not right["complete"]:
            print(f"{left['name']} vs {right['name']}: unavailable (incomplete CSV)")
            failed = True
            continue
        code_overlap = len(left["codes"] & right["codes"])
        value_overlap = len(left["values"] & right["values"])
        print(
            f"{left['name']} vs {right['name']}: exact code overlap={code_overlap}, "
            f"nonblank value overlap={value_overlap}"
        )
        if code_overlap or value_overlap:
            failed = True
    return failed


def main():
    csv.field_size_limit(sys.maxsize)
    sources = [scan_csv(name, SOURCE_FILES[name]) for name in SPLIT_NAMES]
    revised = [scan_csv(name, REVISED_FILES[name]) for name in SPLIT_NAMES]
    failed = False

    print("VALUE-DISJOINT SPLIT AUDIT (aggregate counts only)")
    print("Blank means empty or whitespace-only; nonblank strings compare exactly.")
    print_split_summary("SOURCE / BASELINE SPLITS", sources)
    print_split_summary("REVISED VALUE-DISJOINT SPLITS", revised)

    source_headers = [tuple(scan["headers"] or ()) for scan in sources]
    revised_headers = [tuple(scan["headers"] or ()) for scan in revised]
    expected_header = source_headers[0] if source_headers else ()
    if any(not scan["complete"] for scan in sources + revised):
        failed = True
    if any(header != expected_header for header in source_headers):
        print("\nERROR: source split schemas do not match.")
        failed = True
    if any(header != expected_header for header in revised_headers):
        print("\nERROR: revised split schemas do not match the source schema.")
        failed = True

    source_total = sum(scan["rows"] for scan in sources)
    revised_total = sum(scan["rows"] for scan in revised)
    print("\nROW PRESERVATION")
    print("Total source rows:", source_total)
    print("Total revised rows:", revised_total)
    print("Row totals match:", "yes" if source_total == revised_total else "NO")
    if source_total != revised_total:
        failed = True
    rows_match = (
        all(scan["complete"] for scan in sources + revised)
        and all(header == expected_header for header in source_headers + revised_headers)
        and sum_fingerprints(sources) == sum_fingerprints(revised)
    )
    print("Source rows preserved exactly once (fingerprint multiset):", "yes" if rows_match else "NO")
    if not rows_match:
        failed = True

    failed |= print_pairwise("REVISED SPLIT LEAKAGE CHECKS", revised)

    # Independently establish that each nonblank value appears in only one
    # revised split. This is equivalent to zero pairwise value overlap, and
    # the explicit count makes the guarantee easy to interpret.
    value_splits = defaultdict(set)
    for scan in revised:
        for value in scan["values"]:
            value_splits[value].add(scan["name"])
    spanning_values = sum(len(split_set) > 1 for split_set in value_splits.values())
    print("\nVALUE GROUPING CHECK")
    print("Distinct nonblank values across revised data:", len(value_splits))
    print("Nonblank values occurring in more than one split:", spanning_values)
    if spanning_values:
        failed = True

    blank_total = sum(scan["blank_value"] for scan in sources)
    print("Blank-valued source rows (independently assigned; not value-groupable):", blank_total)
    if blank_total:
        print("WARNING: value-disjointness cannot be established for blank-valued rows.")

    for scan in sources + revised:
        if scan["invalid_labels"]:
            print(f"WARNING: {scan['name']} has invalid/blank label rows.")
            failed = True
        if scan["errors"]:
            for error in scan["errors"]:
                print(f"ERROR: {scan['name']} {scan['path']}: {error}")
            failed = True

    print("\nInterpretation: value-disjointness prevents exact nonblank values crossing revised splits.")
    print("It does not prove that semantically related examples or repositories are separated.")
    print("Origin was not treated as a repository identifier.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
