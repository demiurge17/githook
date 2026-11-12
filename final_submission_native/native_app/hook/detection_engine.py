"""Shared candidate policy and frozen inference API. Results never contain code."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
import threading

if __package__:
    from .secret_detector import DetectorError, load_detector, read_config
else:
    from secret_detector import DetectorError, load_detector, read_config


@dataclass(frozen=True)
class Example:
    path: str
    line: int
    code: str = field(repr=False)


def is_selected(path: str, config: dict) -> bool:
    pure = PurePosixPath(path)
    excluded = {str(value).casefold() for value in config["excluded_directory_names"]}
    if any(part.casefold() in excluded for part in pure.parts[:-1]):
        return False
    filename = pure.name.casefold()
    return (filename in {str(value).casefold() for value in config["file_names"]}
            or any(filename.endswith(str(ext).casefold()) for ext in config["file_extensions"]))


def line_examples(path: str, lines, config: dict) -> list[Example]:
    """All nonblank lines are candidates; preserve indentation and text verbatim.

    No regex/entropy filter or heuristic overrides the classifier's hard labels.
    Limits cause an incomplete scan, never a partial clean result.
    """
    examples = []
    for number, code in lines:
        if not code.strip():
            continue
        if len(code.encode("utf-8")) > int(config["max_candidate_bytes"]):
            raise DetectorError("A candidate exceeds the configured size limit; scan incomplete.", code="candidate_limit")
        examples.append(Example(path, number, code))
        if len(examples) > int(config["max_examples"]):
            raise DetectorError("Candidate example count exceeds the configured limit; scan incomplete.", code="count_limit")
    return examples


def classify_examples(examples: list[Example], detector) -> list[dict]:
    """Validate predictions and return only fully masked, location-based results."""
    if not examples:
        return []
    try:
        labels = detector.predict([example.code for example in examples])
        if (not isinstance(labels, list) or len(labels) != len(examples)
                or any(type(label) is not int or label not in (0, 1) for label in labels)):
            raise ValueError
        return [{"path": example.path, "line": example.line,
                 "label": label, "masked": "[REDACTED]"}
                for example, label in zip(examples, labels)]
    except DetectorError as error:
        raise DetectorError("Inference failed; scan incomplete.", code=error.code) from None
    except Exception:
        # Even a third-party/fake detector exception may contain source text.
        raise DetectorError("Inference failed or returned invalid labels; scan incomplete.") from None


class DetectionEngine:
    """Lazy, thread-safe model reuse for application sessions.

    The Git service already owns one FrozenDetector per process. This class uses
    the same loader and classification boundary, without coupling UI to Git.
    """
    def __init__(self, config: dict | None = None, detector_factory=None):
        self.config = dict(config if config is not None else read_config())
        self._factory = detector_factory or load_detector
        self._detector = None
        self._lock = threading.Lock()

    def _classify(self, examples: list[Example]) -> list[dict]:
        if not examples:
            return []
        with self._lock:
            try:
                if self._detector is None:
                    self._detector = self._factory(self.config)
                return classify_examples(examples, self._detector)
            except DetectorError as error:
                raise DetectorError("Model loading or inference failed; scan incomplete.", code=error.code) from None
            except Exception:
                raise DetectorError("Model loading or inference failed; scan incomplete.") from None

    def predict_snippets(self, snippets: list[str]) -> list[int]:
        """Classify complete dataset records, preserving multiline training inputs."""
        if not snippets or len(snippets) > int(self.config["max_examples"]):
            raise DetectorError("Dataset selection exceeds candidate limits.", code="count_limit")
        if any(not isinstance(code, str) or not code.strip() or "\0" in code for code in snippets):
            raise DetectorError("Dataset code records must contain nonblank text.", code="dataset_invalid")
        if any(len(code.encode("utf-8")) > int(self.config["max_candidate_bytes"]) for code in snippets):
            raise DetectorError("A dataset snippet exceeds the candidate size limit.", code="candidate_limit")
        return [row["label"] for row in self._classify(
            [Example("dataset", i, code) for i, code in enumerate(snippets, 1)])]

    def scan_sources(self, sources: list[tuple[str, bytes]]) -> dict:
        """Scan uploaded/source bytes in memory. Do not cache inputs or results."""
        config = self.config
        if len(sources) > int(config["max_files"]):
            raise DetectorError("File count exceeds the configured limit; scan incomplete.", code="count_limit")
        examples, skipped = [], []
        scanned = 0
        for path, raw in sources:
            if not is_selected(path, config):
                skipped.append({"path": path, "reason": "unsupported or excluded path"})
                continue
            if len(raw) > int(config["max_file_bytes"]):
                raise DetectorError("A source file exceeds the configured size limit; scan incomplete.", code="file_limit")
            if b"\0" in raw:
                skipped.append({"path": path, "reason": "binary file"})
                continue
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                raise DetectorError("A source file is not UTF-8; scan incomplete.", code="encoding") from None
            # Split LF/CRLF only, like Git. Other Unicode separators are code.
            lines = ((i, line.removesuffix("\r")) for i, line in enumerate(text.split("\n"), 1))
            examples.extend(line_examples(path, lines, config))
            scanned += 1
            if len(examples) > int(config["max_examples"]):
                raise DetectorError("Candidate count exceeds the configured limit; scan incomplete.", code="count_limit")
        rows = self._classify(examples)
        findings = [row for row in rows if row["label"] == 1]
        return {"files_scanned": scanned, "candidates": rows, "findings": findings,
                "skipped": skipped, "examples_processed": len(examples)}
