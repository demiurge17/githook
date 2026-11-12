#!/usr/bin/env python3
"""CodeBERT embeddings + Random Forest experiments for secret detection.

Run from /home/foxy/data:
    python3 run_experiment.py --mode smoke
    python3 run_experiment.py --mode full

The smoke mode runs a small, deterministic end-to-end sample. Full mode must
be run only after reviewing the smoke outputs. All artifacts go in a new
output directory; existing output paths are never overwritten. CSVs are read
only. Labels and row metadata are saved separately from CodeBERT embeddings.
No snippet, value, or complete CSV row is printed.
"""

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from chunked_embeddings import ChunkStore, atomic_json


SEED = 42
MODEL_NAME = "microsoft/codebert-base"
MODEL_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
MAX_LENGTH = 512
BATCH_SIZE = 16
CHUNK_ROWS = 1024
N_JOBS = 1
CPU_THREADS = 4
SPLIT_INPUTS = {
    "vd_train": Path("data/value_disjoint/train.csv"),
    "vd_validation": Path("data/value_disjoint/validation.csv"),
    "vd_test": Path("data/value_disjoint/test.csv"),
    "baseline_test": Path("data/test.csv"),
}
TRAIN_SPLITS = ("vd_train", "vd_validation")
FINAL_TEST_SPLITS = ("vd_test", "baseline_test")
REQUIRED_COLUMNS = {"code", "label", "label_type"}
VALID_LABELS = {"0", "1"}
FULL_CANDIDATES = (
    {
        "n_estimators": 40,
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": None,
    },
    {
        "n_estimators": 40,
        "max_depth": 20,
        "max_features": 0.25,
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    {
        "n_estimators": 40,
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced",
    },
)


def dependencies():
    """Import optional ML dependencies only when a run is requested."""
    try:
        import numpy as np
        import torch
        import transformers
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import joblib
    except ImportError as error:
        raise RuntimeError(
            "Missing model dependency. Install the pinned requirements into "
            "the project's virtual environment, then retry."
        ) from error
    return np, torch, transformers, RandomForestClassifier, (
        accuracy_score, precision_score, recall_score, f1_score
    ), joblib


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_count(path):
    """Validate a split and return counts without printing any cell values."""
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames
            if headers is None:
                raise ValueError(f"Missing header row: {path}")
            if len(headers) != len(set(headers)):
                raise ValueError(f"Duplicate column names: {path}")
            missing = REQUIRED_COLUMNS - set(headers)
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {', '.join(sorted(missing))}"
                )
            rows = 0
            labels = Counter()
            label_type_evidence = Counter()
            for row in reader:
                rows += 1
                if None in row or any(row.get(column) is None for column in headers):
                    raise ValueError(
                        f"Malformed row width in {path} near physical line {reader.line_num}"
                    )
                label = row["label"].strip()
                if label not in VALID_LABELS:
                    raise ValueError(
                        f"Invalid/blank label in {path} at data row {rows}"
                    )
                if row["code"].strip() == "":
                    raise ValueError(f"Blank code in {path} at data row {rows}")
                labels[label] += 1
                is_negative_type = row["label_type"].strip().startswith("neg_")
                if label == "0" and is_negative_type:
                    label_type_evidence["label_0_negative_type"] += 1
                elif label == "1" and not is_negative_type:
                    label_type_evidence["label_1_secret_type"] += 1
                else:
                    label_type_evidence["mapping_conflict"] += 1
            if not rows or set(labels) != VALID_LABELS:
                raise ValueError(f"Both binary labels and data rows are required in {path}")
            if label_type_evidence["mapping_conflict"]:
                raise ValueError(
                    f"Label mapping could not be confirmed from label_type in {path}"
                )
            return {
                "rows": rows,
                "labels": dict(sorted(labels.items())),
                "label_type_evidence": dict(sorted(label_type_evidence.items())),
                "headers": headers,
            }
    except FileNotFoundError as error:
        raise ValueError(f"Required CSV not found: {path}") from error
    except (csv.Error, UnicodeError, OSError, OverflowError) as error:
        raise ValueError(f"Could not read/parse {path} ({type(error).__name__})") from error


def iter_csv_batches(path, mode, smoke_rows_per_class, batch_size):
    """Stream full splits; retain only the deterministic smoke sample."""
    if mode == "smoke":
        selected = []
        counts = Counter()
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            for row in reader:
                label = row["label"].strip()
                if counts[label] < smoke_rows_per_class:
                    selected.append(row)
                    counts[label] += 1
                if all(counts[label] >= smoke_rows_per_class for label in VALID_LABELS):
                    break
        if any(counts[label] < 2 for label in VALID_LABELS):
            raise ValueError(f"Both classes need at least two rows in smoke sample from {path}")
        for start in range(0, len(selected), batch_size):
            yield selected[start:start + batch_size]
        return

    batch = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        for row in reader:
            if None in row or any(row.get(column) is None for column in reader.fieldnames):
                raise ValueError(
                    f"Malformed row width in {path} near physical line {reader.line_num}"
                )
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def write_json(path, obj):
    atomic_json(path, obj)


def generate_embeddings(np, torch, model, tokenizer, path, split_name,
                        mode, smoke_rows_per_class, expected_rows, batch_size,
                        output_dir, device):
    """Encode a split in small batches and persist features and labels apart."""
    # Full mode uses every row; smoke mode retains a small sample per class.
    row_count = expected_rows
    if mode == "smoke":
        row_count = sum(
            min(smoke_rows_per_class, count)
            for count in validate_and_count(path)["labels"].values()
        )
    # CodeBERT produces one vector with this many dimensions per snippet (768).
    hidden_size = int(model.config.hidden_size)
    # Keep learned features, target labels, and source metadata in separate files.
    feature_path = output_dir / "embeddings" / f"{split_name}.npy"
    label_path = output_dir / "labels" / f"{split_name}.npy"
    metadata_path = output_dir / "metadata" / f"{split_name}.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    # Create file-backed arrays that can be filled batch by batch. On resume,
    # these consolidated arrays are rebuilt from valid chunks and new extraction.
    features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                         shape=(row_count, hidden_size))
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64,
                                       shape=(row_count,))
    # A storage chunk groups several inference batches. Align its size with
    # batch boundaries, targeting CHUNK_ROWS rows (normally 1,024).
    chunk_rows = max(batch_size, (CHUNK_ROWS // batch_size) * batch_size)
    # Record the input identity and extraction settings so incompatible cached
    # chunks cannot be reused with a different source or configuration.
    chunk_config = {
        "input_sha256": source_digest(path), "model": MODEL_NAME,
        "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH, "pooling": "attention-mask-weighted mean of last hidden states",
        "dtype": "float32", "batch_size": batch_size, "chunk_rows": chunk_rows, "mode": mode,
        "smoke_rows_per_class": smoke_rows_per_class if mode == "smoke" else None,
    }
    store = ChunkStore(output_dir, split_name, chunk_config, row_count, hidden_size)

    # Use evaluation behavior, including disabling dropout in the encoder.
    model.eval()
    # cursor includes reused rows; newly_encoded counts computation this attempt.
    cursor = 0
    newly_encoded = 0
    started = time.monotonic()
    next_progress = 5000
    chunk_index = 0
    cached_chunks = set()
    chunk_features = chunk_labels = chunk_ids = None
    # Read selected CSV rows in order and locate their output/chunk positions.
    for batch in iter_csv_batches(path, mode, smoke_rows_per_class, batch_size):
        end = cursor + len(batch)
        chunk_index = cursor // chunk_rows
        chunk_start = chunk_index * chunk_rows
        chunk_end = min(chunk_start + chunk_rows, row_count)
        if chunk_index in cached_chunks:
            # This entire chunk was already restored at its first batch.
            cursor = end
            continue
        if cursor == chunk_start and store.valid(chunk_index, chunk_start, chunk_end, np):
            # Restore a checksummed completed chunk instead of running CodeBERT.
            cached_features, cached_labels = store.chunk_paths(chunk_index)[:2]
            features[chunk_start:chunk_end] = np.load(cached_features, mmap_mode="r", allow_pickle=False)
            labels[chunk_start:chunk_end] = np.load(cached_labels, mmap_mode="r", allow_pickle=False)
            cached_chunks.add(chunk_index)
            cursor = end
            continue
        if chunk_features is None:
            # Buffer one chunk for checkpointing; the final chunk may be smaller.
            chunk_features = np.empty((chunk_end - chunk_start, hidden_size), dtype=np.float32)
            chunk_labels = np.empty((chunk_end - chunk_start,), dtype=np.int64)
            chunk_ids = [None] * (chunk_end - chunk_start)
        # Hash the split, position, code, and label for row traceability without
        # storing raw snippets in the chunk's identifier file.
        row_ids = [hashlib.sha256(
            (split_name + "\0" + str(cursor + i) + "\0" + row["code"] + "\0" + row["label"].strip()).encode("utf-8")
        ).hexdigest() for i, row in enumerate(batch)]
        # Embed only code. Preserve its text, pad to the batch's longest sequence,
        # and truncate to MAX_LENGTH tokens (512), including special tokens.
        encodings = tokenizer(
            [row["code"] for row in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        # Move tokenizer tensors to the same CPU or GPU device as the encoder.
        encodings = {key: value.to(device) for key, value in encodings.items()}
        # Disable gradient tracking: CodeBERT is a fixed feature extractor.
        with torch.inference_mode():
            # Token representations have shape (batch, sequence length, hidden size).
            hidden = model(**encodings).last_hidden_state
            # Expand the mask to (batch, sequence length, 1) for broadcasting.
            mask = encodings["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            # Mean-pool attended positions into one vector per snippet. Padding
            # contributes zero; attended special tokens remain included. Clamping
            # the token count avoids division by zero. Shape: (batch, hidden size).
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        current_batch_size = len(batch)
        # Convert features to CPU float32 and targets to separate int64 labels.
        pooled_array = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
        label_array = np.asarray([int(row["label"].strip()) for row in batch], dtype=np.int64)
        # Matching slices preserve feature/label alignment in consolidated files.
        features[cursor:end] = pooled_array
        labels[cursor:end] = label_array
        # Also place this batch in the current chunk using chunk-relative indices.
        offset = cursor - chunk_start
        chunk_features[offset:offset + current_batch_size] = pooled_array
        chunk_labels[offset:offset + current_batch_size] = label_array
        chunk_ids[offset:offset + current_batch_size] = row_ids
        cursor = end
        newly_encoded += current_batch_size
        if cursor == chunk_end:
            # Save the completed chunk and its checksums, then release its buffer.
            store.write(chunk_index, chunk_start, chunk_features, chunk_labels, chunk_ids, np)
            chunk_features = chunk_labels = chunk_ids = None
        if mode == "full" and cursor >= next_progress:
            # Estimate throughput from newly computed rows and report aggregates.
            elapsed = max(time.monotonic() - started, 0.001)
            rate = newly_encoded / elapsed
            remaining = max(row_count - cursor, 0) / rate
            progress = {"split": split_name, "rows_processed": cursor,
                        "rows_encoded_this_attempt": newly_encoded,
                        "rows_total": row_count, "elapsed_seconds": elapsed,
                        "rows_per_second": rate, "estimated_remaining_seconds": remaining}
            atomic_json(output_dir / "progress.json", progress)
            print(f"  {split_name}: {cursor:,}/{row_count:,} rows; {rate:.2f} rows/s; "
                  f"ETA about {remaining/3600:.1f} h", flush=True)
            next_progress += 5000

    # Require the expected row count and a complete, valid set of stored chunks.
    if cursor != row_count:
        raise ValueError(f"Encoded row count mismatch for {split_name}")

    store.validate_complete(np)
    # Record completion of this split, including how much work was reused.
    elapsed = max(time.monotonic() - started, 0.001)
    atomic_json(output_dir / "progress.json", {
        "split": split_name, "rows_processed": cursor, "rows_total": row_count,
        "rows_encoded_this_attempt": newly_encoded,
        "elapsed_seconds": elapsed,
        "rows_per_second": newly_encoded / elapsed if newly_encoded else 0,
        "estimated_remaining_seconds": 0,
    })

    # Flush consolidated arrays before writing their metadata and returning paths.
    features.flush()
    labels.flush()
    label_counts = Counter(str(int(value)) for value in labels)
    # Save provenance and class counts without including code or candidate values.
    write_json(metadata_path, {
        "split": split_name,
        "input_csv": str(path),
        "input_sha256": source_digest(path),
        "rows_encoded": row_count,
        "row_selection": (
            "all rows in source order" if mode == "full" else
            f"first {smoke_rows_per_class} rows per label in source order"
        ),
        "label_counts": dict(sorted(label_counts.items())),
        "embedding_file": str(feature_path),
        "label_file": str(label_path),
        "metadata_contains_code_or_values": False,
    })
    # The caller loads these arrays for Random Forest fitting or evaluation.
    return feature_path, label_path, row_count, dict(sorted(label_counts.items()))


def calculate_metrics(y_true, y_pred, metric_functions):
    accuracy_score, precision_score, recall_score, f1_score = metric_functions
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    labels = Counter(str(int(value)) for value in y_true)
    return {
        "sample_count": int(len(y_true)),
        "class_counts": dict(sorted(labels.items())),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive_class_1": float(precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "recall_positive_class_1": float(recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "f1_positive_class_1": float(f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        )),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def default_output(mode):
    name = (
        "smoke_valonly_seed42_v1"
        if mode == "smoke"
        else "codebert_rf_seed42_full_v1"
    )
    return Path("artifacts") / name


def main():
    # 1. Read and validate command-line settings.
    # Code snippets can exceed the CSV reader's default per-field length limit.
    csv.field_size_limit(sys.maxsize)
    # argparse converts command-line options into attributes on args. The module
    # docstring (__doc__) supplies the description shown by --help.
    parser = argparse.ArgumentParser(description=__doc__)
    # Smoke checks a small train/validation sample; full uses complete splits
    # and evaluates test sets after selecting the classifier.
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    # type=Path converts a supplied directory string into a filesystem Path.
    parser.add_argument("--output-dir", type=Path)
    # Smoke sampling takes the first requested number of rows from each class.
    parser.add_argument("--smoke-rows-per-class", type=int, default=32)
    # Batch size controls snippets per CodeBERT forward pass and memory demand.
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Reject impossible batch sizes and smoke samples too small for this pipeline.
    # parser.error() displays usage and exits with status 2 before any training.
    if args.batch_size < 1 or args.smoke_rows_per_class < 2:
        parser.error("batch size must be >=1 and smoke rows per class must be >=2")
    # A single conservative batch setting keeps CPU/RAM use bounded.
    batch_size = args.batch_size

    # 2. Protect existing results and require evidence for resuming a full run.
    # With no --output-dir, `or` falls back to the mode's default directory.
    output_dir = args.output_dir or default_output(args.mode)
    # Each smoke run must have a fresh directory so its outputs stay distinct.
    if output_dir.exists() and args.mode == "smoke":
        raise SystemExit(f"Refusing to overwrite existing artifact directory: {output_dir}")
    if output_dir.exists() and args.mode == "full":
        # The marker identifies unfinished output; its absence prevents this
        # invocation from overwriting a completed or unmarked experiment.
        marker = output_dir / "INCOMPLETE.txt"
        if not marker.is_file():
            raise SystemExit(f"Refusing to reuse existing full output without INCOMPLETE.txt: {output_dir}")
        if not (output_dir / "run_config.json").is_file():
            raise SystemExit(f"Cannot safely resume legacy partial output without run_config.json: {output_dir}")

    # Expected data, dependency, and filesystem failures below are reported by
    # the except block. Argument/path refusals above exit before entering it.
    try:
        # 3. Validate development data before loading expensive model assets.
        # Model development reads only train/validation. Test CSVs are first
        # opened after the selected classifier has been frozen and saved.
        # This dictionary comprehension maps each split name to checked row/class
        # counts. The helper checks schema, row widths, code, labels, and label_type.
        inspected = {
            name: validate_and_count(SPLIT_INPUTS[name]) for name in TRAIN_SPLITS
        }
        print("Validated CSV inputs (rows and labels only):")
        for name, info in inspected.items():
            print(f"  {name}: rows={info['rows']}, labels={info['labels']}")
        print("Label mapping verified: label 0 has negative label_type; label 1 has secret label_type.")

        # 4. Load libraries and set up reproducible execution.
        # Tuple unpacking assigns the helper's returned libraries/tools in order.
        np, torch, transformers, RandomForestClassifier, metric_functions, joblib = dependencies()
        # Seed each library's random generator; forest fitting also receives SEED.
        # Seeds support repeatability but do not guarantee identical bits across
        # different devices, library versions, or runtime implementations.
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        # Limit PyTorch CPU parallelism to the configured number of threads.
        torch.set_num_threads(CPU_THREADS)
        # Use a CUDA GPU when available, otherwise the CPU. Encoder parameters
        # and input tensors must be placed on the same device.
        if torch.cuda.is_available():
            # Seed the generators on every available CUDA device too.
            torch.cuda.manual_seed_all(SEED)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Device: {device}; mode: {args.mode}; batch size: {batch_size}")

        # 5. Record run identity and mark outputs as unfinished.
        # Create a fresh run directory only after input/dependency checks pass.
        # parents=True creates missing parents; exist_ok=True permits safe resume.
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_path = output_dir / "INCOMPLETE.txt"
        # Source hashes identify the CSV bytes; pinned revisions/settings identify
        # the feature transformation used to produce reusable embedding chunks.
        run_config = {
            "mode": args.mode, "seed": SEED, "batch_size": batch_size,
            "model": MODEL_NAME, "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION, "max_length": MAX_LENGTH,
            "pooling": "attention-mask-weighted mean of last hidden states",
            "input_sha256": {name: source_digest(SPLIT_INPUTS[name]) for name in TRAIN_SPLITS},
        }
        run_config_path = output_dir / "run_config.json"
        if run_config_path.exists():
            # Parse the saved JSON back into a dictionary and require an exact
            # settings match before reusing outputs from an interrupted run.
            if json.loads(run_config_path.read_text()) != run_config:
                raise ValueError("Existing run configuration does not match; refusing resume")
        else:
            write_json(run_config_path, run_config)
        if not marker_path.exists():
            # Leave this marker in place until the final record is saved.
            marker_path.write_text("Run incomplete; do not use outputs until the run completes.\n")
        # Separate features, targets, provenance, and trained classifiers on disk.
        for directory in ("embeddings", "labels", "metadata", "models"):
            (output_dir / directory).mkdir(exist_ok=True)

        # 6. Load CodeBERT for fixed feature extraction.
        # The tokenizer converts code text into token IDs and attention masks.
        # Pinning its revision with the encoder avoids silently changing assets.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        )
        # AutoModel loads pretrained token representations, not a secret classifier.
        # .to(device) moves its parameters to the selected CPU or GPU.
        encoder = transformers.AutoModel.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        ).to(device)
        # Hidden size is the dimension of each token and pooled snippet vector
        # (768 for this model); it is later saved with the experiment settings.
        hidden_size = int(encoder.config.hidden_size)

        # Tokenize with padding/truncation to 512 tokens, run inference without
        # updating CodeBERT, and mean-pool attended token vectors per snippet.
        # The helper saves float32 features, separate labels, and checked chunks;
        # on resume, valid chunks skip encoder work. encoded stores output paths
        # and row/class counts for each development split, not all vectors in RAM.
        encoded = {}
        for split_name in TRAIN_SPLITS:
            encoded[split_name] = generate_embeddings(
                np, torch, encoder, tokenizer, SPLIT_INPUTS[split_name], split_name,
                args.mode, args.smoke_rows_per_class, inspected[split_name]["rows"],
                batch_size, output_dir, device
            )

        # 7. Reduce memory pressure before training the classifier.
        # Release the encoder before fitting forests; this machine has limited
        # available RAM. The identical pinned encoder is reloaded for test data.
        # del removes this reference; gc.collect() asks Python to collect garbage.
        del encoder
        gc.collect()
        if device.type == "cuda":
            # Release unused cached GPU allocations, not memory of live tensors.
            torch.cuda.empty_cache()

        # 8. Train Random Forest candidates and measure validation performance.
        # Keep the best settings/metrics and a result record for every candidate.
        best_candidate = None
        configurations = []
        # Each helper result contains feature/label paths at indices 0 and 1.
        # x contains one feature vector per row; y contains its binary target.
        # Read-only memory maps give file-backed access, though fitting can still
        # allocate additional memory. Validation rows are never passed to fit().
        x_train = np.load(encoded["vd_train"][0], mmap_mode="r")
        y_train = np.load(encoded["vd_train"][1], mmap_mode="r")
        x_validation = np.load(encoded["vd_validation"][0], mmap_mode="r")
        y_validation = np.load(encoded["vd_validation"][1], mmap_mode="r")
        # enumerate() yields a zero-based candidate index and its parameter dict.
        # FULL_CANDIDATES specifies tree count/depth, feature sampling, leaf size,
        # and class weighting. Candidates differ in feature sampling, leaf size,
        # and class weighting; fitting learns the trees from these settings.
        for candidate_index, full_config in enumerate(FULL_CANDIDATES):
            # Copy the dictionary so smoke adjustments do not alter the constants.
            config = dict(full_config)
            # This conditional expression chooses fewer trees for a quick smoke
            # check, or the configured full tree count for the actual experiment.
            config["n_estimators"] = 8 if args.mode == "smoke" else full_config["n_estimators"]
            # **config expands its entries into this dictionary. Later entries
            # add a repeatable forest seed and the number of parallel fit workers.
            configuration = {
                **config,
                "random_state": SEED,
                "n_jobs": N_JOBS,
            }
            # **configuration passes dictionary entries as keyword arguments,
            # equivalent to naming each constructor setting individually.
            model = RandomForestClassifier(**configuration)
            # fit() learns decision trees from training embeddings and labels.
            model.fit(x_train, y_train)
            # predict() applies the trained trees to validation features, returning
            # hard labels; calculate_metrics() compares them with validation truth.
            validation_prediction = model.predict(x_validation)
            validation_metrics = calculate_metrics(
                y_validation, validation_prediction, metric_functions
            )
            candidate_result = {
                "candidate_index": candidate_index,
                "configuration": configuration,
                "validation_metrics": validation_metrics,
            }
            configurations.append(candidate_result)
            # 9. Select using validation data; test scores must not guide tuning.
            # Positive-class F1 balances precision and recall for secrets (label 1).
            # Python compares tuples left to right: higher F1 wins, then higher
            # precision (fewer false alarms relative to positive predictions).
            # Negating the index favors the earlier candidate if both scores tie.
            selection_key = (
                validation_metrics["f1_positive_class_1"],
                validation_metrics["precision_positive_class_1"],
                -candidate_index,
            )
            # Accept the first candidate, then replace it only with a better key.
            if best_candidate is None or selection_key > best_candidate[0]:
                best_candidate = (
                    selection_key, configuration, validation_metrics
                )
            # Keep settings rather than every fitted forest to reduce memory use.
            del model
            gc.collect()
            print(
                f"Validation candidate {candidate_index + 1}: "
                f"F1={validation_metrics['f1_positive_class_1']:.4f}, "
                f"precision={validation_metrics['precision_positive_class_1']:.4f}"
            )

        # 10. Fit and save the selected configuration as the final classifier.
        # Select by positive-class F1, then precision, then candidate order.
        # Unpack the winner's tuple; `_` discards the now-unneeded selection key.
        _, final_configuration, selected_validation = best_candidate
        # Candidate models were discarded. Fit a fresh forest with the winning
        # settings on training rows only; validation is not merged into training.
        final_model = RandomForestClassifier(**final_configuration)
        x_train = np.load(encoded["vd_train"][0], mmap_mode="r")
        y_train = np.load(encoded["vd_train"][1], mmap_mode="r")
        final_model.fit(x_train, y_train)
        model_path = output_dir / "models" / "random_forest.joblib"
        # joblib serializes the fitted forest so evaluation/the hook can reuse it
        # without fitting again. Saving now fixes the model before test access.
        joblib.dump(final_model, model_path)
        # Drop array references before reloading the encoder for test extraction.
        del x_train, y_train, x_validation, y_validation
        gc.collect()

        # 11. Reload the same pinned feature transformation for final testing.
        # Reloading restores the encoder released before forest fitting; it does
        # not train it. This reload also occurs in smoke mode, before its empty
        # test loop, so smoke still performs no test CSV access.
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        )
        encoder = transformers.AutoModel.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION
        ).to(device)

        # Only now encode/evaluate either test set. Test data is not used for
        # model selection or configuration changes.
        evaluations = {}
        test_inspected = {}
        # A conditional expression returns both final split names in full mode,
        # or an empty tuple in smoke mode so the following loop runs zero times.
        final_splits = FINAL_TEST_SPLITS if args.mode == "full" else ()
        if args.mode == "smoke":
            print("Smoke mode: test CSVs were not opened or evaluated.")
        for split_name in final_splits:
            # Apply the same input checks and frozen embedding procedure as train.
            test_inspected[split_name] = validate_and_count(SPLIT_INPUTS[split_name])
            # Tuple unpacking retrieves the helper's paths, row count, and class
            # counts in their return order; labels remain separate from features.
            feature_path, label_path, row_count, label_counts = generate_embeddings(
                np, torch, encoder, tokenizer, SPLIT_INPUTS[split_name], split_name,
                args.mode, args.smoke_rows_per_class, test_inspected[split_name]["rows"],
                batch_size, output_dir, device
            )
            x_test = np.load(feature_path, mmap_mode="r")
            y_test = np.load(label_path, mmap_mode="r")
            # Predict using the saved selection; no fitting or tuning occurs here.
            # The value-disjoint test holds out exact values. The original test
            # is a comparison and may overlap the repartitioned training pool.
            prediction = final_model.predict(x_test)
            # Compare hard labels with truth: accuracy is overall correctness;
            # secret precision/recall measure false alarms and missed secrets,
            # and F1 is their harmonic mean. Also count TP, FP, TN, and FN.
            metrics = calculate_metrics(y_test, prediction, metric_functions)
            metrics["source_csv"] = str(SPLIT_INPUTS[split_name])
            evaluations[split_name] = metrics
            print(
                f"Final evaluation {split_name}: n={row_count}, labels={label_counts}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"precision={metrics['precision_positive_class_1']:.4f}, "
                f"recall={metrics['recall_positive_class_1']:.4f}, "
                f"F1={metrics['f1_positive_class_1']:.4f}, "
                f"TP={metrics['confusion_matrix']['TP']}, "
                f"FP={metrics['confusion_matrix']['FP']}, "
                f"TN={metrics['confusion_matrix']['TN']}, "
                f"FN={metrics['confusion_matrix']['FN']}"
            )

        # 12. Save enough provenance to interpret and reproduce this experiment.
        # Library versions affect tokenization, numerical behavior, and model
        # serialization; record the actual installed versions rather than guesses.
        package_versions = {}
        for package in ("torch", "transformers", "scikit-learn", "numpy", "joblib", "tokenizers"):
            try:
                package_versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                package_versions[package] = "not installed"

        # experiment.json combines the environment, feature settings, candidate
        # selection, final classifier identity, and validation/test measurements.
        configuration_record = {
            "mode": args.mode,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "seed": SEED,
            "device": str(device),
            "embedding_model": MODEL_NAME,
            "embedding_model_revision": MODEL_REVISION,
            "max_length_tokens": MAX_LENGTH,
            "truncation": True,
            "pooling": "attention-mask-weighted mean of last hidden states, including attended special tokens",
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "cpu_threads": CPU_THREADS,
            "embedding_dtype": "float32",
            "labels_separate_from_embeddings": True,
            "label_mapping": {"0": "non-secret", "1": "secret"},
            "training_split": "vd_train",
            "selection_split": "vd_validation",
            "selection_metric": "positive-class F1; precision tie-break; candidate order final tie-break",
            "selected_random_forest": final_configuration,
            "validation_metrics_for_selected_configuration": selected_validation,
            "candidate_results": configurations,
            "test_splits_evaluated_after_selection": list(final_splits),
            "test_results": evaluations,
            "model_file": str(model_path),
            # This nested comprehension builds a per-input provenance dictionary.
            # SHA-256 fingerprints the file bytes to detect input changes; it
            # establishes file identity, not correctness of the dataset labels.
            "input_files": {
                name: {
                    "path": str(path),
                    "sha256": source_digest(path),
                    # Use development stats when present; `or` otherwise uses
                    # test stats. Only splits actually inspected are included.
                    "rows": (inspected.get(name) or test_inspected[name])["rows"],
                    "labels": (inspected.get(name) or test_inspected[name])["labels"],
                }
                # ** unpacks and merges these two split-to-path dictionaries;
                # .items() then yields the name/path pairs for the outer loop.
                for name, path in {
                    **{name: SPLIT_INPUTS[name] for name in inspected},
                    **{name: SPLIT_INPUTS[name] for name in test_inspected},
                }.items()
            },
            "test_interpretation": "Value-disjoint test assesses unseen exact values, not repository or semantic separation.",
        }
        # The JSON helper writes via a temporary file and atomic replacement.
        write_json(output_dir / "experiment.json", configuration_record)
        # Remove the unfinished marker only after the final record is saved.
        # missing_ok=True tolerates an already-absent marker without an exception.
        marker_path.unlink(missing_ok=True)
        print("\nSaved embeddings, labels, metadata, classifier, and experiment record under:", output_dir)
        if args.mode == "smoke":
            print("SMOKE RUN ONLY: do not interpret these sample metrics as model performance.")
        # A zero exit status reports success to the shell or calling process.
        return 0

    # 13. Report expected failures and return a nonzero process status.
    # The matching exception is bound to error; stderr separates errors from
    # normal output. An unfinished run retains its marker once it was created.
    except (RuntimeError, ValueError, OSError, csv.Error, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        # The unchanged entry point below calls raise SystemExit(main()): it
        # runs main() for direct script execution and turns this return value
        # into the process exit status (1 for these failures, 0 for success).
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
