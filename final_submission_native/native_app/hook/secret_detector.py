"""Offline CodeBERT + frozen Random Forest inference for the Git hook."""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "hook" / "config.json"


class DetectorError(RuntimeError):
    """Safe-to-display setup or inference failure."""

    def __init__(self, message: str, *, code: str = "scan_failed"):
        super().__init__(message)
        self.code = code


def public_error_message(error: Exception) -> str:
    """Render a fixed diagnostic; never display arbitrary exception text."""
    messages = {
        "cache_missing": "The configured CodeBERT cache is missing or lacks the pinned snapshot. Restart with HF_HUB_CACHE and TRANSFORMERS_CACHE unset to use the project's prepared assets, or set them to the correct cache directory.",
        "memory_startup": "Available memory is below the 1,536 MiB model-loading guard. Close memory-heavy applications and retry.",
        "memory_inference": "Available memory is below the 768 MiB inference guard. Close memory-heavy applications and retry.",
        "memory_check": "Available memory could not be checked. Run the scanner in its supported Linux or WSL environment.",
        "model_load": "The frozen model or local checkpoint could not be loaded. Verify the application environment and local model assets.",
        "file_limit": "A source file exceeds the configured file-size limit.",
        "candidate_limit": "A candidate line exceeds the configured byte limit.",
        "count_limit": "The input exceeds the configured file or candidate count limit. Scan a smaller selection.",
        "encoding": "A source file is not valid UTF-8. Save it as UTF-8 and retry.",
        "dataset_invalid": "The dataset is invalid or its selected columns contain missing or incompatible values. Use UTF-8 CSV, JSON records or JSONL with nonblank code and, when evaluating, binary labels.",
        "dataset_code_column": "No supported code column was selected. Name the column that contains code: code, text, snippet, content or source_code. Metadata cannot be used as model input.",
        "dataset_label_column": "Use label, label_binary, target or is_secret for ground-truth labels. Metadata cannot be used as ground truth.",
        "dataset_size": "The dataset exceeds the upload or row limit. Upload at most 100 MiB and 1,000,000 records; choose up to 500 records for one model run.",
        "dataset_upload_size": "The uploaded dataset exceeds 100 MiB. Split the file into smaller datasets before uploading.",
        "dataset_row_limit": "The uploaded dataset contains more than 1,000,000 records. Split it into smaller datasets before uploading.",
        "dataset_parser": "The streaming JSON parser is unavailable. Install the project requirements-app.txt dependencies and restart Streamlit.",
    }
    code = error.code if isinstance(error, DetectorError) and type(error.code) is str else ""
    reason = messages.get(code, "Model loading, input validation or inference failed. Check the environment, local assets and documented limits.")
    return "Scan incomplete. " + reason + " No clean result is available."


def available_memory_mib() -> float | None:
    """Use the smaller of host MemAvailable and cgroup memory headroom.

    Docker's /proc/meminfo can describe the host rather than the container.
    Count charged file cache conservatively; never weaken the existing guards.
    """
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) / 1024.0
                return min([host, *_cgroup_headroom_mib()])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_headroom_mib() -> list[float]:
    """Support cgroup v2 and v1, including visible ancestor limits."""
    roots = [(Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
             (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")]
    groups = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    headroom = []
    for root, limit_name, usage_name in roots:
        candidates = {root}
        for group in groups:
            _, controllers, relative = group.split(":", 2)
            if (limit_name == "memory.max" and not controllers) or "memory" in controllers.split(","):
                path = root / relative.lstrip("/")
                if ".." not in path.parts:
                    candidates.update([path, *[p for p in path.parents if p == root or root in p.parents]])
        for directory in candidates:
            limit_path = directory / limit_name
            if not limit_path.is_file():
                continue
            raw = limit_path.read_text(encoding="ascii").strip()
            if raw == "max":
                continue
            limit = int(raw)
            # v1 uses a huge integer for unlimited memory.
            if limit >= (1 << 60):
                continue
            usage = int((directory / usage_name).read_text(encoding="ascii"))
            headroom.append(max(0, limit - usage) / (1024 * 1024))
    return headroom


def require_memory(config: dict, *, during_inference: bool = False) -> None:
    available = available_memory_mib()
    threshold = float(config[
        "min_memavailable_during_mib" if during_inference else "min_memavailable_mib"
    ])
    if available is None:
        raise DetectorError("Available memory could not be checked; scan incomplete.", code="memory_check")
    if available < threshold:
        raise DetectorError("Available memory is below the configured scan guard; scan incomplete.",
                            code="memory_inference" if during_inference else "memory_startup")


def read_config(path: Path = DEFAULT_CONFIG) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise DetectorError("Hook configuration is missing or invalid.") from None
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenDetector:
    def __init__(self, config: dict, classifier, tokenizer, encoder, np, torch):
        self.config = config
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.np = np
        self.torch = torch
        self.inference_seconds = 0.0
        self.tokenization_seconds = 0.0
        self.encoder_seconds = 0.0
        self.classifier_seconds = 0.0
        self.encoder.eval()

    def predict(self, code_examples: list[str]) -> list[int]:
        """Return the classifier's established hard predictions in input order."""
        predictions: list[int] = []
        started = time.monotonic()
        batch_size = int(self.config["batch_size"])
        try:
            for start in range(0, len(code_examples), batch_size):
                require_memory(self.config, during_inference=True)
                # Preserve the exact candidate text. The training tokenizer sees
                # CSV code fields verbatim; trimming here would alter features.
                batch = code_examples[start:start + batch_size]
                stage_started = time.monotonic()
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=int(self.config["max_length"]),
                    return_tensors="pt",
                )
                self.tokenization_seconds += time.monotonic() - stage_started
                stage_started = time.monotonic()
                with self.torch.inference_mode():
                    hidden = self.encoder(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                matrix = pooled.detach().cpu().numpy().astype(self.np.float32, copy=False)
                self.encoder_seconds += time.monotonic() - stage_started
                expected_dim = int(self.config["embedding_size"])
                if matrix.shape != (len(batch), expected_dim):
                    raise DetectorError("Encoder returned an incompatible feature matrix.")
                if not bool(self.np.isfinite(matrix).all()):
                    raise DetectorError("Encoder returned non-finite features.")
                stage_started = time.monotonic()
                batch_predictions = self.classifier.predict(matrix)
                self.classifier_seconds += time.monotonic() - stage_started
                if len(batch_predictions) != len(batch):
                    raise DetectorError("Classifier returned an incompatible result.")
                if any(isinstance(value, (bool, self.np.bool_))
                       or not isinstance(value, Integral) or value not in (0, 1)
                       for value in batch_predictions):
                    raise DetectorError("Classifier returned an invalid binary label.")
                predictions.extend(int(value) for value in batch_predictions)
        except DetectorError:
            raise
        except Exception:
            # Library exceptions can contain input fragments; do not relay them.
            raise DetectorError("Inference failed; the staged scan is incomplete.") from None
        finally:
            self.inference_seconds += time.monotonic() - started
        return predictions


def load_detector(config: dict | None = None) -> FrozenDetector:
    """Load only local, pinned assets and verify them against the run record."""
    config = config or read_config()
    try:
        require_memory(config)
        explicit_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
        if explicit_cache:
            snapshot = (Path(explicit_cache) / "models--microsoft--codebert-base" /
                        "snapshots" / config["model_revision"])
            if not snapshot.is_dir():
                raise DetectorError("Configured CodeBERT cache lacks the pinned snapshot.", code="cache_missing")
        import joblib
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        record_path = PROJECT_ROOT / config["experiment_record"]
        classifier_path = PROJECT_ROOT / config["classifier_path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        run_config_path = record_path.with_name("run_config.json")
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        expected_rf = record["selected_random_forest"]
        if record["embedding_model"] != config["model"]:
            raise DetectorError("Hook encoder does not match the completed experiment.")
        if record["embedding_model_revision"] != config["model_revision"]:
            raise DetectorError("Hook encoder revision does not match the completed experiment.")
        if (run_config.get("tokenizer_revision") != config["tokenizer_revision"]
                or run_config.get("model_revision") != config["model_revision"]
                or run_config.get("model") != config["model"]):
            raise DetectorError("Hook tokenizer revision does not match the completed experiment.")
        if record["max_length_tokens"] != config["max_length"] or record["pooling"] != config["pooling"]:
            raise DetectorError("Hook feature settings do not match the completed experiment.")
        if record["hidden_size"] != config["embedding_size"] or record["label_mapping"].get("1") != "secret":
            raise DetectorError("Hook feature or label mapping does not match the completed experiment.")
        if not classifier_path.is_file() or _sha256(classifier_path) != config["classifier_sha256"]:
            raise DetectorError("Frozen classifier is missing or its checksum does not match.")

        pinned_revision = "3b0952feddeffad0063f274080e3c23d75e7eb39"
        if (config["model"] != "microsoft/codebert-base"
                or config["model_revision"] != pinned_revision
                or config["tokenizer_revision"] != pinned_revision
                or config["embedding_size"] != 768 or config["max_length"] != 512
                or config["positive_label"] != 1
                or record["label_mapping"] != {"0": "non-secret", "1": "secret"}):
            raise DetectorError("Inference settings violate the frozen model contract.")

        classifier = joblib.load(classifier_path)
        if classifier.n_features_in_ != int(config["embedding_size"]):
            raise DetectorError("Frozen classifier expects an incompatible feature dimension.")
        if set(int(value) for value in classifier.classes_) != {0, 1}:
            raise DetectorError("Frozen classifier does not contain the expected binary classes.")
        actual = classifier.get_params(deep=False)
        settings = ("n_estimators", "max_depth", "max_features", "min_samples_leaf",
                    "class_weight", "random_state", "n_jobs")
        if {key: actual[key] for key in settings} != expected_rf:
            raise DetectorError("Frozen classifier settings do not match the experiment record.")

        torch.set_num_threads(int(config["torch_threads"]))
        # A transferred checkout already contains this prepared offline cache.
        # Explicit environment configuration retains precedence.
        prepared_cache = PROJECT_ROOT / "model_assets" / "huggingface" / "hub"
        cache_options = {}
        if not os.environ.get("HF_HUB_CACHE") and not os.environ.get("TRANSFORMERS_CACHE") and prepared_cache.is_dir():
            cache_options["cache_dir"] = str(prepared_cache)
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["tokenizer_revision"], local_files_only=True, **cache_options
        )
        encoder = AutoModel.from_pretrained(
            config["model"], revision=config["model_revision"], local_files_only=True, **cache_options
        ).to(torch.device("cpu"))
        require_memory(config, during_inference=True)
        if int(encoder.config.hidden_size) != int(config["embedding_size"]):
            raise DetectorError("Cached encoder has an incompatible hidden size.")
        return FrozenDetector(config, classifier, tokenizer, encoder, np, torch)
    except DetectorError:
        raise
    except Exception:
        # Includes missing offline checkpoint/dependencies; never emit library details.
        raise DetectorError(
            "Could not load the frozen model and local CodeBERT checkpoint. "
            "No network access was attempted; verify the project environment and cache.",
            code="model_load",
        ) from None
