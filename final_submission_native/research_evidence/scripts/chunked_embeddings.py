"""Atomic, validated on-disk chunks for CodeBERT embedding extraction."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


class ChunkStore:
    """Store one output file set per input batch; incomplete chunks are ignored."""

    def __init__(self, root, split, config, row_count, hidden_size):
        self.root = Path(root) / "chunks" / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.split, self.config = split, config
        self.row_count, self.hidden_size = row_count, hidden_size
        self.chunk_size = int(config.get("chunk_rows", config["batch_size"]))
        self.manifest_path = self.root / "manifest.json"
        expected = {"split": split, "config": config, "row_count": row_count,
                    "hidden_size": hidden_size, "chunk_size": self.chunk_size}
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text())
            if current != expected:
                raise ValueError(f"Chunk configuration mismatch for {split}")
        else:
            atomic_json(self.manifest_path, expected)

    def chunk_paths(self, index):
        stem = f"chunk_{index:08d}"
        return tuple(self.root / f"{stem}.{suffix}" for suffix in ("features.npy", "labels.npy", "ids.json", "complete.json"))

    def valid(self, index, start, end, np):
        features_path, labels_path, ids_path, done_path = self.chunk_paths(index)
        if not all(p.is_file() for p in (features_path, labels_path, ids_path, done_path)):
            return False
        try:
            done = json.loads(done_path.read_text())
            features = np.load(features_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
            ids = json.loads(ids_path.read_text())
            labels_ok = bool(np.isin(labels, (0, 1)).all())
            return (done == {"start": start, "end": end,
                             "features_sha256": sha256_file(features_path),
                             "labels_sha256": sha256_file(labels_path),
                             "ids_sha256": sha256_file(ids_path)}
                    and features.shape == (end-start, self.hidden_size)
                    and features.dtype == np.float32
                    and labels.shape == (end-start,) and labels.dtype == np.int64
                    and labels_ok and len(ids) == end-start and len(set(ids)) == len(ids))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def write(self, index, start, features, labels, row_ids, np):
        end = start + len(labels)
        if len(features) != len(labels) or len(row_ids) != len(labels):
            raise ValueError("Chunk arrays and row IDs have different lengths")
        final = self.chunk_paths(index)
        temps = []
        try:
            for destination, writer in zip(final[:3], (
                lambda f: np.save(f, np.asarray(features, dtype=np.float32), allow_pickle=False),
                lambda f: np.save(f, np.asarray(labels, dtype=np.int64), allow_pickle=False),
                lambda f: f.write(json.dumps(row_ids, separators=(",", ":")).encode()),
            )):
                fd, name = tempfile.mkstemp(prefix=".chunk-", dir=self.root)
                os.close(fd); temp = Path(name); temps.append(temp)
                with temp.open("wb") as f: writer(f)
                with temp.open("rb") as f: os.fsync(f.fileno())
            for temp, dest in zip(temps, final[:3]): os.replace(temp, dest)
            atomic_json(final[3], {"start": start, "end": end,
                "features_sha256": sha256_file(final[0]), "labels_sha256": sha256_file(final[1]),
                "ids_sha256": sha256_file(final[2])})
        finally:
            for temp in temps:
                if temp.exists(): temp.unlink()

    def validate_complete(self, np):
        expected_chunks = (self.row_count + self.chunk_size - 1) // self.chunk_size
        seen_ids, total = set(), 0
        for index in range(expected_chunks):
            start = index * self.chunk_size
            end = min(start + self.chunk_size, self.row_count)
            if not self.valid(index, start, end, np):
                raise ValueError(f"Missing, invalid, or incompatible chunk {index} for {self.split}")
            ids = json.loads(self.chunk_paths(index)[2].read_text())
            if seen_ids.intersection(ids): raise ValueError(f"Duplicate row IDs in {self.split}")
            seen_ids.update(ids); total += len(ids)
        if total != self.row_count: raise ValueError(f"Chunk row total mismatch for {self.split}")
        extras = list(self.root.glob("chunk_*.complete.json"))
        if len(extras) != expected_chunks: raise ValueError(f"Out-of-range chunks in {self.split}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)
