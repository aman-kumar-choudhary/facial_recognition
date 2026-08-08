"""Durable FAISS vector storage with an optional GPU search replica.

The CPU ``IndexIDMap2(IndexFlatIP)`` is the authoritative, persisted index.
When FAISS was built with GPU support, a clone is created once at startup and
after mutations; requests search that clone directly.  This makes restarts and
recovery deterministic while avoiding a CPU index rebuild per embedding.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_MODEL_EMBEDDING_DIMS = {"ArcFace": 512, "FaceNet": 512, "VGGFace2": 512, "DeepFace": 4096}


class VectorStore:
    def add(self, student_id: str, embedding: np.ndarray) -> None: raise NotImplementedError
    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]: raise NotImplementedError
    def remove(self, student_id: str) -> None: raise NotImplementedError
    def replace_embeddings(self, student_id: str, embeddings: dict[str, np.ndarray]) -> None: raise NotImplementedError
    def migrate_student_ids(self, migrations: dict[str, str]) -> None: raise NotImplementedError


class FaissVectorStore(VectorStore):
    """Thread-safe FAISS store. GPU is an acceleration cache, never data loss risk."""

    def __init__(self, namespace: str = "ArcFace"):
        self._lock = threading.RLock()
        self._dim = _MODEL_EMBEDDING_DIMS.get(namespace, settings.EMBEDDING_DIM)
        suffix = "" if namespace == "ArcFace" else f".{namespace.lower()}"
        self._index_path = f"{settings.VECTOR_INDEX_PATH}{suffix}"
        self._meta_path = f"{settings.VECTOR_META_PATH}{suffix}"
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        self._cpu_index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))
        self._search_index = self._cpu_index
        self._gpu_resources = None
        self._backend = "cpu"
        self._gpu_error: str | None = None
        self._id_to_entry: dict[int, dict[str, str]] = {}
        self._student_to_ids: dict[str, set[int]] = {}
        self._next_id = 0
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._load()
        elif os.path.exists(self._index_path):
            # Never pair an unknown index with guessed metadata.
            raise RuntimeError(f"FAISS index exists but metadata is missing: {self._meta_path}")
        self._refresh_search_replica()
        logger.info("faiss_initialized", extra={"event": "faiss_initialized", **self.status()})

    def _load(self) -> None:
        try:
            index = faiss.read_index(self._index_path)
            with open(self._meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            raise RuntimeError("Could not load FAISS index/metadata; restore a matched backup pair") from exc
        if index.d != self._dim:
            raise RuntimeError(f"FAISS dimension {index.d} does not match model dimension {self._dim}")
        raw_entries = meta.get("id_to_entry")
        if raw_entries is None:
            raw_entries = {key: {"student_id": value, "variant": "color"} for key, value in meta.get("id_to_student", {}).items()}
        if index.ntotal != len(raw_entries):
            raise RuntimeError("FAISS index and metadata vector counts differ; restore a matched backup pair")
        self._cpu_index = index
        self._id_to_entry = {int(key): value for key, value in raw_entries.items()}
        self._student_to_ids = {}
        for faiss_id, entry in self._id_to_entry.items():
            self._student_to_ids.setdefault(entry["student_id"], set()).add(faiss_id)
        self._next_id = int(meta.get("next_id", max(self._id_to_entry, default=-1) + 1))

    def _refresh_search_replica(self) -> None:
        """Clone CPU -> GPU once after startup/mutations; fall back safely on any GPU error."""
        self._search_index, self._backend = self._cpu_index, "cpu"
        self._gpu_resources = None
        self._gpu_error = None
        if str(settings.FAISS_DEVICE).lower() == "cpu":
            return
        try:
            gpu_count = int(faiss.get_num_gpus())
            if gpu_count < 1:
                raise RuntimeError("FAISS reports no CUDA devices")
            self._gpu_resources = faiss.StandardGpuResources()
            device = min(max(0, settings.FAISS_GPU_DEVICE), gpu_count - 1)
            self._search_index = faiss.index_cpu_to_gpu(self._gpu_resources, device, self._cpu_index)
            self._backend = "gpu"
        except Exception as exc:  # CPU FAISS wheels do not expose GPU symbols.
            self._gpu_error = str(exc)
            logger.warning("faiss_gpu_unavailable_using_cpu", extra={"event": "faiss_gpu_unavailable_using_cpu", "reason": self._gpu_error})

    def _persist(self) -> None:
        # Atomic replace means a process crash leaves either old or new files.
        index_tmp, meta_tmp = f"{self._index_path}.tmp", f"{self._meta_path}.tmp"
        faiss.write_index(self._cpu_index, index_tmp)
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump({"id_to_entry": {str(k): v for k, v in self._id_to_entry.items()}, "next_id": self._next_id}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(index_tmp, self._index_path)
        os.replace(meta_tmp, self._meta_path)

    def _validated(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(f"Embedding has dimension {vec.shape[1]}; expected {self._dim}")
        if not np.isfinite(vec).all():
            raise ValueError("Embedding contains non-finite values")
        faiss.normalize_L2(vec)
        return vec

    def add(self, student_id: str, embedding: np.ndarray) -> None:
        self.replace_embeddings(student_id, {"color": embedding})

    def replace_embeddings(self, student_id: str, embeddings: dict[str, np.ndarray]) -> None:
        if not embeddings:
            raise ValueError("At least one embedding is required")
        vectors = {variant: self._validated(value) for variant, value in embeddings.items()}
        with self._lock:
            self._remove_locked(student_id)
            ids = np.arange(self._next_id, self._next_id + len(vectors), dtype=np.int64)
            self._next_id += len(vectors)
            self._cpu_index.add_with_ids(np.vstack(list(vectors.values())), ids)
            for faiss_id, variant in zip(ids.tolist(), vectors):
                self._id_to_entry[faiss_id] = {"student_id": student_id, "variant": variant}
                self._student_to_ids.setdefault(student_id, set()).add(faiss_id)
            self._persist()
            self._refresh_search_replica()

    def embeddings_for_student(self, student_id: str) -> dict[str, np.ndarray]:
        """A rollback snapshot for management operations; not for request paths."""
        with self._lock:
            snapshot = {}
            for fid in self._student_to_ids.get(student_id, set()):
                snapshot[self._id_to_entry[fid]["variant"]] = self._cpu_index.reconstruct(fid).copy()
            return snapshot

    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        vec = self._validated(embedding)
        with self._lock:
            if not self._cpu_index.ntotal:
                return []
            # Search candidates instead of copying every score to CPU.  Make this
            # configurable: more variants per person generally need more candidates.
            candidates = min(self._cpu_index.ntotal, max(top_k, settings.FAISS_CANDIDATE_COUNT))
            scores, ids = self._search_index.search(vec, candidates)
            best: dict[str, float] = {}
            for score, fid in zip(scores[0], ids[0]):
                entry = self._id_to_entry.get(int(fid))
                if entry is not None:
                    sid = entry["student_id"]
                    best[sid] = max(best.get(sid, -1.0), float(score))
            return sorted(best.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def remove(self, student_id: str) -> None:
        with self._lock:
            if self._remove_locked(student_id):
                self._persist()
                self._refresh_search_replica()

    def _remove_locked(self, student_id: str) -> bool:
        ids = self._student_to_ids.pop(student_id, set())
        if not ids:
            return False
        self._cpu_index.remove_ids(np.asarray(sorted(ids), dtype=np.int64))
        for fid in ids:
            self._id_to_entry.pop(fid, None)
        return True

    def migrate_student_ids(self, migrations: dict[str, str]) -> None:
        if not migrations:
            return
        with self._lock:
            changed = False
            for entry in self._id_to_entry.values():
                if entry["student_id"] in migrations:
                    entry["student_id"] = migrations[entry["student_id"]]
                    changed = True
            if changed:
                self._student_to_ids = {}
                for fid, entry in self._id_to_entry.items(): self._student_to_ids.setdefault(entry["student_id"], set()).add(fid)
                self._persist()
                self._refresh_search_replica()

    def status(self) -> dict[str, object]:
        size = os.path.getsize(self._index_path) if os.path.exists(self._index_path) else 0
        return {"backend": self._backend, "dimension": self._dim, "vector_count": int(self._cpu_index.ntotal), "student_count": len(self._student_to_ids), "index_size_bytes": size, "gpu_error": self._gpu_error}


_store_singletons: dict[str, VectorStore] = {}

def get_vector_store(namespace: str = "ArcFace") -> VectorStore:
    if namespace not in _store_singletons:
        if settings.VECTOR_STORE_BACKEND != "faiss":
            raise NotImplementedError(f"Vector backend '{settings.VECTOR_STORE_BACKEND}' is not wired up")
        _store_singletons[namespace] = FaissVectorStore(namespace)
    return _store_singletons[namespace]
