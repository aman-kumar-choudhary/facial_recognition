"""
Vector store abstraction for embedding storage + nearest-neighbor search.

Prototype backend: FAISS (IndexFlatIP over L2-normalized vectors == cosine
similarity), matching the project's "FAISS (prototype)" recommendation.
The interface is intentionally narrow (add / search / remove / persist) so
swapping in Milvus/Qdrant/ChromaDB later only requires a new class behind
the same VectorStore interface -- no changes to calling code.
"""
import json
import logging
import os
import threading
import time
from typing import Optional

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# The optional models use independent indexes.  This keeps both vector shape
# and score distribution isolated from ArcFace without asking operators to
# edit EMBEDDING_DIM every time they run a benchmark.
_MODEL_EMBEDDING_DIMS = {
    "ArcFace": 512,
    "FaceNet": 512,
    "VGGFace2": 512,
    "DeepFace": 4096,
}


class VectorStore:
    def add(self, student_id: str, embedding: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        raise NotImplementedError

    def remove(self, student_id: str) -> None:
        raise NotImplementedError

    def replace_embeddings(self, student_id: str, embeddings: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def migrate_student_ids(self, migrations: dict[str, str]) -> None:
        raise NotImplementedError


class FaissVectorStore(VectorStore):
    """FAISS IndexIDMap2(IndexFlatIP) with a side JSON file mapping
    integer FAISS ids <-> student-image-variant entries (FAISS ids must be
    int64). Each registered student has vectors for the configured image
    storage representation(s)."""

    def __init__(self, namespace: str = "ArcFace"):
        self._lock = threading.RLock()
        self._dim = _MODEL_EMBEDDING_DIMS.get(namespace, settings.EMBEDDING_DIM)
        # Embeddings from different recognition models must never share an
        # index: their dimensionality and distance distributions differ.
        suffix = "" if namespace == "ArcFace" else f".{namespace.lower()}"
        self._index_path = f"{settings.VECTOR_INDEX_PATH}{suffix}"
        self._meta_path = f"{settings.VECTOR_META_PATH}{suffix}"
        os.makedirs(os.path.dirname(self._index_path) or ".", exist_ok=True)

        base_index = faiss.IndexFlatIP(self._dim)
        self._index = faiss.IndexIDMap2(base_index)
        self._id_to_entry: dict[int, dict[str, str]] = {}
        self._student_to_ids: dict[str, set[int]] = {}
        self._next_id = 0

        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._load()

    def _load(self) -> None:
        self._index = faiss.read_index(self._index_path)
        with open(self._meta_path, "r") as f:
            meta = json.load(f)
        # Migrate the old one-vector-per-student metadata format in place.
        raw_entries = meta.get("id_to_entry")
        if raw_entries is None:
            raw_entries = {
                key: {"student_id": value, "variant": "color"}
                for key, value in meta.get("id_to_student", {}).items()
            }
        self._id_to_entry = {int(key): value for key, value in raw_entries.items()}
        self._student_to_ids = {}
        for faiss_id, entry in self._id_to_entry.items():
            self._student_to_ids.setdefault(entry["student_id"], set()).add(faiss_id)
        self._next_id = meta["next_id"]
        logger.info(
            "vector_index_loaded",
            extra={"event": "vector_index_loaded", "vector_count": self._index.ntotal},
        )

    def _persist(self) -> None:
        index_tmp = f"{self._index_path}.tmp"
        meta_tmp = f"{self._meta_path}.tmp"
        faiss.write_index(self._index, index_tmp)
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "id_to_entry": {str(key): value for key, value in self._id_to_entry.items()},
                    "next_id": self._next_id,
                },
                f,
            )
        os.replace(index_tmp, self._index_path)
        os.replace(meta_tmp, self._meta_path)

    def add(self, student_id: str, embedding: np.ndarray) -> None:
        self.replace_embeddings(student_id, {"color": embedding})

    def replace_embeddings(self, student_id: str, embeddings: dict[str, np.ndarray]) -> None:
        if not embeddings:
            raise ValueError("At least one embedding is required")
        started = time.perf_counter()
        with self._lock:
            self._remove_locked(student_id)
            vectors, ids = [], []
            for variant, embedding in embeddings.items():
                vec = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, -1)
                if vec.shape[1] != self._dim:
                    raise ValueError(f"Embedding has dimension {vec.shape[1]}; expected {self._dim}")
                faiss_id = self._next_id
                self._next_id += 1
                vectors.append(vec)
                ids.append(faiss_id)
                self._id_to_entry[faiss_id] = {"student_id": student_id, "variant": variant}
                self._student_to_ids.setdefault(student_id, set()).add(faiss_id)
            self._index.add_with_ids(np.vstack(vectors), np.asarray(ids, dtype=np.int64))
            self._persist()
        logger.info(
            "vector_embeddings_stored",
            extra={
                "event": "vector_embeddings_stored",
                "student_id": student_id,
                "variants": sorted(embeddings),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )

    def search(self, embedding: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            vec = embedding.astype(np.float32).reshape(1, -1)
            # Each student has one or two configured variants for five poses.
            # Fetch all candidates before collapsing to
            # one best cosine score per student; otherwise one student's
            # pose gallery can crowd other students out of a short FAISS list.
            # IndexFlatIP already evaluates the complete index per query.
            search_count = self._index.ntotal
            scores, ids = self._index.search(vec, search_count)
            best_by_student: dict[str, float] = {}
            for score, idx in zip(scores[0], ids[0]):
                if idx == -1:
                    continue
                entry = self._id_to_entry.get(int(idx))
                if entry:
                    student_id = entry["student_id"]
                    best_by_student[student_id] = max(best_by_student.get(student_id, -1.0), float(score))
            results = sorted(best_by_student.items(), key=lambda result: result[1], reverse=True)[:top_k]
        logger.debug(
            "vector_search_completed",
            extra={"event": "vector_search_completed", "top_k": top_k, "match_count": len(results)},
        )
        return results

    def remove(self, student_id: str) -> None:
        with self._lock:
            if self._remove_locked(student_id):
                self._persist()

    def migrate_student_ids(self, migrations: dict[str, str]) -> None:
        """Update FAISS metadata after the database UUID -> roll migration."""
        if not migrations:
            return
        with self._lock:
            changed = False
            for entry in self._id_to_entry.values():
                old_id = entry["student_id"]
                if old_id in migrations:
                    entry["student_id"] = migrations[old_id]
                    changed = True
            if changed:
                self._student_to_ids = {}
                for faiss_id, entry in self._id_to_entry.items():
                    self._student_to_ids.setdefault(entry["student_id"], set()).add(faiss_id)
                self._persist()
                logger.info("vector_student_ids_migrated", extra={"event": "vector_student_ids_migrated", "count": len(migrations)})

    def _remove_locked(self, student_id: str) -> bool:
        faiss_ids = self._student_to_ids.pop(student_id, set())
        if not faiss_ids:
            return False
        self._index.remove_ids(np.asarray(sorted(faiss_ids), dtype=np.int64))
        for faiss_id in faiss_ids:
            self._id_to_entry.pop(faiss_id, None)
        return True


_store_singletons: dict[str, VectorStore] = {}


def get_vector_store(namespace: str = "ArcFace") -> VectorStore:
    if namespace not in _store_singletons:
        if settings.VECTOR_STORE_BACKEND == "faiss":
            _store_singletons[namespace] = FaissVectorStore(namespace)
        else:
            raise NotImplementedError(
                f"Vector backend '{settings.VECTOR_STORE_BACKEND}' not wired up yet. "
                "Add a class implementing VectorStore and register it here."
            )
    return _store_singletons[namespace]
