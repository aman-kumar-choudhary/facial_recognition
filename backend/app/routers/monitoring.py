"""Operational stats; deliberately lightweight and safe when NVIDIA tools are absent."""
import collections
import subprocess
import time

import psutil
from fastapi import APIRouter, Request

from app.vector_store import get_vector_store

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


def gpu_stats() -> dict[str, object]:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=1).strip()
        name, util, used, total = [part.strip() for part in output.splitlines()[0].split(",")]
        return {"available": True, "name": name, "utilization_percent": float(util), "memory_used_mb": float(used), "memory_total_mb": float(total)}
    except Exception:
        return {"available": False}


@router.get("/stats")
async def stats(request: Request):
    metrics = getattr(request.app.state, "metrics", {"active_requests": 0, "latencies": collections.deque(maxlen=500)})
    samples = sorted(metrics["latencies"])
    def percentile(values, fraction):
        if not values:
            return None
        index = (len(values) - 1) * fraction
        lower, upper = int(index), min(int(index) + 1, len(values) - 1)
        return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 2)
    process = psutil.Process()
    faiss_samples = sorted(metrics.get("faiss_latencies", ()))
    return {
        "timestamp": time.time(), "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
        "ram_total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 1),
        "process_rss_mb": round(process.memory_info().rss / 1024 / 1024, 1),
        "gpu": gpu_stats(), "faiss": get_vector_store().status(),
        "models": request.app.state.model_manager.performance_status(),
        "performance_summary": {
            "model_inference": request.app.state.model_manager.performance_summary(),
            "faiss_search_latency_ms": {"average": round(sum(faiss_samples) / len(faiss_samples), 2) if faiss_samples else None, "p95": percentile(faiss_samples, .95), "sample_count": len(faiss_samples)},
            "end_to_end_authentication_latency_ms": {"average": round(sum(samples) / len(samples), 2) if samples else None, "p95": percentile(samples, .95), "sample_count": len(samples)},
        },
        "active_recognition_requests": metrics["active_requests"],
        "recognition_latency_ms": {"average": round(sum(samples) / len(samples), 2) if samples else None, "p95": percentile(samples, .95), "p99": percentile(samples, .99), "sample_count": len(samples)},
    }
