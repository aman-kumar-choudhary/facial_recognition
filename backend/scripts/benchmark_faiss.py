#!/usr/bin/env python3
"""Measure FAISS on this machine and write reproducible CSV/JSON results.

Example: python scripts/benchmark_faiss.py --sizes 10000,100000 --queries 500
Use the default sizes only after confirming available RAM/VRAM; it allocates
the selected corpus in memory and never writes it into the application index.
"""
import argparse, csv, json, os, subprocess, time
from pathlib import Path

import faiss
import numpy as np
import psutil


def percentile(values, p): return float(np.percentile(values, p)) if values else None
def gpu_memory():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=2)
        return float(out.splitlines()[0])
    except Exception: return None

def run_index(name, index, corpus, queries, k, device="cpu"):
    process = psutil.Process(); before_ram, before_gpu = process.memory_info().rss, gpu_memory()
    started = time.perf_counter(); index.add(corpus); build_ms = (time.perf_counter()-started)*1000
    # Warmup avoids mixing allocator/context creation into search statistics.
    index.search(queries[:min(20, len(queries))], k)
    latencies = []
    started = time.perf_counter()
    for query in queries:
        q_start = time.perf_counter(); index.search(query.reshape(1, -1), k); latencies.append((time.perf_counter()-q_start)*1000)
    elapsed = time.perf_counter()-started
    return {"index": name, "device": device, "vectors": len(corpus), "dimension": corpus.shape[1], "build_ms": round(build_ms, 2),
        "ram_delta_mb": round((process.memory_info().rss-before_ram)/1048576, 2), "vram_delta_mb": None if before_gpu is None or gpu_memory() is None else round(gpu_memory()-before_gpu, 2),
        "avg_search_ms": round(float(np.mean(latencies)), 4), "p50_ms": round(percentile(latencies, 50),4), "p95_ms": round(percentile(latencies,95),4), "p99_ms": round(percentile(latencies,99),4),
        "qps": round(len(queries)/elapsed,2), "top_k": k}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--sizes", default="10000,25000,50000,100000,250000,500000,1000000")
    parser.add_argument("--queries", type=int, default=300); parser.add_argument("--dim", type=int, default=512); parser.add_argument("--output", default="./data/benchmarks")
    parser.add_argument("--architectures", default="flat,ivf,hnsw,ivfpq"); args = parser.parse_args()
    rng = np.random.default_rng(42); results=[]; has_gpu=False
    try: has_gpu = faiss.get_num_gpus() > 0
    except Exception: pass
    for size in map(int, args.sizes.split(",")):
        corpus = rng.standard_normal((size,args.dim), dtype=np.float32); faiss.normalize_L2(corpus)
        queries = rng.standard_normal((args.queries,args.dim), dtype=np.float32); faiss.normalize_L2(queries)
        for kind in args.architectures.split(","):
            if kind == "flat": factory = lambda: faiss.IndexFlatIP(args.dim)
            elif kind == "ivf":
                nlist=max(32, min(int(size**.5), 4096)); factory=lambda: faiss.IndexIVFFlat(faiss.IndexFlatIP(args.dim),args.dim,nlist,faiss.METRIC_INNER_PRODUCT)
            elif kind == "hnsw": factory=lambda: faiss.IndexHNSWFlat(args.dim,32,faiss.METRIC_INNER_PRODUCT)
            elif kind == "ivfpq":
                nlist=max(32,min(int(size**.5),4096)); factory=lambda: faiss.IndexIVFPQ(faiss.IndexFlatIP(args.dim),args.dim,nlist,32,8,faiss.METRIC_INNER_PRODUCT)
            else: continue
            index=factory()
            if not index.is_trained: index.train(corpus)
            results.append(run_index(kind,index,corpus,queries,5))
            if kind in {"flat", "ivf"} and has_gpu:
                cpu=factory()
                if not cpu.is_trained: cpu.train(corpus)
                resources=faiss.StandardGpuResources(); gpu=faiss.index_cpu_to_gpu(resources,0,cpu)
                results.append(run_index(kind,gpu,corpus,queries,5,"gpu"))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime("%Y%m%d-%H%M%S"); json_path=output/f"faiss-{stamp}.json"; csv_path=output/f"faiss-{stamp}.csv"
    json_path.write_text(json.dumps(results,indent=2));
    with csv_path.open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=results[0].keys()); writer.writeheader(); writer.writerows(results)
    print(json.dumps({"json":str(json_path),"csv":str(csv_path),"rows":len(results)},indent=2))
if __name__ == "__main__": main()
