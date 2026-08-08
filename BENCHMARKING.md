# Benchmarking and scale decisions

Run the API on the target host, then collect real measurements:

```bash
cd backend
python scripts/benchmark_faiss.py --sizes 10000,25000,50000,100000 --queries 500
python scripts/benchmark_auth_load.py --image ./sample.jpg
```

Both tools write timestamped CSV and JSON to `data/benchmarks`. Increase the
FAISS sizes one step at a time; the tool intentionally does not claim results
for hardware it has not run on.

`IndexFlatIP` is exact brute-force cosine search for normalized vectors. Its
compute grows linearly with the *vector* count, not student count: five poses
for 100,000 students means 500,000 vectors. Raw vectors need approximately
`count × 512 × 4` bytes (about 1.9 GiB at one million), plus index and runtime
overhead. The new store searches a configurable top-64 vector candidate set,
then groups candidates by student. Raise `FAISS_CANDIDATE_COUNT` if five or
more variants routinely occupy the top results; validate recall with your own
genuine/impostor set before changing thresholds.

Start simple: ArcFace + GPU inference + GPU `IndexFlatIP` when the benchmark
shows it meets your latency target. The persisted CPU index remains required
for recovery. At 100K–500K vectors, brute force is often still viable on a
modern GPU; actual model inference and liveness may dominate end-to-end time.
Move to IVF (GPU-compatible) only when measured flat latency or concurrent QPS
fails the target. HNSW is CPU-oriented and has expensive updates; IVFPQ trades
recall for memory and is a later option for multi-million-vector pressure.
Never select ANN solely from vector count—record recall, FAR/FRR and top-k
accuracy on representative poses, lighting, eyewear, and impostor captures.
