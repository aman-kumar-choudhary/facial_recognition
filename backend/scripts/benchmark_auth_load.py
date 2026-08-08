#!/usr/bin/env python3
"""End-to-end authentication/load benchmark against a running API.

The output uses the API's real detector/liveness/recognition timings; it does
not estimate model resource usage. Example:
python scripts/benchmark_auth_load.py --image ./sample.jpg --concurrency 1,5,10,20
"""
import argparse, asyncio, base64, csv, json, statistics, time
from pathlib import Path
from urllib import request

def pct(values, value): return round(sorted(values)[min(len(values)-1, int((len(values)-1)*value))], 2)
def call(url, payload):
    req=request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    started=time.perf_counter()
    try:
        with request.urlopen(req, timeout=30) as response: body=json.loads(response.read()); return (time.perf_counter()-started)*1000, response.status, body
    except Exception as exc: return (time.perf_counter()-started)*1000, 0, {"error":str(exc)}
async def run_group(url,payload,concurrency,requests):
    started=time.perf_counter(); rows=await asyncio.gather(*[asyncio.to_thread(call,url,payload) for _ in range(requests)])
    latencies=[r[0] for r in rows]; good=[r for r in rows if r[1]==200]; elapsed=time.perf_counter()-started
    stages={}
    for _,_,body in good:
        for key,value in body.get("step_latencies_ms",{}).items(): stages.setdefault(key,[]).append(value)
    return {"concurrency":concurrency,"requests":requests,"failed_requests":requests-len(good),"avg_latency_ms":round(statistics.mean(latencies),2),"p95_latency_ms":pct(latencies,.95),"p99_latency_ms":pct(latencies,.99),"requests_per_second":round(requests/elapsed,2),"stages_avg_ms":{k:round(statistics.mean(v),2) for k,v in stages.items()}}
async def main():
    p=argparse.ArgumentParser();p.add_argument("--url",default="http://localhost:8000/api/v1/auth/authenticate");p.add_argument("--image",required=True);p.add_argument("--concurrency",default="1,5,10,20,30,50,100");p.add_argument("--requests-per-level",type=int,default=50);p.add_argument("--output",default="./data/benchmarks");args=p.parse_args()
    encoded=base64.b64encode(Path(args.image).read_bytes()).decode(); payload={"image_base64":encoded}; results=[]
    for level in map(int,args.concurrency.split(",")):
        # Each level dispatches that many simultaneous requests, repeated until
        # the requested sample count is reached.
        batches=max(1,(args.requests_per_level+level-1)//level); samples=[]
        for _ in range(batches): samples.append(await run_group(args.url,payload,level,level))
        # Combine statistically by issuing one representative line per level.
        results.append({"concurrency":level,"requests":sum(x["requests"] for x in samples),"failed_requests":sum(x["failed_requests"] for x in samples),"avg_latency_ms":round(statistics.mean(x["avg_latency_ms"] for x in samples),2),"p95_latency_ms":max(x["p95_latency_ms"] for x in samples),"p99_latency_ms":max(x["p99_latency_ms"] for x in samples),"requests_per_second":round(statistics.mean(x["requests_per_second"] for x in samples),2),"stages_avg_ms":samples[-1]["stages_avg_ms"]})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);stamp=time.strftime("%Y%m%d-%H%M%S");(out/f"auth-load-{stamp}.json").write_text(json.dumps(results,indent=2))
    with (out/f"auth-load-{stamp}.csv").open("w",newline="") as f: writer=csv.DictWriter(f,fieldnames=["concurrency","requests","failed_requests","avg_latency_ms","p95_latency_ms","p99_latency_ms","requests_per_second"]);writer.writeheader();writer.writerows([{k:v for k,v in row.items() if k!="stages_avg_ms"} for row in results])
    print(json.dumps(results,indent=2))
if __name__=="__main__": asyncio.run(main())
