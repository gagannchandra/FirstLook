"""E9: edge benchmark of the shipped bundles (size, cold start, latency, RAM, network).

Runs on the host or inside Docker:
  docker run --rm --network none --cpus 2 --memory 2g -v $PWD/results/bench:/out firstlook \
      python -m src.benchmark --out /out/bench_docker.json --setting "Docker --network none, 2 CPU, 2 GB"
Inputs are synthetic patients drawn uniformly inside the valid ranges (no dataset rows).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time

T_START = time.perf_counter()

import numpy as np  # noqa: E402
import psutil  # noqa: E402


def nic_bytes():
    return {n: (c.bytes_sent, c.bytes_recv) for n, c in psutil.net_io_counters(pernic=True).items()}


def synthetic_forms(spec, n, seed=0):
    rng = np.random.default_rng(seed)
    cc = spec["complaint_order"][:120]
    bins = ["23-02", "03-06", "07-10", "11-14", "15-18", "19-22"]
    modes = ["Walk-in", "Car", "ambulance", "Wheelchair", "Other"]
    forms = []
    for _ in range(n):
        f = {"age": int(rng.integers(18, 100)), "gender": str(rng.choice(["Female", "Male"])),
             "arrivalmode": str(rng.choice(modes)), "arrivalhour_bin": str(rng.choice(bins)),
             "triage_vital_hr": float(rng.integers(50, 150)), "triage_vital_sbp": float(rng.integers(85, 200)),
             "triage_vital_dbp": float(rng.integers(45, 110)), "triage_vital_rr": float(rng.integers(12, 32)),
             "triage_vital_o2": float(rng.integers(85, 101)), "triage_vital_o2_device": float(rng.integers(0, 2)),
             "triage_vital_temp": float(np.round(rng.uniform(96, 103), 1)),
             "complaints": [str(rng.choice(cc))], "esi": str(rng.integers(1, 6))}
        for v in ("triage_vital_o2", "triage_vital_temp"):
            if rng.random() < 0.2:
                f[v] = None
        forms.append(f)
    return forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench/bench_host.json")
    ap.add_argument("--setting", default="host")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    net0 = nic_bytes()
    t_imp = time.perf_counter()
    from src.runtime import load_all
    bundles = load_all()
    cold = (time.perf_counter() - T_START) * 1000
    res = {"setting": a.setting, "platform": platform.platform(), "cpu_count_visible": os.cpu_count(),
           "cpu_affinity": len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else None,
           "interfaces": sorted(psutil.net_if_addrs().keys()), "cold_start_ms_all_bundles": cold,
           "import_and_load_ms": (time.perf_counter() - t_imp) * 1000, "bundles": {}}
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            res["cgroup_memory_max"] = f.read().strip()
        with open("/sys/fs/cgroup/cpu.max") as f:
            res["cgroup_cpu_max"] = f.read().strip()
    except OSError:
        pass
    for slot, b in bundles.items():
        forms = synthetic_forms(b.spec, a.n)
        for f in forms[:20]:
            b.predict(f)
        # model-only latency (ONNX run on a prepared vector) and end-to-end latency (form -> result incl. explanation)
        X = [b.vectorize(__import__("src.runtime", fromlist=["form_to_raw"]).form_to_raw(f, b.spec)) for f in forms[:200]]
        mt = []
        for i in range(a.n):
            x = X[i % len(X)]
            t0 = time.perf_counter(); b.raw_proba(x); mt.append((time.perf_counter() - t0) * 1000)
        et = []
        for f in forms:
            t0 = time.perf_counter(); b.predict(f); et.append((time.perf_counter() - t0) * 1000)
        res["bundles"][slot] = {
            "version": b.version, "n_features": len(b.features), "model_onnx_kb": b.model_bytes / 1024,
            "bundle_kb": b.size_bytes / 1024,
            "model_p50_ms": float(np.percentile(mt, 50)), "model_p95_ms": float(np.percentile(mt, 95)),
            "e2e_p50_ms": float(np.percentile(et, 50)), "e2e_p95_ms": float(np.percentile(et, 95)),
            "n_predictions": a.n}
    net1 = nic_bytes()
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    res["net_bytes_by_interface"] = {n: {"sent": net1[n][0] - net0.get(n, (0, 0))[0], "recv": net1[n][1] - net0.get(n, (0, 0))[1]}
                                     for n in net1}
    res["net_bytes_sent_non_loopback"] = sum(v["sent"] for n, v in res["net_bytes_by_interface"].items() if not n.startswith("lo"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
