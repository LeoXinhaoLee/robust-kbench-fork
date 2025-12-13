#!/usr/bin/env python3
"""
Compute per-run times (i=1..5) and their mean/std, then write a TSV:

name    time1   time2   time3   time4   time5   mean    std

- highlighted: ./hightlighted/<middle_path>/eval_results_i/time_results.json
    reads: ["summary"]["avg_mean_time"]
- native/compile: ./tasks/<task_name>/eval_results_i/torch_native_results.json
                  ./tasks/<task_name>/eval_results_i/torch_compile_results.json
    tries to read ["summary"]["avg_mean_time"] first, then a few common fallbacks.

Std is population std (ddof=0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional


def extract_avg_mean_time(data: Any) -> Optional[float]:
    """Best-effort extraction of 'avg_mean_time' from slightly different JSON shapes."""
    try:
        v = data["summary"]["avg_mean_time"]
        return float(v)
    except Exception:
        pass

    # common fallbacks
    for keypath in [
        ("avg_mean_time",),
        ("mean_time",),
        ("avg_time",),
        ("time",),
        ("summary", "mean_time"),
        ("summary", "avg_time"),
        ("summary", "time"),
    ]:
        cur = data
        ok = True
        for k in keypath:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            try:
                return float(cur)
            except Exception:
                pass

    return None


def read_time(json_path: Path) -> float:
    data = json.loads(json_path.read_text())
    v = extract_avg_mean_time(data)
    if v is None:
        raise KeyError(f"Could not find avg_mean_time in {json_path}")
    return v


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)  # population variance
    return m, math.sqrt(var)


def collect_times(paths: list[Path], strict: bool) -> list[float]:
    out: list[float] = []
    for p in paths:
        if p.exists():
            out.append(read_time(p))
        else:
            if strict:
                raise FileNotFoundError(p)
            out.append(float("nan"))
    return out


def nanmean_std(times: list[float]) -> tuple[float, float]:
    clean = [t for t in times if not math.isnan(t)]
    return mean_std(clean)


def fmt(x: float) -> str:
    return "" if math.isnan(x) else f"{x:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    # ap.add_argument("--middle-path", default="mnist_linear_relu/backward/pool0-01438")
    ap.add_argument("--middle-path", default="resnet_block/forward/pool0-02992")
    # ap.add_argument("--middle-path", default="mnist_cross_entropy/forward/pool0-02992")
    ap.add_argument("--out", default="sakana_times.tsv", help="Output TSV path")
    ap.add_argument("--strict", action="store_true", help="Fail if any file is missing")
    args = ap.parse_args()

    middle_path = args.middle_path.strip("/")

    task_name = middle_path.split("/")[0]  # e.g. mnist_linear_relu
    task_mode = middle_path.split("/")[1]
    node = middle_path.split("/")[2]

    highlighted_paths = [
        Path(f"./highlighted/{middle_path}/eval_results_{i}/time_results.json") for i in range(1, 6)
    ]
    native_paths = [
        Path(f"./tasks/{task_name}/{node}/eval_results_{i}/{task_mode}/torch_native_results.json") for i in range(1, 6)
    ]
    compile_paths = [
        Path(f"./tasks/{task_name}/{node}/eval_results_{i}/{task_mode}/torch_compile_results.json") for i in range(1, 6)
    ]

    rows = []
    for label, paths in [
        ("highlight", highlighted_paths),
        ("native", native_paths),
        ("compile", compile_paths),
    ]:
        times = collect_times(paths, strict=args.strict)
        m, s = nanmean_std(times)
        rows.append([f"{middle_path} {label}", *[fmt(t) for t in times], fmt(m), fmt(s)])

    header = ["name", "time1", "time2", "time3", "time4", "time5", "mean", "std"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)

    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
