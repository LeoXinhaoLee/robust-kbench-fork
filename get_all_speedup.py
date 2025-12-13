#!/usr/bin/env python3
"""
Compute speedups from the per-run TSV.

Input TSV columns:
  task_name  mode  node_name  algo  eval1 eval2 eval3 eval4 eval5
Where algo in {native, compile, highlight}

Output TSV columns:
  task_name  mode  node_name
  avg_speedup_over_native  std_speedup_over_native
  avg_speedup_over_compile std_speedup_over_compile

Speedup is computed per-eval as:
  speedup_over_native[i]  = native_time[i] / highlight_time[i]
  speedup_over_compile[i] = compile_time[i] / highlight_time[i]

Only eval indices where both numerator and denominator exist are used.
Std is population std (divide by N). If <2 samples, std=0.0.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IN_TSV = Path("out/results.tsv")          # your out.tsv
OUT_TSV = Path("out/speedups.tsv")


Key = Tuple[str, str, str]  # (task_name, mode, node_name)


def _parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except ValueError:
        return None


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))  # population std


def fmt(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


def main() -> None:
    # Map: (task, mode, node) -> algo -> [eval1..eval5] as Optional[float]
    data: Dict[Key, Dict[str, List[Optional[float]]]] = {}

    with IN_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key: Key = (row["task_name"], row["mode"], row["node_name"])
            algo = row["algo"].strip().lower()
            evals = [_parse_float(row.get(f"eval{i}", "")) for i in range(1, 6)]
            data.setdefault(key, {})[algo] = evals

    out_rows: List[List[str]] = []
    for (task, mode, node), by_algo in sorted(data.items()):
        native = by_algo.get("native")
        compile_ = by_algo.get("compile")
        highlight = by_algo.get("highlight")

        # compute per-eval speedups where both exist
        su_native: List[float] = []
        su_compile: List[float] = []

        if highlight is not None and native is not None:
            for a, b in zip(native, highlight):  # a=native_time, b=highlight_time
                if a is None or b is None or b == 0:
                    continue
                su_native.append(a / b)

        if highlight is not None and compile_ is not None:
            for a, b in zip(compile_, highlight):  # a=compile_time, b=highlight_time
                if a is None or b is None or b == 0:
                    continue
                su_compile.append(a / b)

        avg_su_native = mean(su_native) if su_native else None
        std_su_native = std(su_native) if su_native else None
        avg_su_compile = mean(su_compile) if su_compile else None
        std_su_compile = std(su_compile) if su_compile else None

        out_rows.append([
            task, mode, node,
            fmt(avg_su_native), fmt(std_su_native),
            fmt(avg_su_compile), fmt(std_su_compile),
        ])

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "task_name",
            "mode",
            "node_name",
            "avg_speedup_over_native",
            "std_speedup_over_native",
            "avg_speedup_over_compile",
            "std_speedup_over_compile",
        ])
        w.writerows(out_rows)

    print(f"Wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
