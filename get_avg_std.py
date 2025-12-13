#!/usr/bin/env python3
"""
Read results TSV and compute avg/std over eval1..eval5.

Input TSV columns:
  task_name  mode  node_name  algo  eval1 eval2 eval3 eval4 eval5

Output TSV columns:
  task_name  mode  node_name  algo  avg_of_eval  std_of_eval
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List


IN_TSV = Path("out/results.tsv")
OUT_TSV = Path("out/results_avg_std.tsv")


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main() -> None:
    rows_out = []

    with IN_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vals: List[float] = []
            for k in ("eval1", "eval2", "eval3", "eval4", "eval5"):
                v = row.get(k, "").strip()
                if v:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass

            if not vals:
                avg = ""
                sd = ""
            else:
                avg = f"{mean(vals):.6f}"
                sd = f"{std(vals):.6f}"

            rows_out.append([
                row["task_name"],
                row["mode"],
                row["node_name"],
                row["algo"],
                avg,
                sd,
            ])

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "task_name",
            "mode",
            "node_name",
            "algo",
            "avg_of_eval",
            "std_of_eval",
        ])
        writer.writerows(rows_out)

    print(f"Wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
