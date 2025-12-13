from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple


Triplet = Tuple[str, str, str]  # (node_name, task_name, mode)

# -------- configure here --------
TRIPLETS: List[Triplet] = [
    # (“nodeA”, “my_task”, “train”),
    # (“nodeB”, “my_task”, “eval”),
    ("pool0-01438", "mnist_linear_relu", "backward"),
    ("pool0-01438", "mnist_cross_entropy", "forward"),
    ("pool0-01438", "resnet_block", "forward"),
    ##
    ("pool0-02992", "mnist_linear_relu", "backward"),
    ("pool0-02992", "mnist_cross_entropy", "forward"),
    ("pool0-02992", "resnet_block", "forward"),
    
]
OUT_TSV = Path("out/results.tsv")
# --------------------------------


def read_avg_mean_time(path: Path) -> Optional[float]:
    try:
        data: Any = json.loads(path.read_text())
        v = data.get("summary", {}).get("avg_mean_time")
        if v is None:
            return None
        return float(v)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return None


def fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return f"{v:.6f}"


def collect_row(
    node_name: str,
    task_name: str,
    mode: str,
    algo: str,
    pattern: str,
) -> List[str]:
    vals: List[Optional[float]] = []
    for i in range(1, 6):
        p = Path(pattern.format(
            task_name=task_name,
            mode=mode,
            node_name=node_name,
            i=i,
        ))
        vals.append(read_avg_mean_time(p))

    return (
        [task_name, mode, node_name, algo]
        + [fmt(v) for v in vals]
    )


def main() -> None:
    if not TRIPLETS:
        raise SystemExit("TRIPLETS is empty — add (node_name, task_name, mode).")

    header = [
        "task_name",
        "mode",
        "node_name",
        "algo",
        "eval1",
        "eval2",
        "eval3",
        "eval4",
        "eval5",
    ]

    rows: List[List[str]] = []

    for node_name, task_name, mode in TRIPLETS:
        rows.append(collect_row(
            node_name,
            task_name,
            mode,
            "highlight",
            "./highlighted/{task_name}/{mode}/{node_name}/eval_results_{i}/time_results.json",
        ))
        rows.append(collect_row(
            node_name,
            task_name,
            mode,
            "native",
            "tasks/{task_name}/{node_name}/eval_results_{i}/{mode}/torch_native_results.json",
        ))
        rows.append(collect_row(
            node_name,
            task_name,
            mode,
            "compile",
            "tasks/{task_name}/{node_name}/eval_results_{i}/{mode}/torch_compile_results.json",
        ))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {OUT_TSV}")


if __name__ == "__main__":
    main()
