from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


Triplet = Tuple[str, str, str]  # (node_name, task_name, mode)

# -------- configure here --------
TRIPLETS: List[Triplet] = [
    ("pool0-01438", "mnist_linear_relu", "backward"),
]

OUT_DIR = Path("out")
OUT_TSV = OUT_DIR / "results.tsv"

N_EVALS = 25  # <-- changed from 5 to 25
# --------------------------------


def read_avg_mean_time(path: Path) -> Optional[float]:
    try:
        data: Any = json.loads(path.read_text())
        v = data.get("summary", {}).get("avg_mean_time")
        if v is None:
            return None
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
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
    dist_accum: Dict[str, List[float]],
) -> List[str]:
    vals: List[Optional[float]] = []
    for i in range(1, N_EVALS + 1):
        p = Path(
            pattern.format(
                task_name=task_name,
                mode=mode,
                node_name=node_name,
                i=i,
            )
        )
        v = read_avg_mean_time(p)
        vals.append(v)
        if v is not None:
            dist_accum[algo].append(v)

    return [task_name, mode, node_name, algo] + [fmt(v) for v in vals]


def save_distribution_plot(values: List[float], title: str, out_path: Path) -> None:
    plt.figure()
    # Histogram distribution
    plt.hist(values, bins="auto")
    plt.title(title)
    plt.xlabel("avg_mean_time")
    plt.ylabel("count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    if not TRIPLETS:
        raise SystemExit("TRIPLETS is empty — add (node_name, task_name, mode).")

    header = (
        ["task_name", "mode", "node_name", "algo"]
        + [f"eval{i}" for i in range(1, N_EVALS + 1)]
    )

    rows: List[List[str]] = []

    # Collect distributions across all triplets/evals, per algo
    dist: Dict[str, List[float]] = {"highlight": [], "native": [], "compile": []}

    for node_name, task_name, mode in TRIPLETS:
        rows.append(
            collect_row(
                node_name,
                task_name,
                mode,
                "highlight",
                "./highlighted/{task_name}/{mode}/{node_name}/eval_results_{i}/time_results.json",
                dist,
            )
        )
        rows.append(
            collect_row(
                node_name,
                task_name,
                mode,
                "native",
                "tasks/{task_name}/{node_name}/eval_results_{i}/{mode}/torch_native_results.json",
                dist,
            )
        )
        rows.append(
            collect_row(
                node_name,
                task_name,
                mode,
                "compile",
                "tasks/{task_name}/{node_name}/eval_results_{i}/{mode}/torch_compile_results.json",
                dist,
            )
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

    # Save three plots
    save_distribution_plot(
        dist["highlight"],
        f"Distribution of avg_mean_time (highlight) across eval1..eval{N_EVALS}",
        OUT_DIR / "dist_highlight.png",
    )
    save_distribution_plot(
        dist["native"],
        f"Distribution of avg_mean_time (native) across eval1..eval{N_EVALS}",
        OUT_DIR / "dist_native.png",
    )
    save_distribution_plot(
        dist["compile"],
        f"Distribution of avg_mean_time (compile) across eval1..eval{N_EVALS}",
        OUT_DIR / "dist_compile.png",
    )

    print(f"Wrote {OUT_TSV}")
    print(f"Wrote {OUT_DIR / 'dist_highlight.png'}")
    print(f"Wrote {OUT_DIR / 'dist_native.png'}")
    print(f"Wrote {OUT_DIR / 'dist_compile.png'}")


if __name__ == "__main__":
    main()
