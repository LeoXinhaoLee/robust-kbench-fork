import os
import json

ROOT_DIR = "TTT/mnist_linear_relu/backward"
OUT_TSV = "raw_times.tsv"
NUM_EVALS = 5

# step -> {eval_id: value}
data = {}

for name in os.listdir(ROOT_DIR):
    step_dir = os.path.join(ROOT_DIR, name)
    if not (name.startswith("step_") and os.path.isdir(step_dir)):
        continue

    step_num = name.replace("step_", "")
    data[step_num] = {}

    for i in range(1, NUM_EVALS + 1):
        json_path = os.path.join(
            step_dir,
            f"eval_results_{i}",
            "time_results.json",
        )
        if not os.path.isfile(json_path):
            continue

        with open(json_path, "r") as f:
            j = json.load(f)

        data[step_num][i] = j["summary"]["avg_mean_time"]

# Sort steps numerically
steps_sorted = sorted(data.keys(), key=int)

with open(OUT_TSV, "w") as f:
    # Header
    header = ["step"] + [f"eval_{i}" for i in range(1, NUM_EVALS + 1)]
    f.write("\t".join(header) + "\n")

    # Rows
    for step in steps_sorted:
        row = [f"step_{step}"]
        for i in range(1, NUM_EVALS + 1):
            val = data[step].get(i, "")
            row.append(str(val))
        f.write("\t".join(row) + "\n")

print(f"Wrote TSV to {OUT_TSV}")
