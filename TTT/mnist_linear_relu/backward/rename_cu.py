import os
import re

pattern = re.compile(r"_train_(\d+)_")

for filename in os.listdir("."):
    match = pattern.search(filename)
    if match and filename.endswith(".cu"):
        step = match.group(1)
        new_name = f"step_{step}.cu"
        os.rename(filename, new_name)
        print(f"{filename} -> {new_name}")
