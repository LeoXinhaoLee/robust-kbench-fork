import csv

# Replace with your CSV file path
filename = "./highlighted/results.csv"


with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(f'Name: {row["Kernel Name"]}, Runtime: {row["Runtime (ms)"]}, Speedup: {row["Speedup"]}, Speedup compile: {row["Speedup Compile"]}')

