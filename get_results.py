import os
import pathlib
import pandas as pd

def get_csv(file_list):
    for file in file_list:
        if file.endswith("metrics.csv"):
            return file
    return ""

path = "results/run1/"
dirs = os.listdir(path)

# print(f"{'folder':>40} | result")
# print("----------------------------------------------------")
results = []
for dir in dirs:
    d = os.path.join(path, dir)
    if os.path.isdir(d):
        contents = os.listdir(d)
        csv_file = get_csv(contents)
        data = pd.read_csv(os.path.join(d, csv_file))

        last_te_bce_error = data['te_bce_error'].iloc[-1]
        
        results.append((dir, last_te_bce_error))
        print(f"{dir:>40} | {last_te_bce_error}")

df = pd.DataFrame(results)
df.reindex(["run", "last_te_bce_error"])
print(df.sort_values(by=1))

