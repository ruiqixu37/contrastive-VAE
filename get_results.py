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

df = None

# print(f"{'folder':>40} | result")
# print("----------------------------------------------------")
for dir in dirs:
    d = os.path.join(path, dir)
    if os.path.isdir(d):
        contents = os.listdir(d)
        csv_file = get_csv(contents)
        data = pd.read_csv(os.path.join(d, csv_file))

        if df is None:
            df = pd.DataFrame(columns=['run_name', *data.columns])
            df['run_name'] = df['run_name'].astype('str')
        
        last_stat = data.iloc[-1]
        last_stat['run_name'] = dir
        print(d)
        df.loc[len(df.index)] = last_stat
        # print(f"{dir:>40} | {last_te_bce_error}")

if df is None:
    print("No result found.")
    exit(1)

df = df.sort_values(by=["te_bce_error"])
df = df.reset_index(drop=True)
print(df)
df.to_csv("results_report.csv", index=False)

