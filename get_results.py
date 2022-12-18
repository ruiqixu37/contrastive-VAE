import os
import pathlib
import pandas as pd
import json

def get_csv(file_list):
    for file in file_list:
        if file.endswith("metrics.csv"):
            return file
    return ""

def get_read_config(dir):
    with open(os.path.join(dir, "params.json"), 'r') as f:
        return json.load(f)

path = "results/"
dirs = os.listdir(path)

df = None

# print(f"{'folder':>40} | result")
# print("----------------------------------------------------")
for dir in dirs:
    d = os.path.join(path, dir)
    if os.path.isdir(d):
        contents = os.listdir(d)
        csv_file = get_csv(contents)
        if csv_file == "":
            continue
        config = get_read_config(d)
        curr_run_data = pd.read_csv(os.path.join(d, csv_file))

        if df is None:
            df = pd.DataFrame(columns=['run_name', 'type', 'lambda_bt', 'n_dims_code', *curr_run_data.columns])
            df['run_name'] = df['run_name'].astype('str')
            df['type'] = df['type'].astype('str')
        
        last_stat = curr_run_data.iloc[-1].copy()
        last_stat['run_name'] = dir
        last_stat['type'] = config['method']
        last_stat['lambda_bt'] = config['lambda_bt']
        last_stat['n_dims_code'] = config['n_dims_code']
        print("Processed", d)
        df.loc[len(df.index)] = last_stat
        # print(f"{dir:>40} | {last_te_bce_error}")

if df is None:
    print("No result found.")
    exit(1)

df = df.sort_values(by=["te_bce_error"])
df = df.reset_index(drop=True)
pd.set_option('display.max_rows', None)
print()
print("                   ******** Table of results, sorted by test bce error ********")
print(df)
df.to_csv("results_report.csv", index=False)

