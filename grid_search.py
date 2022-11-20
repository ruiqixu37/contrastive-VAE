from grid_search_config import CMD, NUM_PARALLEL_JOBS, PARAMS_TO_SEARCH
from multiprocessing.pool import ThreadPool
import subprocess

params_list = []

def worker(params):
    ident = ", ".join(params[1::2])
    print(f'Working on {ident}')
    process = subprocess.Popen(
        [*CMD, *params], 
        stdout=subprocess.PIPE, 
        universal_newlines=True,
        bufsize=8192
    )
    for line in process.stdout:
        if "on test" in line and "per-pixel VI-loss" in line:
            print(ident + " | " + line.strip())
    print(f'Finished {ident}')


for n_epochs in PARAMS_TO_SEARCH['n_epochs']:
    for lr in PARAMS_TO_SEARCH['lr']:
        for n_dims_code in PARAMS_TO_SEARCH['n_dims_code']:
            for hidden_sizes in PARAMS_TO_SEARCH['hidden_layer_sizes']:
                for method in PARAMS_TO_SEARCH['method']:
                    for n_mc_samples in PARAMS_TO_SEARCH['n_mc_samples']:
                        params_list.append([
                            "--method", str(method),
                            "--lr", str(lr),
                            "--n_epochs", str(n_epochs),
                            "--n_dims_code", str(n_dims_code),
                            "--hidden_layer_sizes", str(hidden_sizes),
                            "--n_mc_samples", str(n_mc_samples)
                        ])

tpool = ThreadPool(NUM_PARALLEL_JOBS)
for params in params_list:
    tpool.apply_async(worker, (params,))

tpool.close()
tpool.join()
