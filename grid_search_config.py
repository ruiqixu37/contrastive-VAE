CMD = ["python3", "training_on_mnist.py"]

PARAMS_TO_SEARCH = {
    "method": ["VAEBT"],
    "n_epochs": [200],
    "lr": [0.005],
    "n_dims_code": [2, 16, 64, 128],
    "n_mc_samples": [10],
    "lambda_bt": [0.25, 0.025, 0.0025, 0.00025, -0.00025],
    "hidden_layer_sizes": [32, 64, 128]
}

NUM_PARALLEL_JOBS = 5
