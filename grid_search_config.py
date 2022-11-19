CMD = ["python3", "training_on_mnist.py"]

PARAMS_TO_SEARCH = {
    "method": ["VAE", "VAET", "VAEBT", "VAEBTT"],
    "n_epochs": [200],
    "lr": [0.005],
    "n_dims_code": [10, 50],
    "hidden_layer_sizes": [32, 64, 128, "128,32"]
}

NUM_PARALLEL_JOBS = 4
