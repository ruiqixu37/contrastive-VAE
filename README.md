# Contrastive Variational Autoencoders with Barlowtwins models

*Tufts CS 152 FA22 Final Project*

## Installation

This repository utilizes `pipenv` for dependency management. If you have not installed Pipenv already, run the following:
~~~
pip install pipenv
~~~

The dependencies can be installed with the following command:
~~~
pipenv install 
~~~

PyTorch is not included in the above install command and need to be installed seperately, as PyTorch is dependent on the hardware specification. To install PyTorch, run with the following command:
~~~
pipenv run pip3 install torch torchvision torchaudio --index-url ...
~~~
where --index-url points to the exact PyTorch version you wish to install.

## Training
First, datasets need to be installed at the `Data/` directory. The default dataset, `CelebA`, can be installed via this [link](https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing).

To start the training script, use:
```
pipenv run python bt_vae/train_vae.py
```