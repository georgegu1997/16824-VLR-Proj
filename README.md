# 16824-VLR-Proj
Course project repo for 16824 Visual Learning and Representation @ CMU

The implementation is based on [PyTorch](https://pytorch.org/), [PyTorch-Lightning](https://www.pytorchlightning.ai/), [Hydra](https://hydra.cc/docs/intro/) and [wandb](https://docs.wandb.ai/)

## Get started

Create a python environment and install the following package
```
pytorch
pytorch-lightning
hydra
wandb
```

Install this project codebase as a package: at the root folder of this project run
```
pip install -e .
```

## Run the experiment

* Navigate to `python/ppf_net/` folder
* Simple run `python train.py`