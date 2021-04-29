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

# Download dataset

The pre-processed ModelNet40 dataset are used, containing sampled point cloud with normal vectors. It can be downloaded [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip). Put the zip under `data/` folder and unzip it. At the first time running the ModelNet40Cls dataset, the data will be processed again and cache for fast dataloading will be generated. 

## Run the experiment

* Navigate to `python/ppf_net/` folder
* Run `python train.py`

### Reproduce robust point net result

Train only on the original data and test on rotation-perturbed data
```
python train.py dataset=rps dataset.test_tasks=["test_rotation.npy"] exp_suffix="testrot"
```

### Train with processed ModelNet data

```
python train.py dataset=modelnet dataset.normal=False 

python train.py dataset=modelnet dataset.normal=True exp_suffix="normal"

