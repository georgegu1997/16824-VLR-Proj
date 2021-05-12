# 16824-VLR-Proj
Course project repo for 16824 Visual Learning and Representation @CMU

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
* Run the following code repectively

### Reproduce RobustPointSet result

Train only on the original data and test on rotation-perturbed data
```
python train.py dataset=rps dataset.test_tasks=["test_rotation.npy"] exp_suffix="testrot"
```

### Train with processed ModelNet40 data

```
python train.py dataset=modelnet dataset.normal=False 
python train.py dataset=modelnet dataset.normal=True exp_suffix=normal
```

### Train PointNet with computed Point Pair Features (PPFs) 

```
# Different sampling strategy of sampling the PPF reference points
python train.py dataset=modelnet dataset.normal=True model.ppf_mode=random exp_suffix=ppfrandom
python train.py dataset=modelnet dataset.normal=True model.ppf_mode=mean exp_suffix=ppfmean
python train.py dataset=modelnet dataset.normal=True model.ppf_mode=far exp_suffix=ppffar
```

### Train PointNet++

```
python train.py model=pn2 dataset=modelnet dataset.normal=True exp_suffix=normal
python train.py model=pn2 dataset=modelnet dataset.normal=True model.ppf_mode=mean exp_suffix=ppfmean
# Compute the PPF at the first abstration layer
python train.py model=pn2 dataset=modelnet dataset.normal=True model.ppf_first=True exp_suffix=ppf_first
```

### Train DGCNN

```
# Different sampling strategy of sampling the PPF reference points
python train.py model=dgcnn dataset=modelnet dataset.normal=True model.ppf_mode=random exp_suffix=ppfrandom
python train.py model=dgcnn dataset=modelnet dataset.normal=True model.ppf_mode=mean exp_suffix=ppfmean
python train.py model=dgcnn dataset=modelnet dataset.normal=True model.ppf_mode=far exp_suffix=ppffar
```

# Train with augmentation

### Not using PPF
```
python train.py model=pn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=False exp_suffix=trainaug
CUDA_VISIBLE_DEVICES=1 python train.py model=pn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=True exp_suffix=trainaug_validrot

python train.py model=pn2 dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=False exp_suffix=trainaug
CUDA_VISIBLE_DEVICES=1 python train.py model=pn2 dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=True exp_suffix=trainaug_validrot

python train.py model=dgcnn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=False exp_suffix=trainaug
CUDA_VISIBLE_DEVICES=1 python train.py model=dgcnn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=True exp_suffix=trainaug_validrot
```

### Using PPF
```
python train.py model=pn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=False model.ppf_mode=mean exp_suffix=ppfmean_trainaug
CUDA_VISIBLE_DEVICES=1 python train.py model=pn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=True model.ppf_mode=mean exp_suffix=ppfmean_trainaug_validrot

python train.py model=pn2 dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=False model.ppf_first=True exp_suffix=ppf_first_trainaug
CUDA_VISIBLE_DEVICES=1 python train.py model=pn2 dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=True model.ppf_first=True exp_suffix=ppf_first_trainaug_validrot

python train.py model=dgcnn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=False model.ppf_mode=mean exp_suffix=ppfmean_trainaug
CUDA_VISIBLE_DEVICES=1 python train.py model=dgcnn dataset=modelnet dataset.normal=True dataset.train_aug=True dataset.valid_rot=True model.ppf_mode=mean exp_suffix=ppfmean_trainaug_validrot
```