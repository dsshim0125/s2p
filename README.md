# S2P: State-conditioned Image Synthesis for Data Augmentation in Offline Reinforcement Learning

## ICML 2022 submission

### Setup
```shell
conda create -n ht_dcmnet python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Our experiments have been done with PyTorch 1.10.1, CUDA 11.4, Python 3.8.5 and Ubuntu 18.04. We use  a single NVIDIA RTX A6000 for training, but you can still run our code with GPUs which have smaller memory by reducing the batchSize. A simpel visualziation of the generation results can be done by GPUs with 4GB of memory use.

### Simple Generation

```shell
python evaluate.py --env_type=cheetah --netG=s2p_light --dataroot=./datasets/cheetah.hdf5 --start_idx=0 --seq_len=5
```

### Train S2P

```shell
python train.py --env_type=cheetah --netG=s2p --dataroot=./datasets/cheetah.hdf5 --batchSize=16
```

### Reference
1. https://github.com/NVlabs/SPADE
2. https://github.com/yenchenlin/nerf-pytorch
3. https://github.com/huangzh13/StyleGAN.pytorch
