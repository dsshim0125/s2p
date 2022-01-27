# S2P: State-conditioned Image Synthesis for Data Augmentation in Offline Reinforcement Learning

## ICML 2022 submission

### Setup
```shell
conda create -n s2p python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
Our experiments have been done with PyTorch 1.10.1, CUDA 11.4, Python 3.8.5 and Ubuntu 18.04. We use  a single NVIDIA RTX A6000 for training, but you can still run our code with GPUs which have smaller memory by reducing the batchSize. A simpel visualziation of the generation results can be done by GPUs with 4GB of memory use.

### Simple Generation
Due to the meomory limit in the submission of supplementary materials, we cannot provide pre-trained weights of our S2P architecture. Instead, we provide the pre-trained weights of S2P_light which is lighter version of S2P in two different tasks, cheetah and walker.

After camera-ready submission, we will publicly open the source code and the pre-trained weights of our full S2P in 6 different tasks we've shown in the paper.
```shell
python evaluate.py --dataroot=./datasets/cheetah.hdf5 --env_type=cheetah --netG=s2p_light --start_idx=0 --seq_len=5 --gpu_ids=0
```

### Train S2P

```shell
python train.py --dataroot=./datasets/cheetah.hdf5 --env_type=cheetah --netG=s2p --batchSize=16 --gpu_ids=0
```

### Reference
1. https://github.com/NVlabs/SPADE
2. https://github.com/yenchenlin/nerf-pytorch
3. https://github.com/huangzh13/StyleGAN.pytorch
