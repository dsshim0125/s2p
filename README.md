# S2P: State-conditioned Image Synthesis for Data Augmentation in Offline Reinforcement Learning

## ICML 2022 submission

### Setup
```shell
conda create -n ht_dcmnet python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Our experiments has been done with PyTorch 1.9.0, CUDA 11.2, Python 3.8.5 and Ubuntu 18.04. We use 4 NVIDIA RTX 3090 GPUs for training, but you can still run our code with GPUs which have smaller memory by reducing the batch_size. A simpel visualziation can be done by GPUs with 3GB of memory use or CPU only is also functional.

### Simple Generation

```shell
python evaluate.py --env_type=cheetah --netG=s2p_light --dataroot=./datasets/cheetah.hdf5 --start_idx=0 --seq_len=5
```

### Train S2P

```shell
python train.py --env_type=cheetah --netG=s2p --dataroot=./datasets/cheetah.hdf5 --batchSize=16
```
