# S2P: State-conditioned Image Synthesis for Data Augmentation in Offline Reinforcement Learning

## ICML 2022 submission

## Setup
```shell
conda create -n s2p python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
Our experiments have been done with PyTorch 1.10.1, CUDA 11.4, Python 3.8.5 and Ubuntu 18.04. 
We use  a single NVIDIA RTX A6000 for training, but you can still run our code with GPUs which have smaller memory by reducing the batchSize. 
A simpel visualziation of the generation results can be done by GPUs with 4GB of memory use.

## Download pre-trained models

Create a folder ```./checkpoints``` and download the model weights into it. 
Here are model weights of S2P trained on cheetah and walker environment of DeepMind Controp Suite.

| Env_type  |  model  |
|----------|:--:|
|cheetah|[cheetah_30.pth](https://drive.google.com/file/d/1Q3fGEIT99BeeLNokkNAwmWv7r5L9Z7LK/view?usp=sharing)|
|walker|[walker_30.pth](https://drive.google.com/file/d/1NKfoIcTJapEzor5VEISnewNi-7_8N5QO/view?usp=sharing)|

## Simple generation

We provide pre-trained models of S2P and some tiny dataset for simple visualization of S2P.
Reviewers can easily visualize N-step generation results with ```--seq_len``` flag in 2 different environments (cheetah, walker).

```shell
python simple_test.py --env_type=cheetah --dataroot=./datasets --netG=s2p --start_idx=0 --seq_len=5 --gpu_ids=0
```


## Offline RL setup
```shell
pip install mujoco-py<2.2,>=2.1
pip install git+https://github.com/deepmind/dm_control
```
We generated the DMControl environment dataset by training the state-based SAC, following the implementation of the https://github.com/rail-berkeley/rlkit.

But, due to the memory limit in the submission of supplementary materials, we cannot provide full offline dataset used for the paper. Instead, we provide tiny dataset of the cheetah-run-mixed environment with the state transition rollout by the random policy in https://drive.google.com/drive/folders/15WzMg_OAN9PBHFNw8iTRcBvnjsqK9TsZ?usp=sharing. Download the cheetah-run-mixed_first_500k folder and paste it in to data/trajwise folder of this repository.  



If you want to follow the generating process of the state transition rollout by the random policy, you should run the below code after download the cheetah-run-mixed_first_500k folder.

### state transition data rollout
```shell
python state_transition_rollout.py
```
Then, you should run generation code... (TODO:DSSHIM)

### Train S2P

```shell
python train.py --dataroot=./datasets/cheetah.hdf5 --env_type=cheetah --netG=s2p --batchSize=16 --gpu_ids=0
```


Then you can get the all_state_1step_random_action_dataset_naive.hdf5 in data/trajwise/cheetah-run-mixed_first_500k/all_state_1step_random_action folder.

For training offline RL, run the below python code.
### Train Offline RL
```shell
bash run_iql_image.sh
bash run_cql_image.sh
```

### Reference
1. https://github.com/NVlabs/SPADE
2. https://github.com/yenchenlin/nerf-pytorch
3. https://github.com/huangzh13/StyleGAN.pytorch
