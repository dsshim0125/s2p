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

After the camera-ready submission, we will publicly open the source code and the pre-trained weights of our full S2P in 6 different tasks we've shown in the paper.
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


### Offline RL setup
```shell
pip install mujoco-py<2.2,>=2.1
pip install git+https://github.com/deepmind/dm_control
```



Due to the memory limit in the submission of supplementary materials, we cannot provide full offline dataset used for the paper. Instead, we provide tiny dataset of the cheetah-run-mixed environment with the state transition rollout by the random policy in /rl_data in the attached link. Download the cheetah-run-mixed_first_500k folder paste it in to data/trajwise folder. If you want to follow the generating process of the state transition rollout by the random policy, you should run the below code after download the cheetah-run-mixed_first_500k folder.

### state transition data rollout
```shell
python state_transition_rollout.py
```

Then you can get the all_state_1step_random_action_dataset_naive.hdf5, which is the same as downloaded data from the link.

For training offline RL, run the below python code.

### Train Offline RL
```shell
bash run_iql_image.sh
bash run_cql_image.sh
```












