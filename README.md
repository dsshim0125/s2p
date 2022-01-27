# S2P: State-conditioned Image Synthesis for Data Augmentation in Offline Reinforcement Learning

## ICML 2022 submission


### Simple Generation

```shell
python evaluate.py --env_type=cheetah --netG=s2p_light --dataroot=./datasets/cheetah.hdf5 --start_idx=0 --seq_len=5
```

### Train S2P

```shell
python train.py --env_type=cheetah --netG=s2p --dataroot=./datasets/cheetah.hdf5 --batchSize=16
```
