
import gym
import numpy as np
import os
from PIL import Image
import dmc2gym
import torch
import torch.nn as nn    
from gaussian_ensemble import EnsembleTransition
import h5py

def generate_rollout_by_dynamics_reward_ftn():
    
    trajwise_data_for_slac = True    
    if trajwise_data_for_slac:
        slac_num_sequences = 8        
        dm_control_env_dict = {'cheetah-run-mixed_first_500k' : dict(domain_name='cheetah', task_name='run', frame_skip=4, load_dir = './data/trajwise/cheetah-run-mixed_first_500k')}
       
    normalize_all_pkl_iter_dict = {'cheetah-run-mixed_first_500k' : 590}
    
    dist_type = 'ensemble' 
    rollout_type = ['all_state_1step_random_action']
    rollout = rollout_type[-1]
    normalize_type = ['dynamics', 'all']    
    normalize = normalize_type[1]
    seed = np.random.randint(10000)
    np.random.seed(seed)
        
    for env_name, env_kwargs in dm_control_env_dict.items():        
        print('env name : ', env_name)
        load_dir = env_kwargs.pop('load_dir')
        
        if 'cheetah-run' in env_name:
            model_load_dir = './world_model/'+'cheetah-run-mixed_first_500k'+'/'+dist_type
            qvel_dim = 9
        
        if normalize=='all':
            model_load_dir = model_load_dir + '/normalize_all'
            pkl_name = 'model_dist_state_dict_'+str(normalize_all_pkl_iter_dict[env_name])+'.pkl'


        env = dmc2gym.make(**env_kwargs, visualize_reward=False, seed = seed)
        
        ensemble_dist_kwargs = dict(obs_dim = env.observation_space.shape[0], 
                                    action_dim = env.action_space.shape[0], 
                                    hidden_features = 256, 
                                    hidden_layers = 3, 
                                    ensemble_size=7, 
                                    mode='local', 
                                    with_reward=True,
                                    )

        device = torch.device("cuda:" + str(0))
        
        if dist_type=='ensemble':
            model = EnsembleTransition(**ensemble_dist_kwargs)
        
        
        dataset = {}
        
        with h5py.File(load_dir+'/image_numpy_dataset_stack3_imgsize_100.hdf5', 'r') as h5py_r:            
            dataset['observations'] = h5py_r['observations'][:] #[bs, dim]            
            dataset['rewards'] = h5py_r['rewards'][:]
            dataset['next_observations'] = h5py_r['next_observations'][:]
            dataset['image_observations'] = h5py_r['image_observations'][:]
            dataset['image_observations_tm1'] = h5py_r['image_observations_tm1'][:]
            dataset['image_observations_tm2'] = h5py_r['image_observations_tm2'][:]
            dataset['qpos_qvel']=h5py_r['qpos_qvel'][:]
            if trajwise_data_for_slac:
                dataset['original_actions'] = h5py_r['actions'][:]
                dataset['original_rewards'] = h5py_r['rewards'][:]
                dataset['terminals'] = h5py_r['terminals'][:]
                dataset['timeouts'] = h5py_r['timeouts'][:]
                original_data_timeout_indices = np.sort(np.where(h5py_r['timeouts'][:]==1)[0])
                if len(original_data_timeout_indices)==0:
                    raise NotImplementedError('You should generate image_numpy_dataset_stack3_imgsize_100.hdf5 by image_render.py')
                print('original data timeout indices : ', original_data_timeout_indices)
                assert (h5py_r['terminals'][:]==0).all(), 'Assume there are no terminal state in the dataset (DMControl)'
            
        h5py_r.close()
        del h5py_r

        original_data_size = dataset['observations'].shape[0]
        if len(dataset['rewards'].shape)==2:
            dataset['rewards'] = dataset['rewards'].squeeze(-1)
            print('reward is squeezed. shape : ', dataset['rewards'].shape)

        if normalize=='all':                        
            normalize_configs = torch.load(model_load_dir+'/normalize_configs_dict.pkl')
            obs_mean = normalize_configs['obs_mean']
            obs_std = normalize_configs['obs_std']
            next_obs_mean = normalize_configs['next_obs_mean']
            next_obs_std = normalize_configs['next_obs_std']
            reward_mean = normalize_configs['reward_mean']
            reward_std = normalize_configs['reward_std']
            

            
        if dist_type=='ensemble':            
            model.load_state_dict(torch.load(model_load_dir+'/'+pkl_name))        
            model.to(device)
            print('load model!')

        
        def generate_obs_act_indices(traj_length, traj_start_idx):
            assert traj_length > slac_num_sequences, 'traj length : {} slac num seq : {}'.format(traj_length, slac_num_sequences)

            obs_list = []
            act_list = []                    
               
            integer_inf = int(1e9) # 1B # It will be used in offline RL for indexing
            for i in range(traj_length):
                if i < slac_num_sequences:
                    temp_obs_indices = np.array([integer_inf]*(slac_num_sequences+1))
                else: #from i = 8
                    temp_obs_indices = np.arange(i-slac_num_sequences, i+1) + traj_start_idx
                obs_list.append(temp_obs_indices)

            for i in range(traj_length):
                if i < slac_num_sequences:                            
                    temp_act_indices = np.array([integer_inf]*(slac_num_sequences))                            
                else: #from i = 8
                    temp_act_indices = np.arange(i-slac_num_sequences, i) + traj_start_idx
                act_list.append(temp_act_indices)
            # NOTE : Assume all traj length is same
            obs_indices = np.stack(obs_list, axis =0) # + traj_length*traj_idx # [traj_length, num_seq(8)]
            act_indices = np.stack(act_list, axis =0) # + traj_length*traj_idx # [traj_length, num_seq-1(7)]

            assert obs_indices.shape == (traj_length, slac_num_sequences+1)
            assert act_indices.shape == (traj_length, slac_num_sequences)

            return obs_indices.astype(int), act_indices.astype(int)

        if rollout in ['all_state_1step_random_action']:
            if trajwise_data_for_slac:
                print('original data size : ', original_data_size)
                act_low = env.action_space.low
                act_high = env.action_space.high
                
                action_list = []
                reward_list = []
                next_observation_list = []
                disagreement_uncertainty_list = []
                aleatoric_uncertainty_list = []
                # fos slac
                slac_action_indices_list = []
                slac_observation_indices_list = []

                normalized_obs = (dataset['observations'] - obs_mean)/obs_std

                for idx, timeout_idx in enumerate(original_data_timeout_indices):
                    
                    if idx == 0:
                        start_idx = 0
                        end_idx = timeout_idx 
                    else:
                        start_idx = original_data_timeout_indices[idx-1]+1
                        end_idx = timeout_idx 
                    
                    print('env : {}  traj : {} index from : {} to : {}'.format(env_name, idx, start_idx, end_idx)) if idx % 50 ==0 else None
                    
                    assert end_idx == original_data_timeout_indices[idx]
                    
                    if end_idx >= original_data_size:
                        raise NotImplementedError
                    
                    batch_size = end_idx-start_idx+1
                    traj_length = end_idx-start_idx+1
                    
                    obs_indices, act_indices = generate_obs_act_indices(traj_length=traj_length, traj_start_idx=start_idx)

                    
                    if rollout=='all_state_1step_random_action':
                        batch_obs = torch.from_numpy(normalized_obs[start_idx:end_idx+1]).to(device)
                        actions = np.random.uniform(low=act_low, high=act_high, size = (batch_size, env.action_space.shape[0])).astype(np.float32)                        
                    

                    batch_act = torch.from_numpy(actions).to(device)
                    if dist_type=='ensemble':
                        dist = model(torch.cat([batch_obs, batch_act], axis =-1).to(device))
                        predicted_mean = dist.mean # [ensemble, batch, dim]
                        
                        predicted_obs = predicted_mean[:, :, :env.observation_space.shape[0]]
                        predicted_rew = predicted_mean[:, :, -1] #[ensemble, batch]              
                        
                        if rollout in ['all_state_1step_random_action']:
                            assert predicted_mean.shape == (7, batch_size, env.observation_space.shape[0]+1)

                        num_data_generation = predicted_obs.shape[1]

                        if normalize=='all':
                            ensemble_idx = np.random.randint(0, 7, size=num_data_generation)                    
                            batch_indices = np.arange(num_data_generation)
                            predicted_next_obs = predicted_obs[ensemble_idx, batch_indices].detach().cpu().numpy()*next_obs_std + next_obs_mean  #[bs,dim]
                            predicted_reward = predicted_rew[ensemble_idx, batch_indices].detach().cpu().numpy()*reward_std + reward_mean #[bs,]
                            
                            ####### added for uncertainty measure
                            if rollout in ['all_state_1step_random_action']:
                                next_obses_mode = dist.mean[:, :, :-1] # [ensemble, batch, dim]
                            
                            next_obs_mean_ensemble_average = torch.mean(next_obses_mode, dim=0).detach() #average alogn ensemble [bs, dim]
                            diff = next_obses_mode - next_obs_mean_ensemble_average
                            disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0].detach().cpu().numpy() #[bs, 1]
                            aleatoric_uncertainty = torch.max(torch.norm(dist.stddev, dim=-1, keepdim=True), dim=0)[0].detach().cpu().numpy() #[bs, 1]
                        
                        else:
                            raise NotImplementedError
                        assert predicted_next_obs.shape == (batch_size, env.observation_space.shape[0])
                        assert predicted_reward.shape == (batch_size,)

                    action_list.append(actions) #[small_bs, dim]
                    reward_list.append(predicted_reward)
                    next_observation_list.append(predicted_next_obs)
                    disagreement_uncertainty_list.append(disagreement_uncertainty)
                    aleatoric_uncertainty_list.append(aleatoric_uncertainty)
                    
                    # for slac (store previous num_seq obs & act indices)
                    slac_action_indices_list.append(act_indices) #[traj_length, num_seq]
                    slac_observation_indices_list.append(obs_indices) #[traj_length, num_seq+1]


                dataset['actions'] = np.concatenate(action_list, axis =0) #[bs, dim]
                dataset['rewards'] = np.concatenate(reward_list, axis =0)
                dataset['next_observations'] = np.concatenate(next_observation_list, axis =0)
                dataset['disagreement_uncertainty'] = np.concatenate(disagreement_uncertainty_list, axis =0) #[bs,] or [bs, dim]
                dataset['aleatoric_uncertainty'] = np.concatenate(aleatoric_uncertainty_list, axis =0) #[bs,] or [bs, dim]
                # for slac
                dataset['slac_action_indices'] = np.concatenate(slac_action_indices_list, axis =0) #[bs(all), num_seq]
                dataset['slac_observation_indices'] = np.concatenate(slac_observation_indices_list, axis =0) #[bs(all), num_seq+1]

        
        h5f_w = h5py.File(load_dir+'/'+rollout+'/all_state_1step_random_action_dataset_naive.hdf5', 'w')
        import time
        start = time.time()
        for k,v in dataset.items():
            try:
                print('key : {} value shape : {}'.format(k, v.shape))
            except Exception as e:
                print('key : {} value.shape has error'.format(k))
            h5f_w.create_dataset(str(k), data=v)
        print('saving hdf5 time : ', time.time() - start)
        h5f_w.close()
        del h5f_w


if __name__=='__main__':    
    generate_rollout_by_dynamics_reward_ftn()