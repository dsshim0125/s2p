import os

import numpy as np
import torch
from torch.optim import Adam

from rlkit.torch.slac.buffer import ReplayBuffer
from rlkit.torch.slac.network import GaussianPolicy, LatentModel, TwinnedQNetwork
from rlkit.torch.slac.utils import create_feature_actions, grad_false, soft_update


class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        action_repeat,
        device,
        seed,
        gamma=0.99,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 5,
        num_sequences=8,
        lr_sac=3e-4,
        lr_latent=1e-4,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
        tau=5e-3,
        
        image_size = 64,
        use_seperate_buffer = False,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, num_sequences, state_shape, action_shape, device)
        self.use_seperate_buffer = use_seperate_buffer
        if use_seperate_buffer:
            self.buffer_gen = ReplayBuffer(buffer_size, num_sequences, state_shape, action_shape, device)

        # Networks.
        self.latent = LatentModel(state_shape, action_shape, feature_dim, z1_dim, z2_dim, hidden_units, image_size = image_size).to(device)
        

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])
        

        # Optimizers.
        self.optim_latent = Adam(self.latent.parameters(), lr=lr_latent)

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau

        # JIT compile to speed up.
        fake_feature = torch.empty(1, num_sequences + 1, feature_dim, device=device)
        fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        with torch.no_grad():
            feature = self.latent.encoder(state).view(1, -1) #[bs, num_seq, dim]->[bs, num_seq*dim]
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device) #[bs, num_seq*dim]
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_random):
        t += 1

        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)

        state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        ob.append(state, action)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            t = 0
            state = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return t

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)
        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.optim_latent.step()
        return loss_kld, loss_image, loss_reward


    def prepare_batch(self, state_, action_): 
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_) #[bs, num_seq, dim]
            # z(1:t+1)
            z_ = torch.cat(self.latent.sample_posterior(feature_, action_)[2:4], dim=-1) # return (z1_mean_, z1_std_, z1_, z2_)를 concat -> [bs, num_seq, z1_dim + z2_dim]

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1] #[bs, dim]
        # a(t)
        action = action_[:, -1] #[bs,  dim]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        
    
    
    def load_data_in_buffer(self, h5f_r_name, savedir, env, data_num=None, uncertainty_type = None, uncertainty_penalty_lambda = None, \
                            generated_for_slac = False, data_mix_type=None, generalization_test = None, ball_in_cup_debug = False):
        import h5py
        import os
        import torch
        import time
        dataset = {}                
        if savedir is not None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)

        h5py_r = h5py.File(h5f_r_name, 'r')
        start = time.time()
        if data_num is None:
            dataset['observations'] = h5py_r['observations'][:]
            dataset['actions'] = h5py_r['actions'][:]
            dataset['rewards'] = h5py_r['rewards'][:]
            dataset['next_observations'] = h5py_r['next_observations'][:]
            # dataset['terminals'] = h5py_r['terminals'][:]
            dataset['timeouts'] = h5py_r['timeouts'][:]            
            dataset['image_observations'] = np.transpose(h5py_r['image_observations'][:], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
            dataset['image_observations_tp1'] = np.transpose(h5py_r['image_observations_tp1'][:], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
        elif data_num ==0:
            return
        else:
            if generated_for_slac:
                if data_mix_type=='all_state_1step_random_action':
                    
                    # should access trajwise (일단은 data num까지 자르고 그 안에서 trajwise로 재정리)
                    dataset['observations'] = h5py_r['observations'][:data_num]
                    dataset['actions'] = h5py_r['actions'][:data_num]
                    dataset['rewards'] = h5py_r['rewards'][:data_num]
                    dataset['next_observations'] = h5py_r['next_observations'][:data_num]
                    # dataset['terminals'] = h5py_r['terminals'][:data_num]
                    dataset['timeouts'] = h5py_r['timeouts'][:data_num]                      
                    dataset['image_observations'] = np.transpose(h5py_r['image_observations'][:data_num], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
                    dataset['image_observations_tp1'] = np.transpose(h5py_r['image_observations_tp1'][:data_num], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]

                    dataset['original_actions'] = h5py_r['original_actions'][:data_num]
                    dataset['original_rewards'] = h5py_r['original_rewards'][:data_num]

                    dataset['slac_observation_indices'] = h5py_r['slac_observation_indices'][:data_num]
                    dataset['slac_action_indices'] = h5py_r['slac_action_indices'][:data_num]
                    
                    if uncertainty_type=='aleatoric':
                        dataset['aleatoric_uncertainty'] = h5py_r['aleatoric_uncertainty'][:data_num].squeeze(-1)
                    elif uncertainty_type=='disagreement':
                        dataset['disagreement_uncertainty'] = h5py_r['disagreement_uncertainty'][:data_num].squeeze(-1)
                    elif uncertainty_type in ['max_of_both','min_of_both','average_both'] :
                        dataset['aleatoric_uncertainty'] = h5py_r['aleatoric_uncertainty'][:data_num].squeeze(-1)
                        dataset['disagreement_uncertainty'] = h5py_r['disagreement_uncertainty'][:data_num].squeeze(-1)                
                    elif uncertainty_type is None:
                        pass
                    else:
                        raise NotImplementedError

                    assert uncertainty_penalty_lambda is not None

                elif data_mix_type in ['random_state_5step_random_action', 'random_state_1step_random_action', 'random_state_5step_offRL_action']:
                    # dataset['observations'] = h5py_r['observations'][:data_num]
                    dataset['actions'] = h5py_r['actions'][:data_num]
                    dataset['rewards'] = h5py_r['rewards'][:data_num]
                    if data_mix_type in ['random_state_1step_random_action', 'random_state_5step_random_action']:
                        dataset['next_observations'] = h5py_r['next_observations'][:data_num]
                    # dataset['terminals'] = h5py_r['terminals'][:data_num]
                    dataset['timeouts'] = h5py_r['timeouts'][:data_num]                      
                    dataset['image_observations'] = np.transpose(h5py_r['image_observations'][:data_num], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
                    dataset['image_observations_tp1'] = np.transpose(h5py_r['image_observations_tp1'][:data_num], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]

                    if uncertainty_type=='aleatoric':
                        dataset['aleatoric_uncertainty'] = h5py_r['aleatoric_uncertainty'][:data_num].squeeze(-1)
                        assert uncertainty_penalty_lambda is not None
                    elif uncertainty_type=='disagreement':
                        dataset['disagreement_uncertainty'] = h5py_r['disagreement_uncertainty'][:data_num].squeeze(-1)
                        assert uncertainty_penalty_lambda is not None
                    elif uncertainty_type in ['max_of_both','min_of_both','average_both'] :
                        dataset['aleatoric_uncertainty'] = h5py_r['aleatoric_uncertainty'][:data_num].squeeze(-1)
                        dataset['disagreement_uncertainty'] = h5py_r['disagreement_uncertainty'][:data_num].squeeze(-1)                
                        assert uncertainty_penalty_lambda is not None
                    elif uncertainty_type is None:
                        pass
                    else:
                        raise NotImplementedError

                print('uncertainty_type : ', uncertainty_type)
                    

            else:
                dataset['observations'] = h5py_r['observations'][:data_num]
                dataset['actions'] = h5py_r['actions'][:data_num]
                dataset['rewards'] = h5py_r['rewards'][:data_num]
                dataset['next_observations'] = h5py_r['next_observations'][:data_num]
                # dataset['terminals'] = h5py_r['terminals'][:data_num]
                dataset['timeouts'] = h5py_r['timeouts'][:data_num]            
                dataset['image_observations'] = np.transpose(h5py_r['image_observations'][:data_num], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
                dataset['image_observations_tp1'] = np.transpose(h5py_r['image_observations_tp1'][:data_num], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
            

        
        original_data_size = dataset['actions'].shape[0]
        print('original data size : ', original_data_size)
        

        print('load h5py time : ', time.time()-start)
        
        timeout_indices = np.where(dataset['timeouts']==1)[0]
        
        print('timeout indices : ', timeout_indices)
        h5py_r.close()
        del h5py_r
        
        # Time to start training.
        
        
        if generated_for_slac and data_mix_type=='all_state_1step_random_action':
               
            if self.use_seperate_buffer:
                pass            
            else:      
                self.buffer.buff.reset()            
            t = 0
            integer_inf = int(1e9)
            buffer_input_start = time.time()
            
            for i in range(original_data_size):            
                
                # for SLAC
                slac_obs_indices = dataset['slac_observation_indices'][i] #[num_seq+1(9)]
                slac_act_indices = dataset['slac_action_indices'][i] #[num_seq(8)]
                slac_rew_indices = dataset['slac_action_indices'][i] #[num_seq(8)]
                slac_done_indices = dataset['slac_action_indices'][i] #[num_seq(8)]
                

                assert (slac_act_indices == slac_obs_indices[:-1]).all()

                if (slac_obs_indices>=integer_inf).any(): # Ignore first (num_seq(8)) steps in trajectory
                    assert (slac_obs_indices>=integer_inf).all()
                    continue
                

                if i==original_data_size-1: # when last data
                    timeout = dataset['timeouts'][i]
                    # print('last data timeout is : ', bool(timeout))
                    if timeout:
                        break

                print('buffer storing : ', i) if i % 1000 ==0 else None
                
                previous_state = dataset['image_observations'][slac_obs_indices] #[num_seq+1(9), c, h, w]
                previous_act = dataset['original_actions'][slac_act_indices] #[num_seq(8), dim]
                previous_rew = dataset['original_rewards'][slac_rew_indices] #[num_seq(8),]
                previous_timeout = dataset['timeouts'][slac_done_indices] #[num_seq(8), ]
                
                state = previous_state[0]
                if self.use_seperate_buffer:
                    self.buffer_gen.reset_episode(state)
                else:
                    self.buffer.reset_episode(state)
                for j in range(self.num_sequences):                    
                    timeout = previous_timeout[j]
                    if timeout:
                        print('Logging : Originally, it is desired to be no timeouts in this loop. timeout i : {} j : {}'.format(i,j))
                        raise NotImplementedError

                    if j==self.num_sequences-1: # augmented(generated) action, reward by dynamics, reward model                        
                        # print('when j : {}, i-1  : {}'.format(j,i-1)) if i <= 10000 else None
                        action = dataset['actions'][i-1]          
                        if uncertainty_type=='aleatoric':
                            reward = dataset['rewards'][i-1] - uncertainty_penalty_lambda*dataset['aleatoric_uncertainty'][i-1]
                        elif uncertainty_type=='disagreement':
                            reward = dataset['rewards'][i-1] - uncertainty_penalty_lambda*dataset['disagreement_uncertainty'][i-1]
                        elif uncertainty_type=='max_of_both':
                            reward = dataset['rewards'][i-1] - uncertainty_penalty_lambda*max(dataset['aleatoric_uncertainty'][i-1], dataset['disagreement_uncertainty'][i-1])
                        elif uncertainty_type=='min_of_both':
                            reward = dataset['rewards'][i-1] - uncertainty_penalty_lambda*min(dataset['aleatoric_uncertainty'][i-1], dataset['disagreement_uncertainty'][i-1])
                        elif uncertainty_type=='average_both':
                            reward = dataset['rewards'][i-1] - uncertainty_penalty_lambda*0.5*(dataset['aleatoric_uncertainty'][i-1]+dataset['disagreement_uncertainty'][i-1])
                        elif uncertainty_type is None:
                            reward = dataset['rewards'][i-1]

                        else:
                            raise NotImplementedError
                        state = dataset['image_observations_tp1'][i-1]
                        done = True # episode_done when last seq
                    else:                        
                        action = previous_act[j]
                        reward = previous_rew[j]
                        state = previous_state[j+1]
                        done = timeout
                    
                    mask = False # if timeout else done
                    assert not mask
                    if self.use_seperate_buffer:
                        self.buffer_gen.append(action, reward, mask, state, done) 
                    else:
                        self.buffer.append(action, reward, mask, state, done) 
                    
                    if done:
                        break
                        


            # print('offline data buffer store time : ', time.time() - buffer_input_start)
            if savedir is not None:
                if self.use_seperate_buffer:
                    torch.save(self.buffer_gen, savedir+'/buffer_gen.pt')
                else:
                    torch.save(self.buffer, savedir+'/buffer.pt')

            # for tsne
            if not generated_for_slac:
                import copy
                self.buffer._real_n = copy.deepcopy(self.buffer._n)

        else:            
            state = dataset['image_observations'][0]            
            if generated_for_slac: # in case of random_action_5step_random_action.
                self.buffer.buff.reset()
            self.buffer.reset_episode(state)
            t = 0
            buffer_input_start = time.time()
            for i in range(original_data_size):            
                if i==original_data_size-1:
                    timeout = dataset['timeouts'][i]
                    # print('last data timeout is : ', timeout)
                    if timeout:
                        break

                print('buffer storing : ', i) if i % 1000 ==0 else None                
                action = dataset['actions'][i]
                reward = dataset['rewards'][i]                                
                # double check for episode end
                timeout = dataset['timeouts'][i]
                done = timeout
                
                # mask = False if t == env._max_episode_steps else done
                mask = False #if t == env._max_episode_steps or timeout else done
                
                
                state = dataset['image_observations_tp1'][i] # means next state at the end of episode

                assert not mask

                self.buffer.append(action, reward, mask, state, done)
                t+=1
                if done:
                    assert timeout                
                    t = 0
                    if i==original_data_size-1: # when last data but not timeout
                        break
                    else:
                        state = dataset['image_observations'][i+1]                    
                    # when buffer.reset_episode call, the code assumes episode ends
                    self.buffer.reset_episode(state)
            
            # print('offline data buffer store time : ', time.time() - buffer_input_start)
            if savedir is not None:
                torch.save(self.buffer, savedir+'/buffer.pt')

            # for tsne
            if not generated_for_slac:
                import copy
                self.buffer._real_n = copy.deepcopy(self.buffer._n)