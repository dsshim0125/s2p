from os import replace
from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import time
import torch
import rlkit.torch.pytorch_util as ptu
class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,            
            curl = False,
            crop_image_size = None,
            rad_aug = False,
            aug_funcs = None,
            **image_buffer_kwargs, 
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
         
        
        self.curl = curl
        self.crop_image_size = crop_image_size
        self.rad_aug = rad_aug
        self.aug_funcs = aug_funcs

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            **image_buffer_kwargs
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
    
    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)
        
        return batch


    def random_batch_w_relabeling(self, batch_size):
        # assume trajwise buffer
        all_episode_end_indices = np.where(self._terminals[:self._size]==1)[0] #e.g. [99, 199, 299, 340]
        all_episode_start_indices = np.concatenate([np.zeros(1), all_episode_end_indices[:-1]+1], axis =-1) #e.g. [0, 100, 200, 300] # 마지막 traj 는 그냥 뺴
        num_trajs = all_episode_end_indices.shape[0]
        
        
        n_goals = 4
        n_non_goals = 1
        n_sample_per_traj = n_goals + n_non_goals
        random_indices = np.random.randint(0, num_trajs, size =int(batch_size/n_sample_per_traj) )
        # episode_end_indices = np.sort(np.random.choice(all_episode_end_indices, size = 10, replace=self._replace))
        
        observations = []
        actions = []
        rewards = []
        dones = []
        next_observations = []
        goals = []

        for i in random_indices:
            start_idx = all_episode_start_indices[i]
            end_idx = all_episode_end_indices[i]
            current_traj_length = end_idx +1 -start_idx
            current_traj_obs = self._observations[start_idx:end_idx+1] # [current_traj_length]
            current_traj_next_obs = self._next_obs[start_idx:end_idx+1] # [current_traj_length]
            current_traj_act = self._actions[start_idx:end_idx+1] # [current_traj_length]
            
            if current_traj_length <=4:
                print('@@@@@@@ You should check very short trajectory is in buffer!')
                continue
            
            # goal_indices = np.random.randint(1, current_traj_length, size = n_goals)
            goal_indices = np.random.choice(1, current_traj_length, size = n_goals, replace=False)
            obs = current_traj_obs[goal_indices-1] #[n_goals]
            act = current_traj_act[goal_indices-1] #[n_goals]
            next_obs = current_traj_obs[goal_indices] #[n_goals]
            goal = current_traj_obs[goal_indices] #[n_goals]
            reward = np.ones(n_goals) #[n_goals]
            done = np.ones(n_goals, dtype=np.uint8)
            
            observations.append(np.concatenate([obs, goal], axis =1)) #[bs, c*2, h,w]
            actions.append(act)
            next_observations.append(np.concatenate([next_obs, goal], axis =1))
            # goals.append(goal)
            rewards.append(reward)
            dones.append(done)

            # non terminal sample 
            time_distance_min = 5
            time_distance_max = 20
            random_timestep_indices = np.random.randint(time_distance_max, current_traj_length, size =1)
            time_distance = np.random.randint(time_distance_min, time_distance_max, size =1)
            indices = random_timestep_indices - time_distance
            obs = current_traj_obs[indices] 
            act = current_traj_act[indices] 
            next_obs = current_traj_obs[indices+1] 
            goal = current_traj_obs[random_timestep_indices] 
            reward = np.zeros(n_non_goals)
            done = np.zeros(n_non_goals, dtype=np.uint8)
            
            observations.append(np.concatenate([obs, goal], axis =1))
            actions.append(act)
            next_observations.append(np.concatenate([next_obs, goal], axis =1))
            goals.append(goal)
            rewards.append(reward)
            dones.append(done)

        
        batch = dict(
            observations=np.stack(observations, axis =0),
            actions=np.stack(actions, axis =0),
            rewards=np.stack(rewards, axis =0),
            terminals=np.stack(dones, axis =0),
            next_observations=np.stack(next_observations, axis =0),
            # goals=np.stack(goals, axis =0),
        )
        # for key in self._env_info_keys:
        #     assert key not in batch.keys()
            # batch[key] = self._env_infos[key][indices]
        return batch


########### Below one might be garbage (Too SLOW!)

import itertools
class EnvReplayBufferTorchDataLoader(SimpleReplayBuffer):
    # load data when random_batch is called
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            torchvision_data_loader = None, 
            dataloader_buffer_size = None,
            **image_buffer_kwargs, 
    ):
        # TODO : 일단 그냥 10만장짜리 data하나 만들어서 그냥 버퍼가지고 테스트 먼저!
        # 기존 buffer 타입 그대로 둬서 add_path는 그대로 쓰고, random_batch만 offline일땐 path에서 불러서, online일땐 섞어서 불러오도록?
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.torchvision_data_loader = torchvision_data_loader
        self.dataloader_buffer_size = dataloader_buffer_size
        
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        
        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            **image_buffer_kwargs
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


    # TODO : random_batch, add_paths(그냥둬도 되려나? 어처피 online 돌아갈때 데이터 들어가는거니까?)
    def random_batch(self, batch_size):
        # TODO :  뺑뺑이 안돌게 변환!
        batch_set_index = np.random.randint(0, self.dataloader_buffer_size, size = 1)
        start = time.time()

        data = next(itertools.islice(torchvision_data_loader, k, None))
        data = self.torchvision_data_loader[batch_set_index]
        print('@@@@@@@@ for debug, torchvision load data batch size : {} time : {}'.format(batch_size, time.time() - start))
        img_m2 = data['img_m2']
        img_m1 = data['img_m1']
        img = data['img']
        img_p1 = data['img_p1']
        state = data['state']
        
        action = data['action']
        reward = data['reward']
        terminal = data['terminal']

        return dict(observations = img,
                    actions = action,
                    rewards = reward ,
                    next_observations = img_p1,
                    terminals = terminal, 
                    )
        
        
    
    
    