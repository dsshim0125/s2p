import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences, reset_w_same_obs = False):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences
        
        self.reset_w_same_obs = reset_w_same_obs

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        if self.reset_w_same_obs:
            for _ in range(self.num_sequences - 1):
                self._state.append(state.copy().astype(np.uint8))
                self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        else:
            for _ in range(self.num_sequences - 1):
                self._state.append(np.zeros(self.state_shape, dtype=np.uint8)) 
                self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Trainer:
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        num_steps=3 * 10 ** 6,
        initial_collection_steps=10 ** 4,
        initial_learning_steps=10 ** 5,
        num_sequences=8,
        eval_interval=10 ** 4,
        num_eval_episodes=5,        
        
        debug = False,
    ):
        # Env to collect samples.
        self.env = env
        self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2 ** 31 - seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Algorithm to learn.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.log_dir = log_dir
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = self.env.action_repeat
        self.num_steps = num_steps
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        self.debug = debug


    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        for step in range(1, self.initial_collection_steps + 1):
            t = self.algo.step(self.env, self.ob, t, step <= self.initial_collection_steps)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        for step in range(self.initial_collection_steps + 1, self.num_steps // self.action_repeat + 1):
            t = self.algo.step(self.env, self.ob, t, False)

            # Update the algorithm.
            self.algo.update_latent(self.writer)
            self.algo.update_sac(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.eval_interval == 0:
                self.evaluate(step_env)
                self.algo.save_model(os.path.join(self.model_dir, f"step{step_env}"))

        # Wait for logging to be finished.
        sleep(10)

    def evaluate(self, step_env):
        mean_return = 0.0

        for i in range(self.num_eval_episodes):
            state = self.env_test.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                state, reward, done, _ = self.env_test.step(action)
                self.ob_test.append(state, action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))



    def only_train_latent_model(self):
        import h5py
        import os
        import torch
        dataset = {}
        env_name= 'cheetah-run-mixed'        
        load_dir = ''
        savedir = load_dir+'/slac'
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        h5py_r = h5py.File(load_dir+'/image_numpy_dataset_stack3_imgsize_100.hdf5', 'r')
        start = time()
        dataset['observations'] = h5py_r['observations'][:]
        dataset['actions'] = h5py_r['actions'][:]
        dataset['rewards'] = h5py_r['rewards'][:]
        dataset['next_observations'] = h5py_r['next_observations'][:]
        dataset['terminals'] = h5py_r['terminals'][:]
        dataset['timeouts'] = h5py_r['timeouts'][:]
        
        dataset['image_observations'] = np.transpose(h5py_r['image_observations'][:], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
        dataset['image_observations_tp1'] = np.transpose(h5py_r['image_observations_tp1'][:], (0, 3,1,2)) #[bs, h, w, c] -> [bs, c,h,w]
        print('load h5py time : ', time()-start)
        assert (dataset['terminals']==0).all(), 'Assume there is no terminal. Only timeouts exists'
        print('terminal indices : ', np.where(dataset['terminals']==1)[0])
        print('timeout indices : ', np.where(dataset['timeouts']==1)[0])
        h5py_r.close()
        del h5py_r
        # raise NotImplementedError('지금 이 format은 trajwise data여야 가능함! trajwise data로 넣던가 아니면 buffer랑 학습하는부분 잘 뜯어보든가!')
        # Time to start training.
        self.start_time = time()

        original_data_size = dataset['observations'].shape[0]
        print('original data size : ', original_data_size)
        state = dataset['image_observations'][0]
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)
        t = 0
        buffer_input_start = time()
        for i in range(original_data_size):            
            if i==original_data_size-1:
                timeout = dataset['timeouts'][i]
                print('last data timeout is : ', timeout)
                if timeout:
                    break

            print('buffer storing : ', i) if i % 500 ==0 else None
            # t+=1
            action = dataset['actions'][i]
            reward = dataset['rewards'][i]
            # done = dataset['terminals'][i]
            
            # double check for episode end
            timeout = dataset['timeouts'][i]
            done = timeout
            # if timeout:
            #     assert t==self.env._max_episode_steps, 't : {} env max episode step : {}'.format(t, self.env._max_episode_steps)
            # if t==self.env._max_episode_steps:
            #     assert timeout, 't : {} env max episode step : {}'.format(t, self.env._max_episode_steps)
            
            mask = False if t == self.env._max_episode_steps else done
            
            
            
            # if timeout:
            #     state = dataset['image_observations_tp1'][i] # means next state at the end of episode
            # else:
            #     state = dataset['image_observations'][i+1] # means next_state 
            
            # timeout이던 말던 tp1쓰면 되잖아?
            state = dataset['image_observations_tp1'][i] # means next state at the end of episode


            self.ob.append(state, action)
            self.algo.buffer.append(action, reward, mask, state, done)
            t+=1
            if done:
                assert timeout                
                t = 0
                if i==original_data_size-1: # when last data but not timeout
                    break
                else:
                    state = dataset['image_observations'][i+1]
                self.ob.reset_episode(state)
                self.algo.buffer.reset_episode(state)

            '''
            # original step fuction
            state, reward, done, _ = env.step(action)
            mask = False if t == env._max_episode_steps else done
            ob.append(state, action)
            self.buffer.append(action, reward, mask, state, done)

            if done:
                t = 0
                state = env.reset()
                ob.reset_episode(state)
                self.buffer.reset_episode(state)
            '''
        
        print('offline data buffer store time : ', time() - buffer_input_start)
        torch.save(self.algo.buffer, savedir+'/buffer.pt')


        # # Time to start training.
        # self.start_time = time()
        # # Episode's timestep.
        # t = 0
        # # Initialize the environment.
        
        # state = self.env.reset()
        # self.ob.reset_episode(state)
        # self.algo.buffer.reset_episode(state)

        # # Collect trajectories using random policy.
        # for step in range(1, self.initial_collection_steps + 1):
        #     t = self.algo.step(self.env, self.ob, t, step <= self.initial_collection_steps)
        #     '''
        #     state, reward, done, _ = env.step(action)
        #     mask = False if t == env._max_episode_steps else done
        #     ob.append(state, action)
        #     self.buffer.append(action, reward, mask, state, done)

        #     if done:
        #         t = 0
        #         state = env.reset()
        #         ob.reset_episode(state)
        #         self.buffer.reset_episode(state)
        #     '''





        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        
        num_latent_train = 300000
        # num_latent_train = 100 # debug
        bar = tqdm(range(num_latent_train))
        for i in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)
            if i % 5000 == 0:
                # self.evaluate(step_env)
                self.algo.save_model(os.path.join(self.model_dir, f"step{i}"))
            

        # Wait for logging to be finished.
        sleep(10)
