import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import time
from rlkit.torch.core import np_to_pytorch_batch
import torch
from rlkit.torch import pytorch_util as ptu


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0, # negative epochs are offline, positive epochs are online
            image_rl = False, 
            num_pretrains =0, 
            # for SLAC
            slac_representation = False,
            replay_buffer_gen = None,
            # for RAD
            rad_aug = False,
            
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        
        self.image_rl = image_rl 
        self.num_pretrains = num_pretrains
        # for SLAC
        self.slac_representation = slac_representation
        self.replay_buffer_gen = replay_buffer_gen
        # for RAD
        self.rad_aug = rad_aug


    def policy_fn(self, obs):
        """
        Used when sampling actions from the policy and doing max Q-learning
        """
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            state = ptu.from_numpy(obs.reshape(1, -1)).repeat(self.num_actions_sample, 1)
            action, _, _, _, _, _, _, _  = self.trainer.policy(state)
            q1 = self.trainer.qf1(state, action)
            ind = q1.max(0)[1]
        return ptu.get_numpy(action[ind]).flatten()

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0: 
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl: 
                if self.slac_representation: 
                    raise NotImplementedError
                else:
                    self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')


        for _ in range(self.num_train_loops_per_epoch):
            # if not self.offline_rl:
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop if not self.offline_rl else 1, # meaningless path when offline training
                discard_incomplete_paths=False,
            )
            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                if self.slac_representation: 
                    print('WARNING: Currently online rl with SLAC representation is not developed!!!!')
                    pass
                else:
                    self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for idx in range(self.num_trains_per_train_loop): 
                
                if self.replay_buffer_gen is not None:
                    train_data = self.replay_buffer.random_batch(int(self.batch_size/2))                
                    train_data_gen = self.replay_buffer.random_batch(int(self.batch_size/2))
                    self.trainer.train(train_data, np_batch_gen=train_data_gen, slac_representation = self.slac_representation, rad_aug = self.rad_aug)
                else:
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data, slac_representation = self.slac_representation, rad_aug = self.rad_aug)    

                
                
            gt.stamp('training', unique=False)
            self.training_mode(False)
