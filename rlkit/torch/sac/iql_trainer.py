

import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from rlkit.torch.sac.policies import MakeDeterministic
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
import torch.nn.functional as F
from rlkit.torch.networks import LinearTransform
import time


class IQLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,
            quantile=0.5,
            target_qf1=None,
            target_qf2=None,
            buffer_policy=None,
            z=None,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            policy_update_period=1,
            q_update_period=1,

            reward_transform_class=None,
            reward_transform_kwargs=None,
            terminal_transform_class=None,
            terminal_transform_kwargs=None,

            clip_score=None,
            soft_target_tau=1e-2,
            target_update_period=1,
            beta=1.0,
            image_rl = False,
            encoder_lr = 1e-3,
            critic = None,
            vf_critic = None,
            seperate_vf_encoder = False,
            curl_learning = False,
            curl = None,
            curl_update_period = 1,
            encoder_target_tau = 0.05,
            no_curl_contrastive_learning = False,
            training_start_steps = 0,
            pretrain_contrastive = False,
            add_sac_entropy = False,
            # slac
            slac_representation = False,
            slac_algo = None,
            freeze_slac = False,
            slac_update_period = 1,
            slac_policy_input_type = 'feature_action', #'latent_z'
            
            
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.vf = vf
        self.z = z
        self.buffer_policy = buffer_policy

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.optimizers = {}

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
            
            betas=(0.9, 0.999),
        )
        self.optimizers[self.policy] = self.policy_optimizer

        
        self.image_rl = image_rl
        self.curl_learning = curl_learning
        self.encoder_target_tau = encoder_target_tau
        self.curl_update_period = curl_update_period
        self.no_curl_contrastive_learning = no_curl_contrastive_learning
        self.pretrain_contrastive = pretrain_contrastive
        self.training_start_steps = training_start_steps
        
        self.add_sac_entropy = add_sac_entropy
        
        self.seperate_vf_encoder = seperate_vf_encoder
        

        # SLAC related
        self.slac_representation = slac_representation
        self.slac_algo = slac_algo
        self.freeze_slac = freeze_slac
        self.slac_update_period = slac_update_period
        self.slac_policy_input_type = slac_policy_input_type


        if self.image_rl:                        
            if self.slac_representation:
                self.integrated_optimize = True                
                if self.integrated_optimize:
                    self.critic = critic
                    self.critic_optimizer = optimizer_class(self.critic.parameters(), 
                                                            weight_decay = q_weight_decay, 
                                                            lr=qf_lr,                                                            
                                                            # betas=(0.9, 0.999),
                                                            )
                    self.critic.train(True)            
                


            else:
                self.critic = None                
                self.curl = None

                self.critic_optimizer = optimizer_class(self.critic.parameters(), 
                                                        weight_decay = q_weight_decay, 
                                                        lr=qf_lr,                                                        
                                                        betas=(0.9, 0.999),
                                                        )
                self.vf_critic_optimizer = optimizer_class(self.vf_critic.parameters(), 
                                                        weight_decay = q_weight_decay, 
                                                        lr=qf_lr,                                                        
                                                        betas=(0.9, 0.999),
                                                        )
                self.vf_critic.train(True)

            
            self.policy.train(True)
            

        else:
            self.qf1_optimizer = optimizer_class(
                self.qf1.parameters(),
                weight_decay=q_weight_decay,
                lr=qf_lr,
            )
            self.qf2_optimizer = optimizer_class(
                self.qf2.parameters(),
                weight_decay=q_weight_decay,
                lr=qf_lr,
            )
            self.vf_optimizer = optimizer_class(
                self.vf.parameters(),
                weight_decay=q_weight_decay,
                lr=qf_lr,
            )

        if self.z:
            self.z_optimizer = optimizer_class(
                self.z.parameters(),
                weight_decay=q_weight_decay,
                lr=qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period

        self.reward_transform_class = reward_transform_class or LinearTransform
        self.reward_transform_kwargs = reward_transform_kwargs or dict(m=1, b=0)
        self.terminal_transform_class = terminal_transform_class or LinearTransform
        self.terminal_transform_kwargs = terminal_transform_kwargs or dict(m=1, b=0)
        self.reward_transform = self.reward_transform_class(**self.reward_transform_kwargs)
        self.terminal_transform = self.terminal_transform_class(**self.terminal_transform_kwargs)

        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile

    
    

    def train_from_torch(self, batch, batch_gen=None, train=True, pretrain=False,):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        if self.slac_representation: 
            pass
        else:
            next_obs = batch['next_observations']
        if self.reward_transform: 
            rewards = self.reward_transform(rewards)

        if self.terminal_transform:
            terminals = self.terminal_transform(terminals)
       
       
              
        if self.image_rl:            
            if self.slac_representation:
                # Assume z values are already no grad (The encoder is not trained). obs is [0~1] normalized image pixels [bs, num_seq, c, h,w]
                z, next_z, actions, feature_action, next_feature_action = self.slac_algo.prepare_batch(obs, actions) 
                
                if self.integrated_optimize:
                    q1_pred, q2_pred, target_q1_pred, target_q2_pred, vf_pred = self.critic(z, actions)
                    next_vf_pred = self.critic(next_z)
                    target_vf_pred = next_vf_pred.detach()
                    
                    
                    q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_vf_pred
                    q_target = q_target.detach()
                    
                    
                    qf1_loss = self.qf_criterion(q1_pred, q_target)
                    qf2_loss = self.qf_criterion(q2_pred, q_target)

                    """
                    VF Loss
                    """
                    
                    q_pred = torch.min(
                        target_q1_pred,
                        target_q2_pred,
                    ).detach()

                    
                    vf_err = vf_pred - q_pred
                    vf_sign = (vf_err > 0).float()
                    vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
                    vf_loss = (vf_weight * (vf_err ** 2)).mean()
                
                """
                Policy and Alpha Loss
                """
                
                if self.slac_policy_input_type=='feature_action':                    
                    policy_input = feature_action
                elif self.slac_policy_input_type=='latent_z':
                    policy_input = z

                dist = self.policy(policy_input) 




        else: # state rl
            """
            QF Loss
            """
            q1_pred = self.qf1(obs, actions) 
            q2_pred = self.qf2(obs, actions)
            target_vf_pred = self.vf(next_obs).detach()

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_vf_pred
            q_target = q_target.detach()
            qf1_loss = self.qf_criterion(q1_pred, q_target)
            qf2_loss = self.qf_criterion(q2_pred, q_target)
            """
            VF Loss
            """
            q_pred = torch.min(
                self.target_qf1(obs, actions),
                self.target_qf2(obs, actions),
            ).detach()
            vf_pred = self.vf(obs)
            vf_err = vf_pred - q_pred
            vf_sign = (vf_err > 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
            vf_loss = (vf_weight * (vf_err ** 2)).mean()
            
            """
            Policy and Alpha Loss
            """
            dist = self.policy(obs)


        """
        Policy Loss
        """
        policy_logpp = dist.log_prob(actions)

        adv = q_pred - vf_pred
        exp_adv = torch.exp(adv / self.beta)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        weights = exp_adv[:, 0].detach()
        policy_loss = (-policy_logpp * weights).mean()

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0 and self._n_train_steps_total >=self.training_start_steps:
            if self.image_rl :                
                if self.slac_representation:
                    if self.integrated_optimize:
                        self.critic_optimizer.zero_grad()
                        critic_loss = qf1_loss+qf2_loss+vf_loss
                        critic_loss.backward()
                        self.critic_optimizer.step()
                    
            else:
                self.qf1_optimizer.zero_grad()
                qf1_loss.backward()
                self.qf1_optimizer.step()

                self.qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self.qf2_optimizer.step()

                self.vf_optimizer.zero_grad()
                vf_loss.backward()
                self.vf_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0 and self._n_train_steps_total >=self.training_start_steps:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        
        if self._n_train_steps_total % self.slac_update_period == 0 and self.slac_representation and (not self.curl_learning) and (not self.freeze_slac):
            print('update slac latent model!') if self._n_train_steps_total%1000==0 else None
            loss_kld, loss_image, loss_reward = self.slac_algo.update_latent(writer = None) # self.writer


        if self.image_rl and self.curl_learning:            
            assert (self.policy.encoder.convs[0].weight.detach().cpu().numpy()==self.vf_critic.encoder.convs[0].weight.detach().cpu().numpy()).all()
            assert (self.critic.encoder.convs[0].weight.detach().cpu().numpy()==self.vf_critic.encoder.convs[0].weight.detach().cpu().numpy()).all()

        """
        Soft Updates
        """
        
        if self._n_train_steps_total % self.target_update_period == 0 :
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )
            if self.image_rl and self.curl_learning and (not self.no_curl_contrastive_learning) and (not self.pretrain_contrastive):
                ptu.soft_update_from_to(
                    self.critic.encoder, self.critic.encoder_target, self.encoder_target_tau
                )
            
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'terminals',
                ptu.get_numpy(terminals),
            ))
            if self.slac_representation:
                self.eval_statistics['replay_buffer_len'] = len(self.replay_buffer)
            else:
                self.eval_statistics['replay_buffer_len'] = self.replay_buffer._size
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            self.eval_statistics.update(policy_statistics)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Weights',
                ptu.get_numpy(weights),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantage Score',
                ptu.get_numpy(adv),
            ))

            self.eval_statistics.update(create_stats_ordered_dict(
                'V1 Predictions',
                ptu.get_numpy(vf_pred),
            ))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            
            
            if self.slac_representation and (not self.freeze_slac):                
                self.eval_statistics['SLAC Loss kld'] = np.mean(ptu.get_numpy(loss_kld))
                self.eval_statistics['SLAC Loss image'] = np.mean(ptu.get_numpy(loss_image))
                self.eval_statistics['SLAC Loss reward'] = np.mean(ptu.get_numpy(loss_reward))

                
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf,
        ]
        if self.image_rl:            
            if self.slac_representation :
                if self.integrated_optimize:
                    nets.append(self.critic)
                
                if not self.freeze_slac:
                    nets.append(self.slac_algo.latent)
                
                
        
        return nets

    def get_snapshot(self):
        if self.image_rl:            
            if self.slac_representation:
                snapshot = dict(
                    policy=self.policy,
                    qf1=self.qf1,
                    qf2=self.qf2,
                    target_qf1=self.target_qf1,
                    target_qf2=self.target_qf2,
                    vf=self.vf,
                    critic = self.critic,
                    policy_optimizer = self.policy_optimizer,
                    critic_optimizer = self.critic_optimizer,                    
                    slac_algo_latent = self.slac_algo.latent,
                    slac_algo_latent_optimizer = self.slac_algo.optim_latent,
                )
            return snapshot
        else:
            return dict(
                policy=self.policy,
                qf1=self.qf1,
                qf2=self.qf2,
                target_qf1=self.target_qf1,
                target_qf2=self.target_qf2,
                vf=self.vf,
            )
