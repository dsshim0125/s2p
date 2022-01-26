from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd
import time

class CQLTrainer(TorchTrainer):
    def __init__(
            self, 
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs=2,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,
            
            image_rl = False,
            encoder_lr = 1e-3,
            critic = None,
            vf_critic = None,
            vf = None,
            # seperate_vf_encoder = False,
            curl_learning = False,
            curl = None,
            curl_update_period = 1,
            encoder_target_tau = 0.05,
            no_curl_contrastive_learning = False,
            training_start_steps = 0,
            pretrain_contrastive = False,
            # for SLAC
            slac_representation = False,
            slac_algo = None,
            freeze_slac = False,
            slac_update_period = 1,
            slac_policy_input_type = 'feature_action', #'latent_z'

            **kwargs
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item() 
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )
        
        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class(
                [self.log_alpha_prime],
                lr=qf_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        
        self.image_rl = image_rl
        self.curl_learning = curl_learning
        self.encoder_target_tau = encoder_target_tau
        self.curl_update_period = curl_update_period
        self.no_curl_contrastive_learning = no_curl_contrastive_learning
        self.pretrain_contrastive = pretrain_contrastive
        self.training_start_steps = training_start_steps
        self.augment_rollout_data = False
                
        # SLAC related
        self.slac_representation = slac_representation
        self.slac_algo = slac_algo
        self.freeze_slac = freeze_slac
        self.slac_update_period = slac_update_period
        self.slac_policy_input_type = slac_policy_input_type

        if self.image_rl:            
            
            if self.slac_representation:
                self.critic = critic
             
            else:
                self.critic = None                
                self.curl = None

            self.critic_optimizer = optimizer_class(self.critic.parameters(), 
                                                    # weight_decay = q_weight_decay, 
                                                    lr=qf_lr,
                                                    
                                                    betas=(0.9, 0.999),
                                                    )
            

            self.critic.train(True)            
            self.policy.train(True)
            
        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1
        
        self.num_qs = num_qs

        ## min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        # For implementation on the 
        self.discrete = False



    
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int (action_shape / obs_shape)
        
        if self.image_rl and self.curl_learning:
            # obs : [bs, c*stack, h, w]
            # [bs, num_act, c*stack, h, w] -> [bs*num_act, c*stack, h, w]
            obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1, 1, 1).view(obs.shape[0] * num_repeat, obs.shape[1], obs.shape[2], obs.shape[3])            
            q1_preds, q2_preds, _, _ = network(obs_temp, actions)
            q1_preds = q1_preds.view(obs.shape[0], num_repeat, 1)
            q2_preds = q2_preds.view(obs.shape[0], num_repeat, 1)
            return q1_preds, q2_preds
        else:
            # obs : [bs, dim]
            # [bs, num_act, dim] -> [bs*num_act, dim]
            obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
            preds = network(obs_temp, actions)
            preds = preds.view(obs.shape[0], num_repeat, 1)
            return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        if self.image_rl and self.curl_learning:
            # obs : [bs, c*stack, h, w]
            # [bs, num_act, c*stack, h, w] -> [bs*num_act, c*stack, h, w]
            obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1, 1, 1).view(obs.shape[0] * num_actions, obs.shape[1], obs.shape[2], obs.shape[3])
            new_obs_actions, _, _, new_obs_log_pi, *_ = network(
                obs_temp, detach_encoder = True, reparameterize=True, return_log_prob=True,
            )
        else:
            # obs : [bs, dim]
            # [bs, num_act, dim] -> [bs*num_act, dim]
            obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
            new_obs_actions, _, _, new_obs_log_pi, *_ = network(
                obs_temp, reparameterize=True, return_log_prob=True,
            )
        if not self.discrete:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def train_from_torch(self, batch, batch_gen = None):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        if self.slac_representation: 
            pass
        else:
            next_obs = batch['next_observations']
        
        batch_size = batch['observations'].shape[0]

        if self.image_rl:            
            if self.slac_representation:
                gradient_accumulate = False
                
                if gradient_accumulate:
                    raise NotImplementedError('Currently not implemented for slac representation with grad accumulate!')
                else:
                    # Assume z values are already no grad (The encoder is not trained). obs is [0~1] normalized image pixels [bs, num_seq, c, h,w]
                    z, next_z, actions, feature_action, next_feature_action = self.slac_algo.prepare_batch(obs, actions) 
                    if self.slac_policy_input_type=='feature_action':                    
                        policy_input = feature_action
                        policy_next_input = next_feature_action
                    elif self.slac_policy_input_type=='latent_z':
                        policy_input = z
                        policy_next_input = next_z

                    new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(policy_input, return_log_prob = True, reparameterize=True) 
                    if self.use_automatic_entropy_tuning:
                        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        alpha = self.log_alpha.exp()
                    else:
                        alpha_loss = 0
                        alpha = 1
                    
                    if self.num_qs == 1:
                        q_new_actions = self.qf1(obs, new_obs_actions)
                        raise NotImplementedError
                    else:
                        q1_pred_new_actions, q2_pred_new_actions, _, _, _ = self.critic(z, new_obs_actions)
                        q_new_actions = torch.min(q1_pred_new_actions, q2_pred_new_actions)


                    policy_loss = (alpha*log_pi - q_new_actions).mean()
                    if self._current_epoch < self.policy_eval_start:
                        # print('Currently Debugging : current epoch : ', self._current_epoch)
                        """
                        For the initial few epochs, try doing behaivoral cloning, if needed
                        conventionally, there's not much difference in performance with having 20k 
                        gradient steps here, or not having it
                        """
                        # def logprob(self, action, mean, std):
                        policy_log_prob = self.policy.logprob(actions, policy_mean, policy_log_std.exp())
                        policy_loss = (alpha * log_pi - policy_log_prob).mean()
                    
                    
                    self._num_policy_update_steps += 1
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=False)
                    self.policy_optimizer.step()

                    """
                    QF Loss
                    """
                    q1_pred, q2_pred, _, _, _ = self.critic(z,actions)
                    
                    new_next_actions, _, _, new_log_pi, *_ = self.policy(
                        policy_next_input, reparameterize=True, return_log_prob=True,
                    )
                    
                    

                    if not self.max_q_backup:
                        if self.num_qs == 1:
                            # target_q_values = self.target_qf1(next_obs, new_next_actions)
                            raise NotImplementedError
                        else:
                            
                            _, _, target_q1_pred, target_q2_pred, _ = self.critic(next_z, new_next_actions)
                            target_q_values = torch.min(target_q1_pred, target_q2_pred)

                        if not self.deterministic_backup:
                            target_q_values = target_q_values - alpha * new_log_pi


                    if self.max_q_backup:
                        raise NotImplementedError
                        """when using max q backup"""
                        next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
                        target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
                        target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
                        target_q_values = torch.min(target_qf1_values, target_qf2_values)

                    q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
                    q_target = q_target.detach()
                        
                    qf1_loss = self.qf_criterion(q1_pred, q_target)
                    if self.num_qs > 1:
                        qf2_loss = self.qf_criterion(q2_pred, q_target)

                ## add CQL
                

                start = time.time()
                
                
                if gradient_accumulate:
                    raise NotImplementedError('Currently not implemented for slac representation with grad accumulate!')
                    
                else: # no grad accumulate
                    assert self.slac_representation
                    ## add CQL
                    random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(ptu.device)
                    curr_actions_tensor, curr_log_pis = self._get_policy_actions(policy_input, num_actions=self.num_random, network=self.policy)
                    new_curr_actions_tensor, new_log_pis = self._get_policy_actions(policy_next_input, num_actions=self.num_random, network=self.policy)
                    q1_rand = self._get_tensor_values(z, random_actions_tensor, network=self.qf1)
                    q2_rand = self._get_tensor_values(z, random_actions_tensor, network=self.qf2)
                    q1_curr_actions = self._get_tensor_values(z, curr_actions_tensor, network=self.qf1)
                    q2_curr_actions = self._get_tensor_values(z, curr_actions_tensor, network=self.qf2)
                    q1_next_actions = self._get_tensor_values(z, new_curr_actions_tensor, network=self.qf1)
                    q2_next_actions = self._get_tensor_values(z, new_curr_actions_tensor, network=self.qf2)

                    cat_q1 = torch.cat(
                        [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
                    )
                    cat_q2 = torch.cat(
                        [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
                    )
                    std_q1 = torch.std(cat_q1, dim=1)
                    std_q2 = torch.std(cat_q2, dim=1)

                    if self.min_q_version == 3:
                        # importance sammpled version
                        random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
                        cat_q1 = torch.cat(
                            [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
                        )
                        cat_q2 = torch.cat(
                            [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
                        )
                        
                    min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                    min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                                
                    """Subtract the log likelihood of data"""
                    min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
                    min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
                    
                    if self.with_lagrange:
                        alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                        min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                        min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

                        self.alpha_prime_optimizer.zero_grad()
                        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
                        alpha_prime_loss.backward(retain_graph=True)
                        self.alpha_prime_optimizer.step()

                    qf1_loss = qf1_loss + min_qf1_loss
                    qf2_loss = qf2_loss + min_qf2_loss
                    """
                    Update networks
                    """
                    # Update the Q-functions iff 
                    self._num_q_update_steps += 1
                    self.critic_optimizer.zero_grad()
                    critic_loss = qf1_loss+qf2_loss
                    critic_loss.backward(retain_graph=True)
                    self.critic_optimizer.step()

                    

                if self._n_train_steps_total % self.curl_update_period == 0 and self.curl_learning and (not self.no_curl_contrastive_learning) and (not self.pretrain_contrastive):
                    print('update curl cpc (contrastive)!') if self._n_train_steps_total%1000==0 else None
                    obs_anchor, obs_pos = batch["obs_anchor"], batch["obs_pos"]
                    curl_loss = self.update_curl(obs_anchor, obs_pos) #,cpc_kwargs, L, step)
                
                if self._n_train_steps_total % self.slac_update_period == 0 and self.slac_representation and (not self.curl_learning) and (not self.freeze_slac):
                    print('update slac latent model!') if self._n_train_steps_total%1000==0 else None
                    loss_kld, loss_image, loss_reward = self.slac_algo.update_latent(writer = None) # self.writer

                if self.image_rl and self.curl_learning:
                    
                    assert (self.policy.encoder.convs[0].weight.detach().cpu().numpy()==self.critic.encoder.convs[0].weight.detach().cpu().numpy()).all()
                        

        else: # state rl
            """
            Policy and Alpha Loss
            """
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )
            
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1

            if self.num_qs == 1:
                q_new_actions = self.qf1(obs, new_obs_actions)
            else:
                q_new_actions = torch.min(
                    self.qf1(obs, new_obs_actions),
                    self.qf2(obs, new_obs_actions),
                )

            policy_loss = (alpha*log_pi - q_new_actions).mean()

            if self._current_epoch < self.policy_eval_start:
                
                """
                For the initial few epochs, try doing behaivoral cloning, if needed
                conventionally, there's not much difference in performance with having 20k 
                gradient steps here, or not having it
                """
                policy_log_prob = self.policy.logprob(actions, policy_mean, policy_log_std.exp())
                # policy_log_prob = self.policy.log_prob(obs, actions)
                policy_loss = (alpha * log_pi - policy_log_prob).mean()
            

            self._num_policy_update_steps += 1
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            self.policy_optimizer.step()


            """
            QF Loss
            """
            q1_pred = self.qf1(obs, actions)
            if self.num_qs > 1:
                q2_pred = self.qf2(obs, actions)
            
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )

            if not self.max_q_backup:
                if self.num_qs == 1:
                    target_q_values = self.target_qf1(next_obs, new_next_actions)
                else:
                    target_q_values = torch.min(
                        self.target_qf1(next_obs, new_next_actions),
                        self.target_qf2(next_obs, new_next_actions),
                    )
                
                if not self.deterministic_backup:
                    target_q_values = target_q_values - alpha * new_log_pi
            
            if self.max_q_backup:
                """when using max q backup"""
                next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
                target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
                target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
                target_q_values = torch.min(target_qf1_values, target_qf2_values)

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
            q_target = q_target.detach()
                
            qf1_loss = self.qf_criterion(q1_pred, q_target)
            if self.num_qs > 1:
                qf2_loss = self.qf_criterion(q2_pred, q_target)

            ## add CQL
            random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1).to(ptu.device)
            curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
            new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)
            q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
            q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
            q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
            q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
            q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
            q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

            cat_q1 = torch.cat(
                [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
            )
            cat_q2 = torch.cat(
                [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
            )
            std_q1 = torch.std(cat_q1, dim=1)
            std_q2 = torch.std(cat_q2, dim=1)

            if self.min_q_version == 3:
                # importance sammpled version
                random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
                cat_q1 = torch.cat(
                    [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
                )
                
            min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
                        
            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight
            
            if self.with_lagrange:
                alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
                min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            qf1_loss = qf1_loss + min_qf1_loss
            qf2_loss = qf2_loss + min_qf2_loss

            """
            Update networks
            """
            # Update the Q-functions iff 
            self._num_q_update_steps += 1
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=True)
            self.qf1_optimizer.step()

            if self.num_qs > 1:
                self.qf2_optimizer.zero_grad()
                qf2_loss.backward(retain_graph=True)
                self.qf2_optimizer.step()

        
        """
        Soft Updates
        """
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        if self.num_qs > 1:
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
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))

            if not self.discrete:
                self.eval_statistics['Std QF1 values'] = np.mean(ptu.get_numpy(std_q1))
                self.eval_statistics['Std QF2 values'] = np.mean(ptu.get_numpy(std_q2))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 in-distribution values',
                    ptu.get_numpy(q1_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 in-distribution values',
                    ptu.get_numpy(q2_curr_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 random values',
                    ptu.get_numpy(q1_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 random values',
                    ptu.get_numpy(q2_rand),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF1 next_actions values',
                    ptu.get_numpy(q1_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'QF2 next_actions values',
                    ptu.get_numpy(q2_next_actions),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'actions', 
                    ptu.get_numpy(actions)
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'rewards',
                    ptu.get_numpy(rewards)
                ))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            if not self.discrete:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            
            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()
            
            
            if self.slac_representation and (not self.freeze_slac):                
                self.eval_statistics['SLAC Loss kld'] = np.mean(ptu.get_numpy(loss_kld))
                self.eval_statistics['SLAC Loss image'] = np.mean(ptu.get_numpy(loss_image))
                self.eval_statistics['SLAC Loss reward'] = np.mean(ptu.get_numpy(loss_reward))


        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats
        # return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        if self.image_rl:            
            if self.slac_representation:
                base_list.append(self.critic)
                if not self.freeze_slac:
                    base_list.append(self.slac_algo.latent)
        
        return base_list

    def get_snapshot(self):
        if self.image_rl:            
            if self.slac_representation:
                snapshot = dict(
                    policy=self.policy,
                    qf1=self.qf1,
                    qf2=self.qf2,
                    target_qf1=self.target_qf1,
                    target_qf2=self.target_qf2,                    
                    critic = self.critic,                    
                    policy_optimizer = self.policy_optimizer,
                    critic_optimizer = self.critic_optimizer,                                    
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
            )

