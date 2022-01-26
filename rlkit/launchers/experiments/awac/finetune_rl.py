from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    
)

from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.visualization.video import VideoSaveFunction

import torch
import numpy as np

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy

import os.path as osp
from rlkit.core import logger
import pickle

from rlkit.envs.make_env import make

from rlkit.torch.networks import LinearTransform

import random

# added for RAD
from examples.iql import data_augs as rad



def split_into_trajectories(replay_buffer): 
    dones_float = np.zeros_like(replay_buffer._rewards)

    for i in range(replay_buffer._size):
        delta = replay_buffer._observations[i + 1, :] - replay_buffer._next_obs[i, :]
        norm = np.linalg.norm(delta)
        if norm > 1e-6 or replay_buffer._terminals[i]: # or replay_buffer._timeouts[i]
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    trajs = [[]]

    for i in range(replay_buffer._size):
        trajs[-1].append((replay_buffer._observations[i],
            replay_buffer._actions[i],
            replay_buffer._rewards[i],
            replay_buffer._terminals[i],
            replay_buffer._next_obs[i]))
        if dones_float[i] == 1.0 and i + 1 < replay_buffer._size:
            trajs.append([])

    return trajs


def get_normalization(replay_buffer):
    trajs = split_into_trajectories(replay_buffer)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    reward_range = compute_returns(trajs[-1]) - compute_returns(trajs[0])
    
    m = 1000.0 / reward_range 
    
    return LinearTransform(m=float(m), b=0)

def experiment(variant):
    

    normalize_env = variant.get('normalize_env', True)
    env_id = variant.get('env_id', None)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', {})
    

    if variant.get('image_rl', False):
        if variant.get('slac_representation'):
            expl_env = make(env_id, env_class, env_kwargs, normalize_env)
            eval_env = make(env_id, env_class, env_kwargs, normalize_env)
            
        state_env = make(env_id, env_class, env_kwargs, normalize_env)
    else:
        framestack = variant.get('frame_stack', 1)
        if framestack > 1:       
            from examples.iql.custom_gym_to_multi_env import StateStack            
            expl_env = StateStack(make(env_id, env_class, env_kwargs, normalize_env), k = framestack, state_type='qpos', env_id = variant.get('env_id'))
            eval_env = StateStack(make(env_id, env_class, env_kwargs, normalize_env), k = framestack, state_type='qpos', env_id = variant.get('env_id'))
            
        else:
            expl_env = make(env_id, env_class, env_kwargs, normalize_env)
            eval_env = make(env_id, env_class, env_kwargs, normalize_env)
    if variant.get('is_dmc', False):
        assert variant.get('max_path_length') == expl_env._max_episode_steps

    seed = int(variant["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    eval_env.seed(seed)
    expl_env.seed(seed)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    
    slac_algo = None

    if variant.get('image_rl', False):
        from examples.iql.custom_networks import Qfunction, Vfunction, TanhGaussianPolicyWithEncoder        
        if variant.get('slac_representation', False):                        
            from rlkit.torch.slac.algo import SlacAlgorithm
            num_sequences = 8
            
            slac_algo = SlacAlgorithm(
                state_shape=expl_env.observation_space.shape,
                action_shape=expl_env.action_space.shape,
                action_repeat=env_kwargs['frame_skip'], #4, # 딱히 안쓰이는듯
                device=ptu.device,
                seed=seed,
                use_seperate_buffer = variant.get('seperate_buffer', False),
                **variant.get('slac_algo_kwargs'),
                
            )
            slac_latent_model_load_dir = variant.get('slac_latent_model_load_dir', None)
            if slac_latent_model_load_dir is not None:
                latent_state_dict = torch.load(slac_latent_model_load_dir+'/latent.pth', map_location=ptu.device)
                slac_algo.latent.load_state_dict(latent_state_dict)
                print('load latent model state dict!')

            
            pixel_encoder_qf1 = pixel_encoder_qf2 = pixel_encoder_target_qf1 = pixel_encoder_target_qf2 = pixel_encoder_vf = None
            pixel_encoder_policy = None

            

        qf_kwargs = variant.get("qf_kwargs", {})
        if variant.get('slac_representation', False):
            slac_algo_kwargs = variant.get('slac_algo_kwargs')
            slac_feature_dim = slac_algo_kwargs.get('feature_dim') # 256            
            z1_dim = slac_algo_kwargs.get('z1_dim') # 32
            z2_dim = slac_algo_kwargs.get('z2_dim') # 256
            feature_dim = z1_dim + z2_dim

        

        qf1 = Qfunction(
            input_size=feature_dim + action_dim,
            output_size=1,
            encoder = pixel_encoder_qf1,
            **qf_kwargs
        )
        qf2 = Qfunction(
            input_size=feature_dim + action_dim,
            output_size=1,
            encoder = pixel_encoder_qf2,
            **qf_kwargs
        )
        
        target_qf1 = Qfunction(
            input_size=feature_dim + action_dim,
            output_size=1,
            encoder= pixel_encoder_target_qf1,
            **qf_kwargs
        )
        target_qf2 = Qfunction(
            input_size=feature_dim + action_dim,
            output_size=1,
            encoder= pixel_encoder_target_qf2,
            **qf_kwargs
        )

        vf_kwargs = variant.get("vf_kwargs", {})
        vf = Vfunction(
            input_size=feature_dim,
            output_size=1,
            encoder= pixel_encoder_vf,
            **vf_kwargs
        )
        
        policy_class = variant.get("policy_class", TanhGaussianPolicyWithEncoder)
        policy_kwargs = variant['policy_kwargs']
        policy_input_dim = num_sequences * slac_feature_dim + (num_sequences - 1) * action_dim if variant.get('slac_policy_input_type')=='feature_action' else feature_dim
        policy = policy_class(
            obs_dim=policy_input_dim, # feature_dim,
            action_dim=action_dim,
            encoder = pixel_encoder_policy,
            **policy_kwargs,
        )
        if variant.get('algo_type') in ['bear']:            
            if variant.get('slac_representation', False):
                from rlkit.torch.sac.policies.vae_policy import VAEPolicy                
                vae_policy = VAEPolicy(obs_dim=feature_dim,
                                    action_dim=action_dim,
                                    hidden_sizes=[750, 750],
                                    latent_dim=action_dim * 2,                                    
                                    )


        
        if variant.get('slac_representation', False):
            from examples.iql.custom_networks import CriticSLAC
            critic = CriticSLAC(qf1, qf2, target_qf1, target_qf2, vf = vf)
            vf_critic = None
            curl = None

        else:
            critic = None
            vf_critic = None
            curl = None


    else: # state rl
        critic = None
        vf_critic = None
        curl = None
        
        qf_kwargs = variant.get("qf_kwargs", {})
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
        target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )

        vf_kwargs = variant.get("vf_kwargs", dict(hidden_sizes=[256, 256, ],))
        vf = ConcatMlp(
            input_size=obs_dim,
            output_size=1,
            **vf_kwargs
        )

        policy_class = variant.get("policy_class", TanhGaussianPolicy)
        policy_kwargs = variant['policy_kwargs']
        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        curl_crop = variant.get('curl_learning', False), 
        curl_crop_output_size = variant.get('curl_crop_image_size', 84),
        slac_algo = slac_algo, 
        slac_policy_input_type = variant.get('slac_policy_input_type', None),
        slac_obs_reset_w_same_obs = variant.get('slac_obs_reset_w_same_obs', False),
        generalization_test = variant.get('generalization_test', None),
    )

    expl_policy = policy
    exploration_kwargs =  variant.get('exploration_kwargs', {})
    if exploration_kwargs:
        if exploration_kwargs.get("deterministic_exploration", False):
            expl_policy = MakeDeterministic(policy)

        exploration_strategy = exploration_kwargs.get("strategy", None)
        if exploration_strategy is None:
            pass
        elif exploration_strategy == 'ou':
            es = OUStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        elif exploration_strategy == 'gauss_eps':
            es = GaussianAndEpsilonStrategy(
                action_space=expl_env.action_space,
                max_sigma=exploration_kwargs['noise'],
                min_sigma=exploration_kwargs['noise'],  # constant sigma
                epsilon=0,
            )
            expl_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=expl_policy,
            )
        else:
            error

    if variant.get('image_rl', False):
        
        # state buffer kwargs to get reward normalization range
        replay_buffer_kwargs = dict(
            max_replay_buffer_size=variant['replay_buffer_size'],
            env=state_env,
        )
        rad_aug_type = variant.get('rad_aug_type', None)
        
        if rad_aug_type is None:
            aug_funcs = {}
        
        image_replay_buffer_kwargs = dict(
            max_replay_buffer_size=variant['image_replay_buffer_size'],
            env=expl_env,
            image_buffer=True, 
            memory_efficient_way = variant.get('memory_efficient_way'),
            image_obs_shape = variant.get('image_obs_shape'),
            curl = variant.get('curl_learning', False),
            crop_image_size = variant.get('curl_crop_image_size', 84),
            # added for RAD
            rad_aug = variant.get('rad_aug', False),            
            aug_funcs = aug_funcs,
        )
        
        if variant.get('slac_representation', False):
            image_replay_buffer = slac_algo.buffer
            if variant.get('seperate_buffer', False):                
                image_replay_buffer_gen = slac_algo.buffer_gen
                algorithm_replay_buffer_gen = image_replay_buffer_gen
            else:
                image_replay_buffer_gen = None
                algorithm_replay_buffer_gen = None
        
        algorithm_replay_buffer = image_replay_buffer

    else:
        replay_buffer_kwargs = dict(
            max_replay_buffer_size=variant['replay_buffer_size'],
            env=expl_env,
        )
    if variant.get('slac_representation', False):
        pass
    else:
        replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
            **replay_buffer_kwargs,
        )
        
        demo_train_buffer = EnvReplayBuffer(
            **replay_buffer_kwargs,
        )
        demo_test_buffer = EnvReplayBuffer(
            **replay_buffer_kwargs,
        )
    

    trainer_class = variant.get("trainer_class")
    if variant.get('algo_type') in ['iql', 'cql']:
        trainer = trainer_class(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,        
            critic = critic,
            vf=vf,
            vf_critic = vf_critic,
            curl = curl,        
            curl_learning = variant.get('curl_learning'),        
            seperate_vf_encoder = variant.get('seperate_vf_encoder'),
            slac_algo = slac_algo,
            **variant['trainer_kwargs']
        )
    elif variant.get('algo_type') in ['bear']:
        trainer = trainer_class(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,        
            vae=vae_policy,
            critic = critic,            
            curl = curl,        
            curl_learning = variant.get('curl_learning'),        
            seperate_vf_encoder = variant.get('seperate_vf_encoder'),
            slac_algo = slac_algo,
            **variant['trainer_kwargs']
        )
    elif variant.get('algo_type') in ['bc']:
        trainer = trainer_class(
            env=eval_env,
            policy=policy,                    
            curl = curl,        
            curl_learning = variant.get('curl_learning'),        
            slac_algo = slac_algo,
            **variant['trainer_kwargs']
        )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
        curl_crop = variant.get('curl_learning', False), 
        curl_crop_output_size = variant.get('curl_crop_image_size', 84),
        slac_algo = slac_algo, 
        slac_policy_input_type = variant.get('slac_policy_input_type', None),
        slac_obs_reset_w_same_obs = variant.get('slac_obs_reset_w_same_obs', False),
        generalization_test = variant.get('generalization_test', None),
    )
    
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=algorithm_replay_buffer if (variant.get('image_rl', False) or variant.get('state_rl_100k_debug', False)) else replay_buffer, 
        replay_buffer_gen = algorithm_replay_buffer_gen if variant.get('seperate_buffer', False) else None,
        max_path_length=variant['max_path_length'],
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)



    if variant.get("save_video", False):
        video_eval_logdir = variant['logdir'] +'/video_eval'
        video_expl_logdir = variant['logdir'] +'/video_expl'
        import os
        if not os.path.exists(video_eval_logdir):
            os.makedirs(video_eval_logdir)
        if not os.path.exists(video_expl_logdir):
            os.makedirs(video_expl_logdir)
        
        # video_eval_env = eval_env
        video_eval_env = eval_env 
        video_expl_env = expl_env 
        render_kwargs = {'mode' : 'rgb_array', 'width' : 256, 'height' : 256}
        if not variant.get('is_dmc', False):
            render_kwargs.update({ 'camera_name' : 'topview'})
        video_eval_func = VideoSaveFunction(savedir = video_eval_logdir, 
                                            deterministic_policy=True,
                                            env = video_eval_env, #eval_env, 
                                            image_rl = variant.get('image_rl'), 
                                            max_path_length = variant.get('video_eval_max_path_length'),
                                            curl_crop = variant.get('curl_learning', False), 
                                            curl_crop_output_size = variant.get('curl_crop_image_size', 84),
                                            render_kwargs = render_kwargs, # only for state rl
                                            env_name = env_id,
                                            # for SLAC
                                            slac_algo = slac_algo, 
                                            slac_policy_input_type = variant.get('slac_policy_input_type', None),
                                            slac_obs_reset_w_same_obs = variant.get('slac_obs_reset_w_same_obs', False),
                                            generalization_test = variant.get('generalization_test', None),
                                            )
        video_expl_func = VideoSaveFunction(savedir = video_expl_logdir,
                                            deterministic_policy=False,
                                            env = video_expl_env, #expl_env, 
                                            image_rl = variant.get('image_rl'), 
                                            max_path_length = variant.get('video_expl_max_path_length'),
                                            curl_crop = variant.get('curl_learning', False), 
                                            curl_crop_output_size = variant.get('curl_crop_image_size', 84),
                                            render_kwargs = render_kwargs,
                                            env_name = env_id,
                                            slac_algo = slac_algo, 
                                            slac_policy_input_type = variant.get('slac_policy_input_type', None),
                                            slac_obs_reset_w_same_obs = variant.get('slac_obs_reset_w_same_obs', False),
                                            generalization_test = variant.get('generalization_test', None),
                                            )
        
        algorithm.post_epoch_funcs.append(video_eval_func)
        algorithm.post_epoch_funcs.append(video_expl_func)

    if variant.get('save_paths', False):
        algorithm.post_train_funcs.append(save_paths)
    if variant.get('load_demos', False):
        path_loader_class = variant.get('path_loader_class', MDPPathLoader)
        path_loader = path_loader_class(trainer,
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
            **path_loader_kwargs
        )
        path_loader.load_demos()
    if variant.get('load_env_dataset_demos', False):
        if variant.get('is_dmc', False) and variant.get('image_rl', False):                        
            pass
        else:
            path_loader_class = variant.get('path_loader_class', HDF5PathLoader)
            path_loader = path_loader_class(trainer,
                replay_buffer=replay_buffer,
                demo_train_buffer=demo_train_buffer,
                demo_test_buffer=demo_test_buffer,
                **path_loader_kwargs
            )
            if not variant.get('is_dmc', False):
                import d4rl
        
        import h5py
        import time
        from PIL import Image
        if variant.get('image_rl', False):                                    
            use_tiny_data = variant.get('use_tiny_data', False)                                         
            if variant.get('data_mix_type') in ['all_state_1step_random_action', 'random_state_1step_random_action', 'random_state_5step_random_action', 'random_state_5step_offRL_action']:
                if variant.get('slac_representation'):
                    trainer.replay_buffer = slac_algo.buffer
                    
                    start = time.time()
                    path_prefix = variant.get('offline_image_data_path_prefix', None)+ '/' + variant.get('env_id', None)
                    framestack_str = '_stack'+str(3)
                    
                    if use_tiny_data:
                        h5f_r_real_name = path_prefix+'/image_numpy_dataset'+framestack_str+'_imgsize_100_tiny.hdf5'
                    else:
                        h5f_r_real_name = path_prefix+'/image_numpy_dataset'+framestack_str+'_imgsize_100.hdf5'
                    use_advanced_data = variant.get('use_advanced_data', False)
                            
                    if variant.get('data_mix_type')=='all_state_1step_random_action':
                        if use_advanced_data:
                            if use_tiny_data:
                                h5f_r_gen_name = path_prefix+'/all_state_1step_random_action' +'/all_state_1step_random_action_dataset-rl_tiny.hdf5'                            
                            else:
                                h5f_r_gen_name = path_prefix+'/all_state_1step_random_action' +'/all_state_1step_random_action_dataset-rl.hdf5'                            
                        else:
                            if use_tiny_data:
                                h5f_r_gen_name = path_prefix+'/all_state_1step_random_action' +'/all_state_1step_random_action_dataset_naive_tiny.hdf5'                                    
                            else:
                                h5f_r_gen_name = path_prefix+'/all_state_1step_random_action' +'/all_state_1step_random_action_dataset_naive.hdf5'                                    
                    

                    data_mix_num_real = variant.get('data_mix_num_real')
                    data_mix_num_gen = variant.get('data_mix_num_gen')
                    savedir = None #variant['logdir'] # 왜인지 모르겠는데 buffer save할때 memory 치솟
                    slac_algo.load_data_in_buffer(h5f_r_real_name, None, expl_env, data_mix_num_real, generated_for_slac = False, \
                        data_mix_type=variant.get('data_mix_type'), generalization_test=variant.get('generalization_test', None))
                    uncertainty_penalty_lambda = variant.get('uncertainty_penalty_lambda', None)
                    uncertainty_type = variant.get('uncertainty_type')
                    slac_algo.load_data_in_buffer(h5f_r_gen_name, savedir, expl_env, data_mix_num_gen, generated_for_slac = True, \
                        data_mix_type=variant.get('data_mix_type'), generalization_test=variant.get('generalization_test', None), \
                        uncertainty_type = uncertainty_type, uncertainty_penalty_lambda=uncertainty_penalty_lambda)
                    
                                    
            
            else:
                if variant.get('slac_representation'):
                    trainer.replay_buffer = slac_algo.buffer                    
                    start = time.time()
                    path_prefix = variant.get('offline_image_data_path_prefix', None)+ '/' + variant.get('env_id', None)
                    framestack_str = '_stack'+str(3)
                    if use_tiny_data:
                        h5f_r_name = path_prefix+'/image_numpy_dataset'+framestack_str+'_imgsize_100_tiny.hdf5'                        
                    else:
                        h5f_r_name = path_prefix+'/image_numpy_dataset'+framestack_str+'_imgsize_100.hdf5'                        
                    savedir = None #variant['logdir']
                    data_num_real = variant.get('data_mix_num_real', None)
                    print('h5f_r name : ', h5f_r_name)
                    slac_algo.load_data_in_buffer(h5f_r_name, savedir, expl_env, data_num = data_num_real, generalization_test=variant.get('generalization_test', None))


                
                
            start =time.time()
            if variant.get('is_dmc', False):
                pass
            else:
                dataset = d4rl.qlearning_dataset(expl_env)
            

            
        else:
            if variant.get('is_dmc', False):
                start = time.time()
                path_prefix = variant.get('offline_image_data_path_prefix', None)+ '/' + variant.get('env_id', None)
                framestack_str = '_stack'+str(3)                
                h5f_r_name = path_prefix+'/image_numpy_dataset'+framestack_str+'_imgsize_100.hdf5'                
                h5f_r = h5py.File(h5f_r_name, 'r')
                dataset = {}
                for k,v in h5f_r.items():
                    if k in ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'timeouts']:
                        dataset[k] = v[:]


                path_loader.load_demos(dataset) 
            else:
                dataset = d4rl.qlearning_dataset(expl_env)

        
        
        if variant.get('is_dmc', False):            
            pass
        else:
            path_loader.load_demos(dataset) 
            remove_dict_for_memory_release(dataset)
            if variant.get('normalize_rewards_by_return_range'):
                normalizer = get_normalization(replay_buffer)
                trainer.reward_transform = normalizer
            
        

    if variant.get('save_initial_buffers', False):
        buffers = dict(
            replay_buffer=replay_buffer,
            demo_train_buffer=demo_train_buffer,
            demo_test_buffer=demo_test_buffer,
        )
        buffer_path = osp.join(logger.get_snapshot_dir(), 'buffers.p')
        pickle.dump(buffers, open(buffer_path, "wb"))

    algorithm.train()
    

def remove_dict_for_memory_release(data_dict):
    data_dict.clear()
    