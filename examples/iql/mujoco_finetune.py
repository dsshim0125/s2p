"""
AWR + SAC from demo experiment
"""

# from multiworld.core.image_env import normalize_image
from numpy.core.fromnumeric import transpose
from torch.nn.functional import normalize
from rlkit.demos.source.hdf5_path_loader import HDF5PathLoader
from rlkit.launchers.experiments.awac.finetune_rl import experiment

from rlkit.launchers.launcher_util import run_experiment

from rlkit.torch.sac.policies import GaussianPolicy, TanhGaussianPolicy
from rlkit.torch.sac.iql_trainer import IQLTrainer
from rlkit.torch.sac.cql_trainer import CQLTrainer


import random
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str)
parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2')
parser.add_argument('--algo_type', type=str, default='iql')
parser.add_argument('--image_rl', action='store_true')
parser.add_argument('--no_curl_contrastive_learning', action='store_true')

parser.add_argument('--seperate_vf_encoder', action='store_true')
parser.add_argument('--pretrain_contrastive', action='store_true')

parser.add_argument('--data_mix_type', type=str, default=None)
parser.add_argument('--data_mix_num_real', type=int, default=None)
parser.add_argument('--data_mix_num_gen', type=int, default=None)
parser.add_argument('--add_sac_entropy', action='store_true')

parser.add_argument('--slac_representation', action='store_true')
parser.add_argument('--freeze_slac', action='store_true')
parser.add_argument('--slac_latent_model_load_dir', type=str, default='')
parser.add_argument('--slac_buffer_load_dir', type=str, default=None)
parser.add_argument('--slac_policy_input_type', type=str, default=None)
parser.add_argument('--slac_obs_reset_w_same_obs', action='store_true')

parser.add_argument('--uncertainty_penalty_lambda', type=float, default=1.0)
parser.add_argument('--uncertainty_type', type=str, default=None)
parser.add_argument('--uncertainty_ablation', action='store_true')
parser.add_argument('--generalization_test', type=str, default=None)

parser.add_argument('--n_experiments', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--gpu_list', nargs="+", default=[0,1,2])
parser.add_argument('--seperate_buffer', action='store_true')
parser.add_argument('--use_advanced_data', action='store_true')
parser.add_argument('--use_tiny_data', action='store_true')


parser.add_argument('--rad_aug', action='store_true')
parser.add_argument('--rad_aug_type', type=str, default=None)

parser.add_argument('--debug', action='store_true')


args = parser.parse_args()

image_rl = args.image_rl

from_pixels = False
visualize_reward = False # prevent green colorizing when render
dm_control_env_dict = { 
                        'cheetah-run-mixed_first_500k' : dict(domain_name='cheetah', task_name='run', frame_skip=4, from_pixels = from_pixels, visualize_reward = visualize_reward),                        
                        
                        }
is_dmc = True if args.env_name in dm_control_env_dict.keys() else False

if args.slac_representation:
    assert args.no_curl_contrastive_learning
elif args.no_curl_contrastive_learning:
    assert not args.slac_representation

if is_dmc:
    max_path_length = int(1000/(dm_control_env_dict.get(args.env_name).get('frame_skip')))


import os

save_path = './data'    
slac_latent_model_path = './slac_pytorch/logs'


trainer_class_dict = {'iql' : IQLTrainer, 'cql' : CQLTrainer}
trainer_kwargs_dict = {'iql' : dict(discount=0.99,
                                    policy_lr=1E-4,
                                    qf_lr=3E-4,
                                    reward_scale=1,
                                    soft_target_tau=0.005,
                                    policy_weight_decay=0,
                                    q_weight_decay=0,
                                    reward_transform_kwargs=None,
                                    terminal_transform_kwargs=None,
                                    beta=1.0/10, 
                                    quantile=0.7, 
                                    clip_score=100,
                                    image_rl = args.image_rl,            
                                    policy_update_period=1,
                                    q_update_period=1,
                                    curl_update_period =1,
                                    target_update_period = 2,
                                    encoder_target_tau = 0.025,
                                    no_curl_contrastive_learning = args.no_curl_contrastive_learning,
                                    pretrain_contrastive = args.pretrain_contrastive,
                                    add_sac_entropy = args.add_sac_entropy,
                                    training_start_steps = 0, 
                                    # SLAC
                                    slac_representation = args.slac_representation,                                        
                                    freeze_slac = args.freeze_slac,
                                    slac_update_period = 1,
                                    slac_policy_input_type = args.slac_policy_input_type,
                                    
                                    ),
                        'cql' : dict(discount=0.99,
                                    soft_target_tau=5e-3,
                                    policy_lr=1E-4,
                                    qf_lr=3E-4,
                                    reward_scale=1,
                                    use_automatic_entropy_tuning=True,

                                    # Target nets/ policy vs Q-function update
                                    policy_eval_start=40000,
                                    num_qs=2,

                                    # min Q
                                    temp=1.0,
                                    min_q_version=3,
                                    min_q_weight=5.0,

                                    # lagrange
                                    with_lagrange=False,
                                    lagrange_thresh=-1.0,
                                    
                                    # extra params
                                    num_random=10,
                                    max_q_backup=False,
                                    deterministic_backup=False,
                                    
                                    image_rl = args.image_rl,              
                                    encoder_lr = 1e-3,                                        
                                                                           
                                    curl_update_period = 1,
                                    encoder_target_tau = 0.05,
                                    no_curl_contrastive_learning = args.no_curl_contrastive_learning,
                                    training_start_steps = 0,
                                    pretrain_contrastive = args.pretrain_contrastive,
                                    # for SLAC
                                    slac_representation = args.slac_representation,                                        
                                    freeze_slac = args.freeze_slac,
                                    slac_update_period = 1,
                                    slac_policy_input_type = args.slac_policy_input_type,
                                ),
                       
                        }
trainer_class=trainer_class_dict[args.algo_type]
trainer_kwargs = trainer_kwargs_dict[args.algo_type]
if args.slac_representation:
    offline_image_data_path_prefix = save_path+'/trajwise'


if not image_rl: 
    if is_dmc:        
        env_kwargs = dm_control_env_dict.get(args.env_name)

    variant = dict(
        algo_kwargs=dict(
            start_epoch=-1000, # offline epochs
            num_epochs=1001, # online epochs
            batch_size=256,
            num_eval_steps_per_epoch=10*max_path_length if not args.debug else 100,
            num_trains_per_train_loop=1000 if not args.debug else 100,
            num_expl_steps_per_train_loop=1000 if not args.debug else 100,
            min_num_steps_before_training=1000 if not args.debug else 100,
        ),
        max_path_length=max_path_length,
        video_eval_max_path_length=max_path_length,
        video_expl_max_path_length=1000,
        replay_buffer_size=int(2E6),
        layer_size=256,
        policy_class=TanhGaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, ],
        ),
        
        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_class=trainer_class,
        trainer_kwargs=trainer_kwargs,
        
        algo_type = args.algo_type,
        offline_image_data_path_prefix = offline_image_data_path_prefix,

        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=HDF5PathLoader,
        path_loader_kwargs=dict(),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=False,
        load_env_dataset_demos=True,

        normalize_env=False,
        env_id=args.env_name, #'halfcheetah-medium-v2',
        normalize_rewards_by_return_range=True,

        seed=random.randint(0, 100000),
        
        is_dmc = is_dmc,
        image_rl = False,
        
        state_rl_100k_debug = False,
        # frame_stack = 3,
        use_tiny_data = args.use_tiny_data,
        save_video = True, #False,
        env_kwargs = env_kwargs,
        
    )

else: # image rl
    from examples.iql.custom_networks import  TanhGaussianPolicyWithEncoder    
    '''
    epoch *num_trains_per_train_loop = total gradient steps (you should consider network update period)
    '''
    
    if is_dmc:
        if args.slac_representation:
            env_kwargs = dm_control_env_dict.get(args.env_name)
            env_kwargs.update({'from_pixels' : True, 'width' : 100, 'height' : 100})
            if args.generalization_test:
                if 'walker' in args.env_name:
                    env_kwargs.update({'task_name' : 'run'})
        else:
            env_kwargs = dm_control_env_dict.get(args.env_name) 
        
    
    
    if (args.algo_type in ['cql', 'bear']) and (not args.slac_representation):
        batch_size = 128
    else:
        batch_size = 128



    variant = dict(
        algo_kwargs=dict(
            start_epoch=-150 if not args.debug else -1, # offline epochs
            num_epochs=151 if not args.debug else 1, # online epochs
            batch_size=batch_size,
            num_eval_steps_per_epoch=10*max_path_length if not args.debug else 125, 
            num_trains_per_train_loop=2000 if not args.debug else 10,
            num_expl_steps_per_train_loop=2000 if not args.debug else 125, 
            min_num_steps_before_training=1000 if not args.debug else 125,
            num_pretrains = 10000 if args.pretrain_contrastive else 0,            
            slac_representation = args.slac_representation,
            rad_aug = args.rad_aug,
        ),
        max_path_length=max_path_length,
        video_eval_max_path_length=max_path_length,
        video_expl_max_path_length=1000,
        replay_buffer_size=int(1E6), # No meaning
        image_replay_buffer_size = int(1E5) if args.data_mix_type is None else int(args.data_mix_num_real+args.data_mix_num_gen), # 100k
        layer_size=256,
        policy_class=TanhGaussianPolicyWithEncoder,
        policy_kwargs=dict(
            hidden_sizes=[1024, 1024, ],
            
        ),
        qf_kwargs=dict(
            hidden_sizes=[1024, 1024, ],
        ),
        vf_kwargs=dict(
            hidden_sizes=[1024, 1024, ],
        ),
        
        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_class=trainer_class,
        trainer_kwargs=trainer_kwargs,
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=HDF5PathLoader,
        path_loader_kwargs=dict(),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=False,
        load_env_dataset_demos=True,

        normalize_env=False,
        env_id=args.env_name,
        normalize_rewards_by_return_range=True if not is_dmc else False,

        seed=random.randint(0, 100000),
        
        use_tiny_data = args.use_tiny_data,
        generalization_test = args.generalization_test,
        use_advanced_data = args.use_advanced_data,
        seperate_buffer = args.seperate_buffer,
        uncertainty_penalty_lambda = args.uncertainty_penalty_lambda,
        uncertainty_type = args.uncertainty_type,
        slac_representation = args.slac_representation,
        slac_latent_model_load_dir = slac_latent_model_path + args.slac_latent_model_load_dir,
        slac_buffer_load_dir = args.slac_buffer_load_dir, 
        slac_policy_input_type = args.slac_policy_input_type,
        slac_obs_reset_w_same_obs =  args.slac_obs_reset_w_same_obs,
        
        
        slac_algo_kwargs = dict(buffer_size = int(1.05e5),
                                image_size = 100,
                                feature_dim = 256,
                                z1_dim = 32,
                                z2_dim = 256,
                                ),
        # for RAD
        rad_aug = args.rad_aug,
        rad_aug_type = args.rad_aug_type,        

        algo_type = args.algo_type,
        memory_efficient_way = True,
        
        data_mix_type = args.data_mix_type,
        data_mix_num_real = args.data_mix_num_real,
        data_mix_num_gen = args.data_mix_num_gen,
        seperate_vf_encoder = args.seperate_vf_encoder,
        
        env_kwargs = env_kwargs,
        offline_image_data_path_prefix = offline_image_data_path_prefix,
        frame_stack = 3,
        image_rl = True,
        is_dmc = is_dmc,
        image_obs_shape = (3*3,100,100), # (C*frame stack, H,W)
        
        curl_learning = False if args.slac_representation else True,        
        curl_crop_image_size = 84,
        
        save_video = True,
        
    )


def main():
    if is_dmc:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()
        print('xvfb started!')

    is_multiprocess = False
    if args.n_experiments > 1:
        print('Currently {} multiprocess is used!'.format(args.n_experiments))
        is_multiprocess = True
        from multiprocessing import Process

    if is_multiprocess:   
        processes = []     
        for idx, e in enumerate(range(args.n_experiments)):            
            gpu_id = args.gpu_list[idx]
            # gpu_id = idx
            print('gpu id : ', gpu_id)
            seed=random.randint(0, 100000)
            variant['seed'] = seed
            def train_func():
                run_experiment(experiment,
                    variant=variant,
                    exp_prefix= args.algo_type+'-'+ args.env_name+'-' + args.exp_name,
                    mode="here_no_doodad",
                    unpack_variant=False,
                    use_gpu=True, 
                    gpu_id = gpu_id,
                    snapshot_mode = 'all',
                )    
            # Awkward hacky process runs, because Tensorflow does not like
            # repeatedly calling train_AC in the same thread.
            
            p = Process(target=train_func, args=tuple())
            p.start()
            processes.append(p)
            
            # if you comment in the line below, then the loop will block
            # until this process finishes
            # p.join()
            time.sleep(10) # waiting for h5py load time!
        
        for p in processes:
            p.join()
            

    else:        
        base_log_dir = save_path+'/OfflineRL/state' if not args.image_rl else save_path+'/OfflineRL/image'
        run_experiment(experiment,
            variant=variant,
            exp_prefix= args.algo_type+'-'+ args.env_name+'-' + args.exp_name,
            mode="here_no_doodad",
            unpack_variant=False,
            use_gpu=True, 
            gpu_id = args.gpu_id,
            snapshot_mode = 'all',
            
            base_log_dir = base_log_dir,
        )
    if is_dmc:
        vdisplay.stop()

if __name__ == "__main__":
    main()
