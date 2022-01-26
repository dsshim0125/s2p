"""
This file provides a more uniform interface to gym.make(env_id) that handles
imports and normalization
"""

import gym

from rlkit.envs.wrappers import NormalizedBoxEnv

DAPG_ENVS = [
    'pen-v0', 'pen-sparse-v0', 'pen-notermination-v0', 'pen-binary-v0', 'pen-binary-old-v0',
    'door-v0', 'door-sparse-v0', 'door-binary-v0', 'door-binary-old-v0',
    'relocate-v0', 'relocate-sparse-v0', 'relocate-binary-v0', 'relocate-binary-old-v0',
    'hammer-v0', 'hammer-sparse-v0', 'hammer-binary-v0',
]

D4RL_ENVS = [
    "maze2d-open-v0", "maze2d-umaze-v0", "maze2d-medium-v0", "maze2d-large-v0",
    "maze2d-open-dense-v0", "maze2d-umaze-dense-v0", "maze2d-medium-dense-v0", "maze2d-large-dense-v0",
    "antmaze-umaze-v0", "antmaze-umaze-diverse-v0", "antmaze-medium-diverse-v0",
    "antmaze-medium-play-v0", "antmaze-large-diverse-v0", "antmaze-large-play-v0",
    "pen-human-v0", "pen-cloned-v0", "pen-expert-v0", "hammer-human-v0", "hammer-cloned-v0", "hammer-expert-v0",
    "door-human-v0", "door-cloned-v0", "door-expert-v0", "relocate-human-v0", "relocate-cloned-v0", "relocate-expert-v0",
    "halfcheetah-random-v0", "halfcheetah-medium-v0", "halfcheetah-expert-v0", "halfcheetah-mixed-v0", "halfcheetah-medium-expert-v0",
    "walker2d-random-v0", "walker2d-medium-v0", "walker2d-expert-v0", "walker2d-mixed-v0", "walker2d-medium-expert-v0",
    "hopper-random-v0", "hopper-medium-v0", "hopper-expert-v0", "hopper-mixed-v0", "hopper-medium-expert-v0"
]

DMC_ENVS = ['cheetah-run-mixed', 'walker-walk-mixed', 'ball_in_cup-catch-mixed', 'reacher-easy-mixed', 'finger-spin-mixed', 'cartpole-swingup-mixed',
            'cheetah-run-expert', 'walker-walk-expert', 'ball_in_cup-catch-expert', 'reacher-easy-expert', 'finger-spin-expert', 'cartpole-swingup-expert',
            'cheetah-run-random', 'walker-walk-random', 'ball_in_cup-catch-random', 'reacher-easy-random', 'finger-spin-random', 'cartpole-swingup-random',
            'cheetah-run-mixed_first_500k', 'walker-walk-mixed_first_500k', 'ball_in_cup-catch-mixed_first_500k', 'reacher-easy-mixed_first_500k', 'finger-spin-mixed_first_500k', 'cartpole-swingup-mixed_first_500k',
            'cheetah-run-mixed_first_100k', 'walker-walk-mixed_first_100k', 'ball_in_cup-catch-mixed_first_100k', 'reacher-easy-mixed_first_100k', 'finger-spin-mixed_first_100k', 'cartpole-swingup-mixed_first_100k',
            'walker-run-mixed_first_500k',
            ]

def make(env_id=None, env_class=None, env_kwargs=None, normalize_env=True):
    assert env_id or env_class
    if env_class:
        env = env_class(**env_kwargs)
    elif env_id in DAPG_ENVS:
        import mj_envs
        assert normalize_env == False
        env = gym.make(env_id)
    elif env_id in D4RL_ENVS:
        import d4rl
        assert normalize_env == False
        env = gym.make(env_id, **env_kwargs)
    
    elif env_id in DMC_ENVS:
        import dmc2gym
        
        env = dmc2gym.make(**env_kwargs)
        # env = dmc2gym.make(
        #     domain_name=args.domain_name,
        #     task_name=args.task_name,
        #     seed=args.seed,
        #     visualize_reward=False,
        #     from_pixels=(args.encoder_type == 'pixel'),
        #     height=args.pre_transform_image_size,
        #     width=args.pre_transform_image_size,
        #     frame_skip=args.action_repeat
        # )
    elif env_id:
        env = gym.make(env_id)
    
    if env_id in DMC_ENVS:
        pass
    else:
        env = env.env # unwrap TimeLimit

    if normalize_env:
        env = NormalizedBoxEnv(env)

    return env
