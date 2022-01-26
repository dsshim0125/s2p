import random

import cv2
import mujoco_py
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict
import copy
from multiworld_custom.core.gym_to_multi_env import GymToMultiEnv # MujocoGymToMultiEnv

class MujocoGymToMultiEnv(GymToMultiEnv):
    def __init__(self, wrapped_env):        
        super().__init__(wrapped_env)
        self._state_goal = None
        

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
        

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.frame_skip
        if self.sim.data.ctrl is not None and ctrl is not None:
            self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])


    def initialize_camera(self, init_fctn):
        sim = self.sim
        viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=self.device_id)
        init_fctn(viewer.cam)
        sim.add_render_context(viewer)

    def get_diagnostics(self, paths, **kwargs):
        return {}
    
    
    def render(self, mode='rgb_array',
               width=128,
               height=128,
               camera_id=None,
               camera_name=None,
               env_name = None,
               is_dmc = False,
               ):
        if 'kitchen' in [env_name]:
            return self.wrapped_env.custom_render(mode, width, height)
        elif is_dmc:
            return self.wrapped_env.render(mode=mode, width=width, height=height)
        else:
            return self.wrapped_env.render(mode, width, height, camera_id, camera_name)

    def get_image(self, width=84, height=84, camera_name=None, env_name = None, is_dmc = False):
        # return self.sim.render(
        #     width=width,
        #     height=height,
        #     camera_name=camera_name,
        # )        
        return self.render(width=width, height=height, camera_name=camera_name, env_name = env_name, is_dmc = is_dmc)

    
    def get_env_state(self):
        joint_state = self.sim.get_state()                
        return copy.deepcopy(joint_state)
        # if goal env
        # goal = self._state_goal.copy()
        # return joint_state, goal

    def set_env_state(self, state):
        joint_state= state
        self.sim.set_state(joint_state)        
        self.sim.forward()
        
        # if goal env
        # joint_state, goal = state
        # self.sim.set_state(joint_state)        
        # self.sim.forward()
        # self._state_goal = goal
        # self._set_goal_marker(goal)
        

    def set_to_goal(self, goal):
        pass
        
    
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        pass
        

        
import gym
from gym.spaces import Box

        
from collections import deque
class FrameStack(object): #gym.Wrapper
    # NOTE : Return obs is [C*n_stack, H, W] ! You have to transpose when you want to visualize!!
    def __init__(self, env, k):
        # gym.Wrapper.__init__(self, env)
        self.wrapped_env = env
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        # self._max_episode_steps = env._max_episode_steps

    def reset(self):        
        obs = self.wrapped_env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

    
    def __getattr__(self, attrname):
        return getattr(self.wrapped_env, attrname)

class StateStack(gym.Wrapper):
    # NOTE : Return obs is [C*n_stack, H, W] ! You have to transpose when you want to visualize!!
    def __init__(self, env, k, state_type = 'qpos', env_id=None):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        
        shp = env.observation_space.shape
        self.state_type = state_type
        self.env_id = env_id
        assert 'cheetah' in env_id, 'currently debugging for cheetah'
        self.qpos_idx = 8

        self.observation_space = gym.spaces.Box(
            low=np.tile(env.observation_space.low[:self.qpos_idx], k),
            high=np.tile(env.observation_space.high[:self.qpos_idx], k),
            shape=((self.qpos_idx*k,)),# (shp[0] * k),
            dtype=env.observation_space.dtype
        )
        
        # self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        obs = obs[:8]
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs[:8]
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

