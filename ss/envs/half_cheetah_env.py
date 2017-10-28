from collections import OrderedDict

import numpy as np
from gym.envs.mujoco import HalfCheetahEnv
import pdb

def obs_to_goal(obs):
    return obs[8:9]

def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    desired_vel = obs[17:18]
    actual_vel = obs[8:9]
    r = np.linalg.norm(desired_vel - actual_vel) < 0.5
    return (r - 1).astype(float)

def half_cheetah_cost_fn(states, actions, next_states):
    input_is_flat = len(states.shape) == 1
    if input_is_flat:
        states = np.expand_dims(states, 0)
    desired_vels = states[:, 17:18]
    actual_vels = states[:, 8:9]
    costs = np.linalg.norm(
        desired_vels - actual_vels,
        axis=1,
        ord=2,
    )
    if input_is_flat:
        costs = costs[0]
    return costs

class HalfCheetah(HalfCheetahEnv):
    def __init__(self):
        self.target_x_vel = np.random.uniform(-10, 10)
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(17, 18)
        self.reward_fn = get_sparse_reward
        super().__init__()

    def get_reward(self, obs):
        return get_sparse_reward(obs)

    def _step(self, action):
        ob, _, done, info_dict = super()._step(action)
        xvel = ob[8]
        desired_xvel = self.target_x_vel
        xvel_error = np.linalg.norm(xvel - desired_xvel)
        new_ob = np.hstack((ob, self.target_x_vel))
        reward = get_sparse_reward(new_ob)
        info_dict['xvel'] = xvel
        info_dict['desired_xvel'] = desired_xvel
        info_dict['xvel_error'] = xvel_error
        info_dict['distance'] = xvel_error
        return new_ob, reward, done, info_dict

    def reset_model(self):
        ob = super().reset_model()
        self.target_x_vel = np.random.uniform(-10, 10)
        return np.hstack((ob, self.target_x_vel))

    def get_state_data(self):
        return self.model.data.qpos.flat[:], self.model.data.qvel.flat[:]

if __name__ == "__main__":
    b = HalfCheetah()
    b.reset()
    b.render()

    for i in range(10):
        b.reset()
        for t in range(100):
            shape = b.action_space.shape
            u = np.zeros(shape)
            # u = np.array((1,))
            b.step(u)
            b.render()
