from ss.envs.mujoco_env import MujocoEnv
import pdb
import numpy as np
from gym import spaces
import ss.path as path
from mujoco_py.generated import const
import scipy.misc

def obs_to_goal(obs):
    """State to goal function for HER.
    To pickle the function it has to be defined like this.
    """
    return obs[1:3]

def get_l2_reward(obs):
    ball_pos = obs[1:3]
    goal_pos = obs[3:5]
    r = -np.linalg.norm(ball_pos - goal_pos)
    return r

def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    ball_pos = obs[1:3]
    goal_pos = obs[3:5]
    r = np.linalg.norm(ball_pos - goal_pos) < 0.1
    return (r - 1).astype(float)

class BallEnv(MujocoEnv):
    def __init__(self, contained=True, reward_type="sparse"):
        self.ball_pos = np.zeros((2))
        self.goal_pos = np.zeros((2))

        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(3, 5)
        self.reward_fn = {"sparse": get_sparse_reward,
                           "l2": get_l2_reward}[reward_type]

        u_range = np.ones((2))
        self.action_space = spaces.Box(-u_range, u_range)

        o_range = np.ones((5))
        self.observation_space = spaces.Box(-o_range, o_range)

        if contained:
            mjfile = "models/ball_env_contained.xml"
            # mjfile = "models/ball_env_contained_blocked.xml"
        else:
            mjfile = "models/ball_env.xml"
        MujocoEnv.__init__(self, mjfile)
        self.reset()

    def reset(self):
        self.t = 0
        interior = 0.3
        # self.ball_pos = np.random.random((2)) * 2 * interior - interior
        self.ball_pos = self.sim.data.qpos[:2]
        self.goal_pos = np.random.random((2)) * 2 * interior - interior # np.random.random((2)) * 2 * interior - interior
        qpos = np.concatenate((self.ball_pos, self.goal_pos))
        # qvel = np.zeros(self.init_qvel.shape)
        qvel = self.sim.data.qvel[:]
        self.set_state(qpos, qvel)
        ob = self.get_obs()
        return ob

    def get_obs(self):
        o = np.zeros((5))
        o[0] = self.t
        o[1:] = self.sim.data.qpos[:4].copy()
        return o

    def set_action(self, u):
        self.sim.data.qvel[:2] = u.copy()

    def get_reward(self, obs):
        return self.reward_fn(obs)

if __name__ == "__main__":
    b = BallEnv()
    b.reset()
    b.render()

    for i in range(10):
        b.reset()
        for t in range(100):
            shape = b.action_space.shape
            u = np.zeros(shape)
            b.step(u)
            b.render()
