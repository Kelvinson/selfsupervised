from ss.envs.mujoco_env import MujocoEnv
import pdb
import numpy as np
from gym import spaces
import ss.path as path
from mujoco_py.generated import const
import scipy.misc

class BallEnv(MujocoEnv):
    def __init__(self):
        self.ball_pos = np.zeros((2))
        self.goal_pos = np.zeros((2))

        u_range = np.ones((2))
        self.action_space = spaces.Box(-u_range, u_range)

        o_range = np.ones((2))
        self.observation_space = spaces.Box(-o_range, o_range)

        MujocoEnv.__init__(self, "models/ball_env.xml")
        self.reset()

    def reset(self):
        self.t = 0
        interior = 0.3
        self.ball_pos = np.random.random((2)) * 2 * interior - interior
        self.goal_pos = np.zeros((2)) # np.random.random((2)) * 2 * interior - interior
        qpos = np.concatenate((self.ball_pos, self.goal_pos))
        qvel = np.zeros(self.init_qvel.shape)
        self.set_state(qpos, qvel)
        ob = self.get_obs()
        return ob

    def get_obs(self):
        return self.sim.data.qpos[:2].copy()

    def set_action(self, u):
        self.sim.data.qvel[:2] = u.copy()

    def get_reward(self, obs):
        # return np.linalg.norm(self.ball_pos - self.goal_pos) < 0.05
        ball_pos = obs[:2]
        r = -np.linalg.norm(ball_pos - self.goal_pos)
        # print(r)
        return r

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
