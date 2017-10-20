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
    return obs[2:4]

def get_l2_reward(obs):
    ball_pos = obs[1:3]
    goal_pos = obs[3:5]
    r = -np.linalg.norm(ball_pos - goal_pos)
    return r

def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    block_pos = obs[2:4]
    goal_pos = obs[4:6]
    r = np.linalg.norm(block_pos - goal_pos) < 0.1
    return (r - 1).astype(float)

class BoxEnv(MujocoEnv):
    def __init__(self, reward_type="sparse"):
        self.ball_pos = np.zeros((2))
        self.block_pos = np.zeros((2))
        self.goal_pos = np.zeros((2))

        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(4, 6)
        self.reward_fn = {"sparse": get_sparse_reward,
                           "l2": get_l2_reward}[reward_type]

        u_range = np.ones((2))
        self.action_space = spaces.Box(-u_range, u_range)

        o_range = np.ones((6))
        self.observation_space = spaces.Box(-o_range, o_range)

        mjfile = "models/pushing2d_controller_goal.xml"
        MujocoEnv.__init__(self, mjfile)
        self.reset()

        # qpos (11):
        # ball_{x, y}, block_{x, y, z}, block_{t1, t2, t3}, marker_{x, y, z}
        # x0_range = np.array([0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0])
        # self.x0_dist = spaces.Box(-x0_range, x0_range)

    def set_action(self, u):
        self.sim.data.qvel[:2] = u.copy()

    def get_reward(self, obs):
        return self.reward_fn(obs)

    def get_obs(self):
        o = np.zeros((6))
        o[:4] = self.sim.data.qpos[:4].copy()
        o[4:] = self.sim.data.qpos[8:10].copy()
        return o

    def set_view(self, cam_id):
        self.viewer.cam.fixedcamid = cam_id
        self.viewer.cam.type = const.CAMERA_FIXED

    def reset(self):
        interior = 0.35
        while True:
            ball_pos = np.random.random((2)) * 2 * interior - interior
            block_pos = np.random.random((2)) * 2 * interior - interior
            goal_pos = np.random.random((2)) * 2 * interior - interior
            if np.linalg.norm(block_pos - ball_pos) > 0.25:
                break
        # block_pos = np.random.random((2)) * 2 * interior - interior
        # ball_pos = block_pos + np.array([0.1, -0.05])

        # qpos = self.x0_dist.sample()
        rest = np.zeros((7))
        rest[4:6] = goal_pos
        # rest[2] = np.random.random() * 5
        qpos = np.concatenate((ball_pos, block_pos, rest))
        qvel = np.zeros(self.init_qvel.shape)
        self.set_state(qpos, qvel)
        for i in range(10): # let scene settle
            self.sim.step()
        ob = self.get_obs()

        self.t = 0
        return ob

# if __name__ == "__main__":
#     b = BoxEnv()
#     b.reset()
#     b.render()
#     b.set_view(0) # top view

#     d = path.mkdir(path.DATADIR + "boxenv/")
#     for i in range(10):
#         b.reset()
#         for t in range(100):
#             shape = b.action_space.shape
#             u = np.zeros(shape)
#             b.step(u)
#             b.render()
#         img = b.get_img()
#         scipy.misc.imsave(d + str(i) + ".jpg", img)

if __name__ == "__main__":
    b = BoxEnv()
    b.reset()
    b.render()

    for i in range(10):
        b.reset()
        for t in range(100):
            shape = b.action_space.shape
            u = np.random.random(shape) * 2 - 1
            b.step(u)
            b.render()
