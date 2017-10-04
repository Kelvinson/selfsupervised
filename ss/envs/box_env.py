from ss.envs.mujoco_env import MujocoEnv
import pdb
import numpy as np
from gym import spaces
import ss.path as path
from mujoco_py.generated import const
import scipy.misc

class BoxEnv(MujocoEnv):
    def __init__(self):
        MujocoEnv.__init__(self, "models/pushing2d_controller.xml")

        # qpos (11):
        # ball_{x, y}, block_{x, y, z}, block_{t1, t2, t3}, marker_{x, y, z}
        # x0_range = np.array([0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 0])
        # self.x0_dist = spaces.Box(-x0_range, x0_range)

    def _set_action(self, action):
        self.sim.data.ctrl[:] = action

    def get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def set_view(self, cam_id):
        self.viewer.cam.fixedcamid = cam_id
        self.viewer.cam.type = const.CAMERA_FIXED

    def reset(self):
        self.t = 0

        interior = 0.35
        # while True:
        #     ball_pos = np.random.random((2)) * 2 * interior - interior
        #     block_pos = np.random.random((2)) * 2 * interior - interior
        #     if np.linalg.norm(block_pos - ball_pos) > 0.25:
        #         break
        block_pos = np.random.random((2)) * 2 * interior - interior
        ball_pos = block_pos + np.array([0.1, -0.05])

        # qpos = self.x0_dist.sample()
        rest = np.zeros((7))
        # rest[2] = np.random.random() * 5
        qpos = np.concatenate((ball_pos, block_pos, rest))
        qvel = np.zeros(self.init_qvel.shape)
        self.set_state(qpos, qvel)
        for i in range(10): # let scene settle
            self.sim.step()
        ob = self.get_obs()

        self.t = 0
        return ob

if __name__ == "__main__":
    b = BoxEnv()
    b.reset()
    b.render()
    b.set_view(0) # top view

    d = path.mkdir(path.DATADIR + "boxenv/")
    for i in range(10):
        b.reset()
        for t in range(100):
            shape = b.action_space.shape
            u = np.zeros(shape)
            b.step(u)
            b.render()
        img = b.get_img()
        scipy.misc.imsave(d + str(i) + ".jpg", img)
