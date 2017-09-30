from ss.envs.mujoco_env import MujocoEnv
import pdb
import numpy as np

class BoxEnv(MujocoEnv):
    def __init__(self):
        MujocoEnv.__init__(self, "models/pushing2d_controller.xml")

    def _set_action(self, action):
        self.sim.data.ctrl[:] = action

if __name__ == "__main__":
    b = BoxEnv()
    for i in range(10):
        b.reset()
        for t in range(100):
            shape = b.action_space.shape
            u = np.zeros(shape) + 1
            b.step(u)
            b.render()
