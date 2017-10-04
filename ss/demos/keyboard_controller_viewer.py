import mujoco_py
import glfw
from ss.envs.mujoco_env import MujocoEnv
import numpy as np

class KeyboardControllerViewer(mujoco_py.MjViewer):
    """Keyboard control with P, L, ;, " as the keys"""
    def __init__(self, sim):
        mujoco_py.MjViewer.__init__(self, sim)
        self.u = np.zeros((2))
        self.sensitivity = 1.0
        glfw.set_key_callback(self.window, self.key_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_P:
            self.u[1] = self.sensitivity
        elif key == glfw.KEY_SEMICOLON:
            self.u[1] = -self.sensitivity
        else:
            self.u[1] = 0
        if key == glfw.KEY_APOSTROPHE:
            self.u[0] = self.sensitivity
        elif key == glfw.KEY_L:
            self.u[0] = -self.sensitivity
        else:
            self.u[0] = 0
        super().key_callback(window, key, scancode, action, mods)

if __name__ == "__main__":
    # b = MujocoEnv("models/ball_env.xml")
    b = MujocoEnv("models/ball_env.xml", mjviewer=KeyboardControllerViewer)
    b.reset()
    b.render()

    for i in range(10):
        b.reset()
        for t in range(1000):
            b.step(b.viewer.u)
            b.render()
