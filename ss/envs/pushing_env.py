from ss.envs.mujoco_env import MujocoEnv
import pdb
import numpy as np
from gym import spaces
from mujoco_py.generated import const
import time

class PushingEnv(MujocoEnv):
    def __init__(self):
        MujocoEnv.__init__(self, "models/pushing2d_controller_goal.xml", 3)

    def _set_action(self, action):
        self.sim.data.ctrl[:] = action

    def get_obs(self):
        return np.concatenate([
                self.sim.data.qpos.ravel().copy(),
                self.sim.data.qvel.ravel().copy()
        ])

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ref_point = self.get_site_pos('reference_point')[:2]
        reward_dist = np.linalg.norm(ref_point)
        reward_ctrl = -np.linalg.norm(action)
        reward = reward_dist + reward_ctrl
        ob = self._get_obs()
        done = np.linalg.norm(ref_point) <= 5e-3
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)


    def set_view(self, cam_id):
        self.viewer.cam.fixedcamid = cam_id
        self.viewer.cam.type = const.CAMERA_FIXED

    def reset(self):
        self.t = 0
        # randomize the position and pose of the L shaped block
        angle = np.random.uniform(1./4.1, 1./3.9)*np.pi
        direction = np.random.choice([angle, -angle, (np.pi - angle), (angle - np.pi)])
        #direction = np.random.choice([np.pi*0.25, -np.pi*0.25, np.pi*0.75, -np.pi*0.75, np.pi*(1./3), -np.pi(1./3), np.pi(2./3), -np.pi(2./3)])
        #ball_pos = np.random.uniform(0.1, 0.15, 2)* np.sqrt(2)*np.array([np.cos(direction), np.sin(direction)])
        block_pos = np.random.uniform(0.15, 0.2, 2) * (1./abs(np.array([np.cos(direction), np.sin(direction)])))* np.array([np.cos(direction), np.sin(direction)])
        block_pos = np.concatenate([block_pos, np.array([0.])])
        alpha = np.random.choice([-np.pi*1./4, np.pi*1./3, -np.pi*1./4, -np.pi*1./3])
        block_pose = np.array([np.cos(alpha/2), 0. , 0., np.sin(alpha/2)])
        block_qpos = np.concatenate([block_pos, block_pose])
        ball_qpos = np.random.choice([0.05, -0.05], 2)
        goal_qpos = np.array([0., 0.])
        qpos = np.concatenate([ball_qpos, block_qpos, goal_qpos])
        qvel = np.zeros(self.init_qvel.shape)
        self.set_state(qpos, qvel)
        for i in range(10): # scene settling time
            self.sim.step()
        ob = self._get_obs()
        self. t= 0
        return ob

if __name__ == '__main__':
    env = PushingEnv()
    env.reset()
    env.render()
    env.set_view(0) # topview
    time.sleep(10)
