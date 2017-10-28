import numpy as np
from gym import utils
import mujoco_py
from gym.envs.mujoco import mujoco_env
# from ss.envs.mujoco_env import MujocoEnv
import pdb
from ss.path import MODELDIR

def obs_to_goal(obs):
    return obs[14:17]

def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    hand_pos = obs[14:17]
    goal_pos = obs[17:20]
    r = np.linalg.norm(hand_pos - goal_pos) < 0.1
    return (r - 1).astype(float)

class Reacher7Dof(mujoco_env.MujocoEnv):
    def __init__(self):
        # utils.EzPickle.__init__(self)
        mjfile = MODELDIR + "reacher_7dof.xml"
        self.viewer = None
        mujoco_env.MujocoEnv.__init__(self, mjfile, 5)
        self._desired_xyz = np.zeros(3)
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(17, 20)
        self.reward_fn = get_sparse_reward

    def get_reward(self, obs):
        return get_sparse_reward(obs)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                                                       high=0.005, size=self.model.nv)
        self._desired_xyz = np.random.uniform(
            np.array([-0.75, -1.25, -0.24]),
            np.array([0.75, 0.25, 0.6]),
        )
        qpos[-7:-4] = self._desired_xyz
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        distance = np.linalg.norm(
            self.get_body_com("tips_arm") - self.get_body_com("goal")
        )
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = self.get_reward(ob)
        done = False
        return ob, reward, done, dict(distance=distance)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("goal"),
        ])

    def get_state_data(self):
        return self.model.data.qpos.flat[:], self.model.data.qvel.flat[:]

if __name__ == "__main__":
    b = Reacher7Dof()
    b.reset()
    # b.render()

    for i in range(1000):
        b.reset()
        for t in range(10):
            shape = b.action_space.shape
            u = np.random.random(shape) * 2 - 1
            u = np.zeros(shape)
            o, _, _, _ = b.step(u)
            print(b.reward_fn(o))
            b.render()
