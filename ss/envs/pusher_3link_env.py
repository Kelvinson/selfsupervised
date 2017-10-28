import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from ss.path import MODELDIR

def obs_to_goal(obs):
    return obs[6:8]

def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    cylinder_pos = obs[8:10]
    goal_pos = obs[10:12]
    r = np.linalg.norm(cylinder_pos - goal_pos) < 0.1
    return (r - 1).astype(float)


class Pusher2DEnv(mujoco_env.MujocoEnv):
    def __init__(self):
        self._target_cylinder_position = np.zeros(2)
        self.viewer = None
        mjfile = MODELDIR + '3link_gripper_push_2d.xml'
        mujoco_env.MujocoEnv.__init__(self, mjfile, 5,)
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(10, 12)
        self.reward_fn = get_sparse_reward

    def get_reward(self, obs):
        return get_sparse_reward(obs)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        rotation_angle = 90
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        qpos = (
            np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos.squeeze()
        )
        qpos[-3:] = self.init_qpos.squeeze()[-3:]
        # Object position
        obj_pos = np.random.uniform(
            #         x      y
            np.array([0.3, -0.8]),
            np.array([0.8, -0.3]),
        )
        qpos[-6:-4] = obj_pos
        self._target_cylinder_position = np.random.uniform(
            np.array([-1, -1]),
            np.array([1, 0]),
            2
        )
        qpos[-4:-2] = self._target_cylinder_position
        qpos[-2:] = np.zeros(2)   # ignore for now
        qvel = self.init_qvel.copy().squeeze()
        qvel[:] = 0

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _step(self, a):
        hand_to_object_distance = np.linalg.norm(
            self.model.data.site_xpos[0][:2] - self.get_body_com("object")[:2]
        )
        object_to_goal_distance = np.linalg.norm(
            self.get_body_com("goal") - self.get_body_com("object")
        )
        hand_to_hand_goal_distance = np.linalg.norm(
            self.model.data.site_xpos[0][:2] - self.get_body_com("hand_goal")[:2]
        )

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = self.get_reward(ob)
        done = False
        return ob, reward, done, dict(
            hand_to_hand_goal_distance=hand_to_hand_goal_distance,
            hand_to_object_distance=hand_to_object_distance,
            object_to_goal_distance=object_to_goal_distance,
            distance=object_to_goal_distance,
        )

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:3],
            self.model.data.qvel.flat[:3],
            self.model.data.site_xpos[0][:2],
            self.get_body_com("object")[:2],
            self._target_cylinder_position,
        ])

    def get_state_data(self):
        return self.model.data.qpos.flat[:], self.model.data.qvel.flat[:]

if __name__ == "__main__":
    b = Pusher2DEnv()
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

