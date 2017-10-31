"""Adapted from OpenAI continuous mountain car env"""

import numpy as np
from gym import utils
import mujoco_py
from gym.envs.mujoco import mujoco_env
# from ss.envs.mujoco_env import MujocoEnv
import pdb
from ss.path import MODELDIR

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

def obs_to_goal(obs):
    return obs[:1]

def get_sparse_reward(obs):
    """-1 if far, 0 if close"""
    state = obs[0:2]
    goal = obs[2:4]
    r = np.linalg.norm(state - goal) < 0.1
    return (r - 1).astype(float)

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.obs_to_goal = obs_to_goal
        self.goal_idx = slice(2, 4)
        self.reward_fn = get_sparse_reward

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        self.state_space = spaces.Box(self.low_state, self.high_state)
        l = np.array([self.min_position, -self.max_speed, self.min_position, -self.max_speed])
        h = np.array([self.max_position, self.max_speed, self.max_position, self.max_speed])
        self.observation_space = spaces.Box(l, h)

        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0
        self.state = np.array([position, velocity])

        o = self.get_obs()
        reward = self.reward_fn(o)

        distance = abs(self.goal[0] - self.state[0])

        return o, reward, False, {"distance": distance}

    def get_reward(self, obs):
        return self.reward_fn(obs)

    def get_obs(self):
        s = self.state
        g = self.goal
        o = np.concatenate((s, g))
        return o

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        # self.state = self.observation_space.sample()
        self.goal = self.state_space.sample()
        return self.get_obs()

    def get_state_data(self):
        return self.get_obs()

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

            goal_car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            goal_car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.goaltrans = rendering.Transform()
            goal_car.add_attr(self.goaltrans)
            self.viewer.add_geom(goal_car)

            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            goal_pos = self.goal_position
            flagx = (goal_pos-self.min_position)*scale
            flagy1 = self._height(goal_pos)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        goal_pos = self.goal[0]
        self.goaltrans.set_translation((goal_pos-self.min_position)*scale, self._height(goal_pos)*scale)
        self.goaltrans.set_rotation(math.cos(3 * goal_pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__ == "__main__":
    b = MountainCarEnv()
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
