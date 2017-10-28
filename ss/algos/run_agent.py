import mujoco_py
import click
import pickle
from ss.envs.ball_env import BallEnv
import pdb
import numpy as np

@click.command()
@click.argument('agent_pickle')
def main(agent_pickle):
    env = BallEnv()
    agent = pickle.load(open(agent_pickle, "rb"))

    for i in range(100):
        env.reset()
        o = env.get_obs()
        for i in range(agent.horizon):
            u, q = agent.pi(o)
            u = u.flatten()
            o, r, _, __ = env.step(u)
            print(r)
            env.render()

if __name__ == "__main__":
    main()
