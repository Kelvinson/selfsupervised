from ss.envs.ball_env import BallEnv
import mujoco_py
import time
import numpy as np
import timeit
import os
from ss.sandbox.parallel.config import TOTAL_STEPS, N_POOL

time_fn = lambda: os.times()[4]
N_STEPS = int(TOTAL_STEPS / N_POOL)

envs = []
sims = []
for i in range(N_POOL):
    env = BallEnv()
    envs.append(env)
    sims.append(env.sim)
pool = mujoco_py.MjSimPool(sims)

start = time_fn()
for _ in range(N_STEPS):
    for i in range(N_POOL):
        envs[i].set_action(np.ones((2)))
    pool.step()
    pool.forward()
    # envs[0].render()
end = time_fn()
elapsed_pool = end - start
sps_pool = TOTAL_STEPS / elapsed_pool
print("time", elapsed_pool)
print("steps per second, pool:", sps_pool)
# print("speedup:", sps_pool / sps_one)
