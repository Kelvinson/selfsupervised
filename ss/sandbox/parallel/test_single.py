from ss.envs.ball_env import BallEnv
import mujoco_py
import time
import numpy as np
import timeit
import os
from ss.sandbox.parallel.config import TOTAL_STEPS

time_fn = lambda: os.times()[4]

env = BallEnv()
N_POOL = 1
N_STEPS = int(TOTAL_STEPS / N_POOL)
start = time_fn()
for _ in range(N_STEPS):
    for i in range(N_POOL):
        env.step(np.ones((2)))
end = time_fn()
elapsed_pool = end - start
sps_one = TOTAL_STEPS / elapsed_pool
print("time", elapsed_pool)
print("steps per second, one:", sps_one)
