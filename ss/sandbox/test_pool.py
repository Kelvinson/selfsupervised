from ss.envs.ball_env import BallEnv
import mujoco_py
import time
import numpy as np

N_POOL = 100
N_STEPS = 1000

envs = []
sims = []
for i in range(N_POOL):
    env = BallEnv()
    envs.append(env)
    sims.append(env.sim)
pool = mujoco_py.MjSimPool(sims)

start = time.time()
for _ in range(N_STEPS):
    for i in range(N_POOL):
        envs[i].set_action(np.ones((2)))
    pool.step()
    pool.forward()
    # envs[0].render()
end = time.time()
elapsed_pool = end - start
sps_pool = N_STEPS * N_POOL / elapsed_pool
print("steps per second, pool:", sps_pool)

start = time.time()
for _ in range(N_STEPS):
    env.step(np.ones((2)))
end = time.time()
elapsed_pool = end - start
sps_one = N_STEPS / elapsed_pool
print("steps per second, one:", sps_one)

print("speedup:", sps_pool / sps_one)
