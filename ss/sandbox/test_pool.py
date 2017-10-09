from ss.envs.ball_env import BallEnv
import mujoco_py
import time
import numpy as np

TOTAL_STEPS = 10000

env = BallEnv()
N_POOL = 1
N_STEPS = int(TOTAL_STEPS / N_POOL)
start = time.time()
for _ in range(N_STEPS):
    for i in range(N_POOL):
        env.step(np.ones((2)))
end = time.time()
elapsed_pool = end - start
sps_one = TOTAL_STEPS / elapsed_pool
print("steps per second, one:", sps_one)

for N_POOL in [1, 2, 4, 5, 10, 20, 50, 100]:
    print("pool:", N_POOL)
    N_STEPS = int(TOTAL_STEPS / N_POOL)

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
    sps_pool = TOTAL_STEPS / elapsed_pool
    print("time", elapsed_pool)
    print("steps per second, pool:", sps_pool)

    print("speedup:", sps_pool / sps_one)
