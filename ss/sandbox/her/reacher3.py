from ss.algos.main import run_parallel
from ss.algos.params import get_params
import click
from ss.envs.reacher_env import Reacher7Dof

render = False
sync = True
expname = "reacher3/"

paramlist = []
i = 0
for noise_sigma in [0, 0.1, 0.2]:
    for seed in range(5):
        p = get_params(nb_train_steps=5,
                       actor_lr=0.001,
                       critic_lr=0.01,
                       her=True,
                       horizon=100,
                       expname=expname+str(i),
                       nb_epochs=5000,
                       render=render,
                       tau=0.99,
                       gamma=0.98,
                       env_type=Reacher7Dof,
                       batch_size=64,
                       noise_sigma=noise_sigma,
                       seed=seed,
                       sync=sync)
        paramlist.append(p)
        i += 1

@click.command()
@click.argument("n", default=-1)
def run_exp_n(n):
    run_parallel(paramlist[n:n+1])

if __name__ == "__main__":
    run_exp_n()
