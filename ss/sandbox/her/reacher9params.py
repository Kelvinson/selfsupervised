from ss.algos.main import run_parallel
from ss.algos.params import get_params
import click
# from ss.envs.reacher_env import Reacher7Dof
Reacher7Dof = None

render = False
sync = True
expname = "reacher8/"

paramlist = []
i = 0
for nb_train_steps in [25, 50]:
    for actor_lr in [1e-3, 1e-4]:
        for critic_lr in [1e-3, 1e-4]:
            for tau in [0.8, 1.0]:
                for her in [True, False]:
                    for seed in [0, 1]:
                        p = get_params(nb_train_steps=nb_train_steps,
                                       actor_lr=actor_lr,
                                       critic_lr=critic_lr,
                                       her=her,
                                       horizon=50,
                                       expname=expname+str(i),
                                       nb_epochs=5000,
                                       render=render,
                                       tau=tau,
                                       gamma=0.98,
                                       env_type=Reacher7Dof,
                                       batch_size=128,
                                       noise_sigma=0.1,
                                       seed=seed,
                                       sync=sync)
                        paramlist.append(p)
                        i += 1

@click.command()
@click.argument("n", default=-1)
def run_exp_n(n):
    print("Launching experiment", n, "of", len(paramlist))
    run_parallel(paramlist[n:n+1])

if __name__ == "__main__":
    run_exp_n()
