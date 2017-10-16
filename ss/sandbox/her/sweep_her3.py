from ss.algos.main import run_parallel
from ss.algos.params import get_params
import click

expname = "sweep_her3/"
paramlist = []
i = 0
for nb_train_steps in [5, 25]:
    for actor_lr in [1e-2, 1e-3]:
        for critic_lr in [1e-2, 1e-3]:
            for her in [True, False]:
                p = get_params(nb_train_steps=nb_train_steps,
                               actor_lr=actor_lr,
                               critic_lr=critic_lr,
                               her=her,
                               horizon=50,
                               expname=expname+str(i))
                paramlist.append(p)
                i += 1

@click.command()
@click.argument("n", default=-1)
def run_exp_n(n):
    run_parallel(paramlist[n:n+1])

if __name__ == "__main__":
    run_exp_n()
