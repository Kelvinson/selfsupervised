from ss.algos.main import run_parallel
from ss.algos.params import get_params
import click

expname = "her_sweeplr1/"
paramlist = []
i = 0
for nb_train_steps in [0, 5, 25]:
    for pi_lr in [1e-2, 1e-3]:
        p = get_params(nb_train_steps=nb_train_steps,
                       pi_lr=pi_lr,
                       expname=expname+str(i))
        paramlist.append(p)

        i += 1

@click.command()
@click.argument("n", default=-1)
def run_exp_n(n):
    run_parallel(paramlist[n:n+1])

if __name__ == "__main__":
    run_exp_n()
