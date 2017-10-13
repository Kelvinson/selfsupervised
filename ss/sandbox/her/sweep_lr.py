from ss.algos.main import run_parallel
from ss.algos.params import get_params

expname = "her_sweeplr1/"
paramlist = []
i = 0
for nb_train_steps in [0]:
    for pi_lr in [1e-2]:
        p = get_params(nb_train_steps=nb_train_steps,
                       pi_lr=pi_lr,
                       expname=expname+str(i))
        paramlist.append(p)

        i += 1

run_parallel(paramlist)
