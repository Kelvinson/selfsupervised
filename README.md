# selfsupervised

Python environment: initialize env3. Add this root directory to PYTHONPATH.

Do `pip install -r requirements.txt` and `pip install -e vendor/...`

Make sure the paths in `ss/path.py` are fine.

# Docker instructions

Set up AWS IAM account and Docker account with credentials.

`docker build -t anair17/ss . ` (only on system code changes)

`docker push` (only on system code changes)

`python ss/remote/provision.py 6` (gets 6 AWS instances with Docker)

`python ss/remote/run.py ss/sandbox/her/sweep_lr_train_steps.py`

This should start training 6 experiments and copy experiment data back to S3.
