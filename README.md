# selfsupervised

Python environment: initialize env3. Add this root directory to PYTHONPATH.

Do `pip install -r requirements.txt` and `pip install -e vendor/...`

Make sure the paths in `ss/path.py` are fine.

# Docker instructions

Set up AWS IAM account and Docker account with credentials.

`docker build -t anair17/ss . ` (only on system code changes)

`docker push` (only on system code changes)

`python ss/remote/provision.py`

`python ss/remote/run.py`
