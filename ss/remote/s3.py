"""Interface to S3"""

import boto3
from subprocess import Popen
import ss.path

# s3 = boto3.client('s3')
# bucket = s3.Bucket('selfsupervised')

def sync_up_expdir():
    d = ss.path.EXPDIR
    cmd = "aws s3 sync %s s3://selfsupervised/experiments/" % d
    cmd_list = cmd.split(" ")
    try:
        p = Popen(cmd_list).wait()
    except:
        print("Failed to sync!....")

if __name__ == "__main__":
    sync_up_expdir()
