"""Interface to S3"""

import boto3
s3 = boto3.client('s3')
bucket_name = 'selfsupervised'
