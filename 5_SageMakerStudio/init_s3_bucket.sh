#!/bin/sh
aws cloudformation deploy --template-file=s3_bucket.yaml --stack-name=algotrading-data-s3 --capabilities=CAPABILITY_IAM
