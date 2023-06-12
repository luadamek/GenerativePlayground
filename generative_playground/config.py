import os
WORKDIR=os.path.split(os.path.split(__file__)[0])[0]
S3_DATA_DIR=os.getenv("S3_DATA_DIR", "s3://generate-faces/datasets/facesdataset.zip")

