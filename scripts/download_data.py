import boto3
import os
from generative_playground.config import WORKDIR, S3_DATA_DIR
import shutil

def get_bucket_key(s3_uri):
    stripped_data_dir = s3_uri.replace("s3://", "")
    split_data_dir = stripped_data_dir.split("/")
    bucket = split_data_dir[0]
    key = "/".join(split_data_dir[1:])
    return bucket, key

bucket, key = get_bucket_key(S3_DATA_DIR)

s3_client = boto3.resource("s3")
bucket_resource = s3_client.Bucket("generate-faces")

local_data_dir = os.path.join(WORKDIR, "datasets", "raw")
local_data_file = os.path.join(local_data_dir, "faces_data.zip")

if not os.path.exists(local_data_dir):
    os.mkdirs(local_data_dir)

print(f"Downloading {key} to {local_data_file}")
bucket_resource.download_file(key, local_data_file)

# unzip the archive
local_unzipped_dir = os.path.join(WORKDIR, "datasets", "unzipped")
if not os.path.exists(local_unzipped_dir):
    os.mkdirs(local_unzipped_dir)

shutil.unpack_archive(local_data_file, local_unzipped_dir)

