from typing import List
import boto3
import multiprocessing
import json
import pathlib
import argparse
import warnings
import time


def create_bucket(
        bucket_name,
        allow_already_existing=False):
    """
    Create an S3 bucket and configure it to serve data to neuroglancer

    Parameters
    ----------
    bucket_name: str
        The name of the bucket to create

    allow_already_existing: bool
        If True, will emit a warning if bucket_name already exists,
        but proceed to configure the bucket anyway. If False, will
        raise an error if bucket_name already exists.

        Regardless, if the bucket already exists, this method
        will not do anything (i.e. it will not try to put
        configs in a bucket it did not just create).
    """

    this_bucket_name = bucket_name
    s3 = boto3.client('s3')
    bucket_list = s3.list_buckets()
    already_extant = set()
    for bucket in bucket_list['Buckets']:
        already_extant.add(bucket['Name'])

    if this_bucket_name in already_extant:
        msg = f"Bucket {this_bucket_name} already exists\n"
        if allow_already_existing:
            msg += "Assuming it is already properly configured and continuing"
            warnings.warn(msg)
            return None
        else:
            raise RuntimeError(msg)
    s3.create_bucket(
        Bucket=this_bucket_name,
        ACL='public-read')

    s3.put_bucket_policy(
        Bucket=this_bucket_name,
        Policy=json.dumps(_create_policy(bucket_name)))

    s3.put_bucket_cors(
        Bucket=this_bucket_name,
        CORSConfiguration=_create_cors())

    return None


def upload_to_bucket(
        dir_list: List[pathlib.Path],
        bucket_name: str,
        bucket_prefix: str,
        n_processors: int):
    """
    Upload ome-zarr data to bucket.

    Parameters
    ----------
    dir_list: List[pathlib.Path]
        list of directories whose contents need to be uploaded
        to S3

    bucket_name: str
        Name of the bucket to which the data will be uploaded

    bucket_prefix: str
        The parent directory in which all of the data objects will
        live.

    n_processors: int
        Number of multiprocessing processes to use
    """
    t0 = time.time()
    print(f"uploading data to S3 bucket {bucket_name}/{bucket_prefix}")

    full_path_list = []
    full_key_list = []
    for parent_path in dir_list:
        data_path_list = [n for n in parent_path.rglob('**/*')
                          if n.is_file()]
        for pth in data_path_list:
            target_key = str(pth.relative_to(parent_path.parent))
            target_key = target_key.replace('\\','/')
            target_key = f"{bucket_prefix}/{target_key}"
            full_path_list.append(pth)
            full_key_list.append(target_key)

    path_sub_lists = []
    key_sub_lists = []
    for ii in range(n_processors):
        path_sub_lists.append([])
        key_sub_lists.append([])
    for ii in range(len(full_path_list)):
        jj = ii % n_processors
        path_sub_lists[jj].append(full_path_list[ii])
        key_sub_lists[jj].append(full_key_list[ii])

    process_list = []
    for pth_list, k_list in zip(path_sub_lists, key_sub_lists):
        p = multiprocessing.Process(
                target=_upload_to_bucket,
                kwargs={"bucket_name": bucket_name,
                        "file_path_list": pth_list,
                        "key_list": k_list})
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    duration = time.time()-t0
    print(f"uploaded data in {duration:.2e} seconds")


def _upload_to_bucket(
        bucket_name: str,
        file_path_list: List[pathlib.Path],
        key_list: List[str]):
    """
    Upload a list of files to an S3 bucket.

    Parameters
    ----------
    bucket_name: str
        Name of the bucket

    file_path_list: List[pathlib.Path]
        List of paths to the files to upload

    key_list: List[str]
        List of keys to which to upload the files
        (in same order as file_path_list)
    """
    s3 = boto3.client('s3')
    for pth, target_key in zip(file_path_list, key_list):
        with open(pth, 'rb') as data:
            s3.upload_fileobj(
                Bucket=bucket_name,
                Fileobj=data,
                Key=target_key)


def _create_cors():
    """
    Returns the CORS rules required to serve neuroglancer
    data out of S3
    """
    cors = {"CORSRules": [{
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "GET"
        ],
        "AllowedOrigins": [
            "*"
        ],
        "ExposeHeaders": [],
        "MaxAgeSeconds": 600
    }]}

    return cors


def _create_policy(
        bucket_name):
    """
    Parameters
    ----------
    bucket_name: str
        The name of the bucket

    Returns
    -------
    policy: dict
        The policy object needed to configure a bucket
        to serve data to neuroglancer.
    """
    policy = {
        "Version": "2008-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket_name}/*"
            },
            {
                "Sid": "PublicReadGetObjectVersion",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObjectVersion",
                "Resource": f"arn:aws:s3:::{bucket_name}/*"
            },
            {
                "Sid": "PublicListObjectVersions",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:ListBucketVersions",
                "Resource": f"arn:aws:s3:::{bucket_name}"
            },
            {
                "Sid": "PublicListObjects",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:ListBucket",
                "Resource":  f"arn:aws:s3:::{bucket_name}"
            },
            {
                "Sid": "PublicLocation",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetBucketLocation",
                "Resource": f"arn:aws:s3:::{bucket_name}"
            }
        ]
    }
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()
    assert args.name is not None
    create_bucket(bucket_name=args.name)


if __name__ == "__main__":
    main()
