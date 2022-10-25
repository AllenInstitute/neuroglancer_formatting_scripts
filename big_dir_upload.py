import time
import boto3
import pathlib
import argparse
import json


def get_log(data_dir, log_path):
    if log_path.is_file():
        with open(log_path, 'rb') as in_file:
            return json.load(in_file)

    file_path_list = [n for n in data_dir.rglob('*')]
    file_path_list.sort()
    log_data = dict()
    for file_pth in file_path_list:
        if file_pth.is_file():
            str_pth = str(file_pth.resolve().absolute())
            log_data[str_pth] = False
    return log_data


def save_log(log_path, log_data):
    with open(log_path, 'w') as out_file:
        out_file.write(json.dumps(log_data, indent=2))

def print_timing(t0, ct, tot):
    duration = time.time()-t0
    if ct == 0:
        print(f"uploaded {ct} in {duration:.2e} seconds")
        return
    per = duration/ct
    pred = per*tot
    remain = pred-duration

    pred = pred/3600.0
    remain = remain/3600.0
    print(f"uploaded {ct} of {tot} in {duration:.2e} seconds; "
          f"{remain:.2e} hrs remaining of {pred:.2e}")


def upload_files(
        data_dir,
        bucket_name,
        log_path):

    log_data = get_log(
                data_dir=data_dir,
                log_path=log_path)
    print("got log data")
    file_path_list = list(log_data.keys())
    file_path_list.sort()

    s3_client = boto3.client('s3')

    t0 = time.time()
    ct_uploaded = 0
    to_upload = 0
    for file_path in file_path_list:
        if not log_data[file_path]:
            to_upload += 1

    abs_dir = data_dir.resolve().absolute()
    try:
        for file_path in file_path_list:
            if log_data[file_path]:
                continue

            s3_key_path = pathlib.Path(file_path).relative_to(abs_dir)
            s3_key = str(s3_key_path)
            print(f"{file_path} -> {s3_key}")
            with open(file_path, 'rb') as data:
                s3_client.upload_fileobj(
                    Fileobj=data,
                    Bucket=bucket_name,
                    Key=s3_key)
            log_data[file_path] = True
            ct_uploaded += 1
            if True: #ct_uploaded % 100 == 0:
                print_timing(t0=t0, ct=ct_uploaded, tot=to_upload)
    finally:
        save_log(log_path=log_path, log_data=log_data)
        duration = time.time()-t0
        print_timing(t0=t0, ct=ct_uploaded, tot=to_upload)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--bucket', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    assert args.log is not None
    log_path = pathlib.Path(args.log)
    if args.clobber and log_path.is_file():
        log_path.unlink()
    data_dir = pathlib.Path(args.dir)
    assert data_dir.is_dir()

    upload_files(data_dir=data_dir,
                 bucket_name=args.bucket,
                 log_path=log_path)


if __name__ == "__main__":
    main()
