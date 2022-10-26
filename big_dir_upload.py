import time
import pathlib
import numpy as np
import argparse
import json
import os
import multiprocessing


def get_log(data_dir, log_path):
    if log_path.is_file():
        with open(log_path, 'rb') as in_file:
            return json.load(in_file)

    file_path_list = [n for n in data_dir.iterdir()]
    file_path_list.sort()
    log_data = dict()
    for file_pth in file_path_list:
        str_pth = str(file_pth.resolve().absolute())
        log_data[str_pth] = False
    return log_data


def save_log(log_path, log_data):
    with open(log_path, 'w') as out_file:
        out_file.write(json.dumps(dict(log_data), indent=2))

def print_timing(t0, ct, tot, prefix=None, log_path=None):
    duration = time.time()-t0
    if ct == 0:
        print(f"uploaded {ct} in {duration:.2e} seconds")
        return
    per = duration/ct
    pred = per*tot
    remain = pred-duration

    pred = pred/3600.0
    remain = remain/3600.0

    msg = f"{prefix} uploaded {ct} of {tot} in {duration:.2e} seconds; "
    msg += f"{remain:.2e} hrs remaining of {pred:.2e}"

    if log_path is not None:
        with open(log_path, "a") as out_file:
           out_file.write(msg + "\n")
    else:
        print(msg)


def _upload_files(
        file_path_list,
        data_dir,
        bucket_name,
        bucket_prefix,
        shared_log,
        timing_log_path,
        lock,
        thread_id):

    this_log = dict()
    t0 = time.time()
    ct_uploaded = 0
    to_upload = len(file_path_list)

    abs_dir = data_dir.resolve().absolute()
    for file_path in file_path_list:
        file_path = pathlib.Path(file_path)
        s3_dest = f"s3://{bucket_name}"
        if bucket_prefix is not None:
            s3_dest += f"/{bucket_prefix}"
        if file_path.is_dir():
            s3_cmd = f"aws s3 sync {file_path} {s3_dest}"
        else:
            s3_cmd = f"aws s3 cp {file_path} {s3_dest}"

        cmd_status = os.system(s3_cmd)
        if cmd_status != 0:
            break
        ct_uploaded += 1
        this_log[file_path] = True

        with lock:
            print_timing(
                t0=t0,
                ct=ct_uploaded,
                tot=to_upload,
                prefix=f"thread {thread_id}",
                log_path=timing_log_path)

    for file_path in this_log:
        shared_log[file_path] = this_log[file_path]

    with lock:
        print_timing(t0=t0, ct=ct_uploaded, tot=to_upload,
                     prefix=f"Ending thread {thread_id}",
                     log_path=timing_log_path)


def upload_files(
        data_dir,
        bucket_name,
        bucket_prefix,
        log_path,
        timing_log_path,
        n_processors=6):

    log_data = get_log(
                data_dir=data_dir,
                log_path=log_path)
    print("got log data")
    file_path_list = list(log_data.keys())
    file_path_list.sort()

    t0 = time.time()
    ct_uploaded = 0
    to_upload = 0
    files_to_upload = []
    for file_path in file_path_list:
        if not log_data[file_path]:
            files_to_upload.append(file_path)

    files_to_upload.sort()

    mgr = multiprocessing.Manager()
    lock = mgr.Lock()
    shared_log = mgr.dict()
    shared_log.update(log_data)

    sub_lists = []
    for ii in range(n_processors):
        sub_lists.append([])
    for ii in range(len(files_to_upload)):
        jj = ii % n_processors
        sub_lists[jj].append(files_to_upload[ii])

    process_list = []
    for ii in range(n_processors):
        p = multiprocessing.Process(
                target=_upload_files,
                kwargs={'file_path_list': sub_lists[ii],
                        'data_dir': data_dir,
                        'bucket_name': bucket_name,
                        'bucket_prefix': bucket_prefix,
                        'shared_log': shared_log,
                        'timing_log_path': timing_log_path,
                        'lock': lock,
                        'thread_id': ii})
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    save_log(log_path=log_path, log_data=shared_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--bucket', type=str, default=None)
    parser.add_argument('--bucket_prefix', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--timing_log', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=6)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    assert args.log is not None
    log_path = pathlib.Path(args.log)
    if args.clobber and log_path.is_file():
        log_path.unlink()
    data_dir = pathlib.Path(args.dir)
    assert data_dir.is_dir()

    if args.timing_log is not None:
        timing_log_path = pathlib.Path(args.timing_log)

    upload_files(data_dir=data_dir,
                 bucket_name=args.bucket,
                 bucket_prefix=args.bucket_prefix,
                 log_path=log_path,
                 timing_log_path=timing_log_path,
                 n_processors=args.n_processors)


if __name__ == "__main__":
    main()
