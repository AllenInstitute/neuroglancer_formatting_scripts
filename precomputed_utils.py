import gzip


def clean_dir(dir_path):
    """
    Unzip all of the files in dir_path
    """
    gzipped_path_list = [n for n in dir_path.rglob('*.gz')]
    for pth in gzipped_path_list:
        gunzip_file(pth)


def gunzip_file(src_path):
    """
    Unzip src_path, writing it to a file with the .gz removed from the end;
    delete src_path after unzipping
    """
    dest_path = src_path.parent / src_path.name.replace('.gz','')
    if dest_path.exists():
        raise RuntimeError(f"{dest_path} already exists")

    chunk_size = 100
    with open(dest_path, 'wb') as out_file:
        with gzip.open(src_path, 'rb') as in_file:
            keep_going = True
            while keep_going:
                chunk = in_file.read(chunk_size)
                if len(chunk) == 0:
                    keep_going = False
                else:
                    out_file.write(chunk)
    src_path.unlink()
