import pathlib
import argparse
import time
import numpy as np
import json

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    create_root_group)


def read_config(config_path):

    with open(config_path, "rb") as in_file:
        config_data = json.load(in_file)

    msg = ""
    if "output_dir" not in config_data:
        msg += "Must specify output_dir\n"
    if "input_configs" not in config_data:
        msg += "Must specify input_configs\n"
    if "clobber" not in config_data:
        config_data["clobber"] = False
    if "downscale" not in config_data:
        config_data["downscale"] = 2
    if "n_processors" not in config_data:
        config_data["n_processors"] = 4

    if not isinstance(config_data["clobber"], bool):
        msg += f"clobber is {type(config_data['clobber']}; "
        msg += "must be boolean\n"

    if len(msg) > 0:
        raise RuntimeError(msg)
    return config_data


def write_sub_group(
        root_group=None,
        input_dir=None,
        prefix=None,
        n_processors=4,
        downscale=2):

    input_dir = pathlib.Path(input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"{input_dir.resolve().absolute()}\n"
                           "is not a dir")

    fpath_list = [n for n in input_dir.rglob('*.nii.gz')
                  if n.is_file()]

    cluster_name_list = [
            pth.name.replace('.nii.gz', '')
            for pth in fpath_list]

    print(f"writing {prefix}")
    root_group = write_nii_file_list_to_ome_zarr(
            file_path_list=fpath_list,
            group_name_list=cluster_name_list,
            output_dir=None,
            downscale=downscale,
            n_processors=n_processors,
            clobber=False,
            prefix=prefix,
            downscale=downscale)

    print(f"done writing {prefix}")


def main():

    parser = argparse.ArgumentParser(
                "Convert cell types to ome-zarr in the case where "
                "each hierarchy level has its own directory")
    parser.add_argument('--config_path', type=str, default=None)
    args = parser.parse_args()

    config_data = read_config(args.config_path)

    output_dir = pathlib.Path(config_data["output_dir"])

    root_group = create_root_group(
                    output_dir=config_data["output_dir"],
                    clobber=config_data["clobber"])

    for input_config in config_data["input_configs"]:
        input_dir = input_config["input_dir"]
        prefix = input_config["output_prefix"]
        write_sub_group(
            root_group=root_group,
            input_dir=input_dir,
            prefix=prefix,
            n_processors=config_data["n_processors"],
            downscale=config_data["downscale"])


if __name__ == "__main__":
    main()