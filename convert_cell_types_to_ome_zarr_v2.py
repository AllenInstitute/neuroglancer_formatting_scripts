import argparse
import json
from neuroglancer_interface.modules.cell_types_ome_zarr import (
    convert_cell_types_to_ome_zarr)

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

    if not isinstance(config_data["clobber"], bool):
        msg += f"clobber is {type(config_data['clobber'])}; "
        msg += "must be boolean\n"

    if len(msg) > 0:
        raise RuntimeError(msg)
    return config_data


def main():

    parser = argparse.ArgumentParser(
                "Convert cell types to ome-zarr in the case where "
                "each hierarchy level has its own directory")
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=4)
    args = parser.parse_args()

    config_data = read_config(args.config_path)

    convert_cell_types_to_ome_zarr(
        output_dir=config_data["output_dir"],
        input_list=config_cata["input_configs"],
        downscale=config_data["downscale"],
        clobber=config_data["clobber"],
        n_processors=args.n_processors)


if __name__ == "__main__":
    main()
