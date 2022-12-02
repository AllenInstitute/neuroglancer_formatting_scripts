import argparse
import json
from neuroglancer_interface.modules.mfish_ome_zarr import (
    convert_mfish_to_ome_zarr)


def main():
    default_input = '/allen/programs/celltypes/workgroups/rnaseqanalysis/'
    default_input += 'mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_1/'
    default_input += 'alignment/mouse1_warpedToCCF_20220906'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=4)
    args = parser.parse_args()

    with open(args.config_path, "rb") as in_file:
        config_data = json.load(in_file)
        input_dir = config_data["input_dir"]
        output_dir = config_data["output_dir"]
        clobber = config_data["clobber"]
        downscale = config_data["downscale"]

    convert_mfish_to_ome_zarr(
        input_dir=input_dir,
        output_dir=output_dir,
        clobber=clobber,
        downscale=downscale,
        n_processors=args.n_processors)


if __name__ == "__main__":
    main()
