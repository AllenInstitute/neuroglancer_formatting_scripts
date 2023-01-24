import argparse
import pathlib
import json

from neuroglancer_interface.modules.jp2_to_ome_zarr import (
    convert_jp2_to_ome_zarr)

from neuroglancer_interface.utils.jp2_utils import _write_data_to_hdf5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_list_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--nz_min', default=None, type=int)
    parser.add_argument('--nz_max', default=None, type=int)
    parser.add_argument('--tmp_dir', default=None, type=str)
    parser.add_argument('--downscale_cutoff', type=int, default=2501)
    parser.add_argument('--default_chunk', type=int, default=512)
    args = parser.parse_args()

    config_list = json.load(open(args.config_list_path, "rb"))

    if args.nz_min is not None and args.nz_max is not None:
        nz_slice = (args.nz_min, args.nz_max)
    else:
        nz_slice = None

    convert_jp2_to_ome_zarr(
        config_list=config_list,
        output_dir=pathlib.Path(args.output_dir),
        clobber=args.clobber,
        nz_slice=nz_slice,
        tmp_dir=args.tmp_dir,
        downscale_cutoff=args.downscale_cutoff,
        default_chunk=args.default_chunk)

if __name__ == "__main__":
    main()
