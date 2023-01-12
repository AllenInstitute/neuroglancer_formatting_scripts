import argparse
import pathlib

from neuroglancer_interface.modules.jp2_to_ome_zarr import (
    convert_jp2_to_ome_zarr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jp2_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    convert_jp2_to_ome_zarr(
        jp2_path=pathlib.Path(args.jp2_path),
        output_dir=pathlib.Path(args.output_dir),
        clobber=args.clobber)

if __name__ == "__main__":
    main()
