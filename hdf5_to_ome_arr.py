from neuroglancer_interface.utils.data_utils import create_root_group
from neuroglancer_interface.modules.jp2_to_ome_zarr import (
   _convert_hdf5_to_ome_zarr)


import pathlib
import argparse

def main():
    #h5_path = pathlib.Path('/allen/aibs/technology/danielsf/highres_test_file.h5')
    default_h5_path = '/local1/highres_test_file.h5'
    default_output_dir = '/alen/ais/technology/danielsf/highres_compressed3'

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default=default_h5_path)
    parser.add_argument('--output_dir', type=str, default=default_output_dir)
    parser.add_argument('--n_processors', type=int, default=4)
    args = parser.parse_args()

    h5_path = pathlib.Path(args.h5_path)
    output_dir = args.output_dir

    assert h5_path.is_file()

    root_group = create_root_group(
        output_dir=output_dir,
        clobber=True)

    _convert_hdf5_to_ome_zarr(
        h5_path=h5_path,
        root_group=root_group,
        default_chunk=128,
        downscale_cutoff=125,
        n_processors=args.n_processors)

if __name__ == "__main__":
    main()
