import argparse
import pathlib
import datetime

from neuroglancer_interface.utils.data_utils import (
    create_root_group)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    write_nii_to_group)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    input_list = [n for n in input_dir.iterdir() if n.is_dir()]

    for dataset in input_list:
        root_group = create_root_group(output_dir/dataset.name)
        write_nii_to_group(
            root_group=root_group,
            group_name='green',
            nii_file_path=dataset/'resampled_green.nii.gz',
            downscale_cutoff=64,
            default_chunk=128,
            channel='red',
            do_transposition=True,
            posxposz=True)

        write_nii_to_group(
            root_group=root_group,
            group_name='red',
            nii_file_path=dataset/'resampled_red.nii.gz',
            downscale_cutoff=64,
            default_chunk=128,
            channel='red',
            do_transposition=True,
            posxposz=True)

        print(f"wrote {dataset}")


if __name__ == "__main__":
    main()
