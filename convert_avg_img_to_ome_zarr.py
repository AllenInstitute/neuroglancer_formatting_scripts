import pathlib
import argparse
from data_utils import write_nii_file_list_to_ome_zarr


def main():
    default_input = '/allen/programs/celltypes/workgroups/rnaseqanalysis/'
    default_input += 'mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_1/'
    default_input += 'RegPrelimDefNN_mouse1/iter0/'
    default_input += 'mouse1_CCFAtlasGlobWarped.nii.gz'

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_group', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=default_input)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--downscale', type=int, default=2)
    args = parser.parse_args()

    assert args.output_dir is not None
    assert args.input_file is not None

    output_dir = pathlib.Path(args.output_dir)
    input_file = pathlib.Path(args.input_file)

    assert input_file.is_file()

    write_nii_file_list_to_ome_zarr(
        file_path_list=input_file,
        group_name_list=args.output_group,
        output_dir=output_dir,
        downscale=args.downscale,
        clobber=args.clobber)


if __name__ == "__main__":
    main()
