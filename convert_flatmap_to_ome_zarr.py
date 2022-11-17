import pathlib
import argparse
from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    write_nii_to_group)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--downscale', type=int, default=2)
    args = parser.parse_args()

    assert args.output_dir is not None

    output_dir = pathlib.Path(args.output_dir)

    input_dir = pathlib.Path(
        "/allen/aibs/ccf/Flatmaps/maps/")
    assert input_dir.is_dir()

    data_to_group = dict()
    avg_path = input_dir /"average_template_10_RightDV_flatmap.nrrd"
    assert avg_path.is_file()
    #data_to_group[avg_path] = "avg_template"

    layer_dir = input_dir / "projections/flatmaps/layers"
    assert layer_dir.is_dir()

    layer_path_names = [n for n in layer_dir.rglob('e*.nrrd')]
    for pth in layer_path_names:
        params = pth.name.split('_')
        group_name = f"{params[0]}_{params[4]}"
        data_to_group[pth] = group_name

    file_path_list = []
    group_name_list = []
    for pth in data_to_group:
        file_path_list.append(pth)
        group_name_list.append(data_to_group[pth])

    root_group = write_nii_file_list_to_ome_zarr(
        file_path_list=file_path_list,
        group_name_list=group_name_list,
        output_dir=output_dir,
        downscale=args.downscale,
        clobber=args.clobber)

    write_nii_to_group(
        root_group=root_group,
        group_name="avg_template",
        nii_file_path=avg_path,
        downscale=2,
        transpose=False)


if __name__ == "__main__":
    main()
