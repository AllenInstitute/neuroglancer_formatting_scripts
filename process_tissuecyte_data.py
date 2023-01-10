# This script will attempt to load a unified dataset (template, CCF
# annotations, tissuecyte nrrd files) into ome-zarr format

import argparse
import json
import pathlib
import shutil

from neuroglancer_interface.modules.ccf_multiscale_annotations import (
    write_out_ccf)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)

from neuroglancer_interface.modules.tissuecyte_ome_zarr import (
    convert_tissuecyte_to_ome_zarr)

from neuroglancer_interface.modules.cell_types_ome_zarr import (
    convert_cell_types_to_ome_zarr)

from neuroglancer_interface.utils.census_utils import (
    get_structure_name_lookup,
    get_mask_lookup,
    create_census)

from neuroglancer_interface.classes.downscalers import (
    XYZScaler)


def print_status(msg):
    print(f"===={msg}====")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=6)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'rb') as in_file:
        config_data = json.load(in_file)

    output_dir = pathlib.Path(config_data['output_dir'])
    if output_dir.exists():
        if not args.clobber:
            raise RuntimeError(
                f"{output_dir.resolve().absolute()} exists")
        else:
            print_status(f"Cleaning up {output_dir}")
            shutil.rmtree(output_dir)
            print_status("Done cleaning")

    output_dir.mkdir()

    if "ccf" in config_data:
        print_status("Formatting CCF annotations")

        write_out_ccf(
            segmentation_path_list =[
                    pathlib.Path(p)
                    for p in config_data["ccf"]["segmentation"]],
            label_path=pathlib.Path(config_data["ccf"]["labels"]),
            output_dir=output_dir/"ccf_annotations")

        print_status("Done formatting CCF annotations")

    if "template" in config_data:
        print_status("Formatting avg template image")
        write_nii_file_list_to_ome_zarr(
            file_path_list=[pathlib.Path(config_data["template"]["template"])],
            group_name_list=[None],
            output_dir=output_dir/"avg_template",
            downscale=config_data["downscale"],
            n_processors=1,
            clobber=False,
            DownscalerClass=XYZScaler)
        print_status("Done formatting avg template image")

    if "tissuecyte" in config_data:
        print_status("Formatting tissuecyte data")
        convert_tissuecyte_to_ome_zarr(
            input_dir=config_data["tissuecyte"]["input_dir"],
            output_dir=output_dir/"tissuecyte",
            downscale=config_data["downscale"],
            n_processors=args.n_processors)
        print_status("Done formatting tissuecyte data")

    print_status("Done formatting all data")
    print(f"written to\n{config_data['output_dir']}")

if __name__ == "__main__":
    main()
