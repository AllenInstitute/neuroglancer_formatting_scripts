# This script will attempt to load a unified dataset (template, CCF
# annotations, mfish heatmaps, celltype heatmaps) into ome-zarr format

import argparse
import json
import pathlib
import shutil
import SimpleITK

from neuroglancer_interface.modules.ccf_annotation_formatting import (
    format_ccf_annotations)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)

from neuroglancer_interface.modules.mfish_ome_zarr import (
    convert_mfish_to_ome_zarr)

from neuroglancer_interface.modules.cell_types_ome_zarr import (
    convert_cell_types_to_ome_zarr)

from neuroglancer_interface.utils.data_utils import (
    get_array_from_img)

from neuroglancer_interface.utils.census_utils import (
    get_structure_name_lookup,
    get_mask_lookup,
    create_census)

from neuroglancer_interface.utils.census_conversion import (
    convert_census_to_hdf5)


def print_status(msg):
    print(f"===={msg}====")


def get_n_slices(config_data):
    eg_dir = pathlib.Path(config_data["cell_types"]["input_list"][0]["input_dir"])
    fname_list = [n for n in eg_dir.rglob("*.nii.gz")]
    img = SimpleITK.ReadImage(fname_list[0])
    arr = get_array_from_img(img)
    n_slices = arr.shape[2]
    return n_slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=6)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--n_test', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--only_metadata', default=False, action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        raise RuntieError("must specify output_dir")

    with open(args.config_path, 'rb') as in_file:
        config_data = json.load(in_file)

    output_dir = pathlib.Path(args.output_dir)
    if output_dir.exists():
        if not args.clobber:
            raise RuntimeError(
                f"{output_dir.resolve().absolute()} exists")
        else:
            print_status(f"Cleaning up {output_dir}")
            shutil.rmtree(output_dir)
            print_status("Done cleaning")

    output_dir.mkdir()

    if "ccf" in config_data and not args.only_metadata:
        print_status("Formatting CCF annotations")
        format_ccf_annotations(
            annotation_path=config_data["ccf"]["labels"],
            segmentation_path=config_data["ccf"]["segmentation"],
            output_dir=output_dir/"ccf_annotations",
            clobber=False)
        print_status("Done formatting CCF annotations")

    if "template" in config_data and not args.only_metadata:
        print_status("Formatting avg template image")
        write_nii_file_list_to_ome_zarr(
            file_path_list=[pathlib.Path(config_data["template"]["template"])],
            group_name_list=[None],
            output_dir=output_dir/"avg_template",
            downscale=config_data["downscale"],
            n_processors=1,
            clobber=False)
        print_status("Done formatting avg template image")

    if "max_counts" in config_data and not args.only_metadata:
        print_status("Formatting max count image")
        write_nii_file_list_to_ome_zarr(
            file_path_list=[pathlib.Path(config_data["max_counts"]["path"])],
            group_name_list=[None],
            output_dir=output_dir/"max_count_image",
            downscale=config_data["downscale"],
            n_processors=1,
            clobber=False)
        print_status("Done formatting max count image")

    do_census = False
    if "census" in config_data:
        print_status("Reading structure masks for census")
        do_census = True
        structure_name_lookup = dict()
        this_dir = pathlib.Path(__file__).parent
        onto_dir = this_dir / "data/ontology_parcellation"

        structure_name_lookup["structures"] = get_structure_name_lookup(
            path_list = [onto_dir / "1_adult_mouse_brain_graph.json"])
        structure_name_lookup["structure_sets"] = get_structure_name_lookup(
            path_list = [onto_dir / "structure_sets.csv"])

        structure_set_masks = get_mask_lookup(
                mask_dir=config_data["census"]["structure_set_masks"],
                n_processors=args.n_processors,
                n_test=args.n_test)

        structure_masks = get_mask_lookup(
                mask_dir=config_data["census"]["structure_masks"],
                n_processors=args.n_processors,
                n_test=args.n_test)

        print_status("Done reading structure masks for census")
    else:
        structure_name_lookup = None
        structure_set_masks = None
        structure_masks = None

    if "mfish" in config_data:
        print_status("Formatting mfish data")
        convert_mfish_to_ome_zarr(
            input_dir=config_data["mfish"]["input_dir"],
            output_dir=output_dir/"mfish_heatmaps",
            clobber=False,
            downscale=config_data["downscale"],
            n_processors=args.n_processors,
            structure_set_masks=structure_set_masks,
            structure_masks=structure_masks,
            n_test=args.n_test,
            only_metadata=args.only_metadata)
        print_status("Done formatting mfish data")

    if "cell_types" in config_data:
        print_status("Formatting cell types data")
        convert_cell_types_to_ome_zarr(
            output_dir=output_dir/"cell_types",
            input_list=config_data["cell_types"]["input_list"],
            downscale=config_data["downscale"],
            clobber=False,
            n_processors=args.n_processors,
            structure_set_masks=structure_set_masks,
            structure_masks=structure_masks,
            n_test=args.n_test,
            only_metadata=args.only_metadata)
        print_status("Done formatting cell types data")

    if do_census:
        print_status("Gathering census")
        census_json_path = output_dir / "census.json"
        if census_json_path.exists():
            raise RuntimeError(
                f"{census_path} exists")
        census = create_census(
                    dataset_dir=output_dir,
                    structure_name_lookup=structure_name_lookup)
        census['structure_masks'] = dict()
        for k in structure_masks:
            human_name = structure_name_lookup['structures'][k]
            assert human_name not in census['structure_masks']
            census['structure_masks'][human_name] = structure_masks[k]['path']
        census['structure_set_masks'] = dict()
        for k in structure_set_masks:
            human_name = structure_name_lookup['structure_sets'][k]
            assert human_name not in census['structure_set_masks']
            census['structure_set_masks'][human_name] = structure_set_masks[k]['path']

        with open(census_json_path, "w") as out_file:
            out_file.write(json.dumps(census, indent=2))
        print_status("Done gathering census")

        print_status("Converting census to HDF5")

        # hack to get n_slices
        n_slices = get_n_slices(config_data)

        census_h5_path = output_dir / 'census_h5.h5'
        convert_census_to_hdf5(
            input_path=census_json_path,
            output_path=census_h5_path,
            n_slices=n_slices,
            clobber=False)
        print_status(f"Done converting census to HDF5 -- n_slices {n_slices}")

    print_status("Copying over config")
    dest_path = output_dir / 'config.json'
    assert not dest_path.exists()
    shutil.copy(args.config_path, dest_path)

    print_status("Done formatting all data")
    print(f"written to\n{output_dir}")

if __name__ == "__main__":
    main()
