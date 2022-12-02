# This script will attempt to load a unified dataset (template, CCF
# annotations, mfish heatmaps, celltype heatmaps) into ome-zarr format

import argparse
import json
import pathlib
import shutil

from neuroglancer_interface.modules.ccf_annotation_formatting import (
    format_ccf_annotations)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)

from neuroglancer_interface.modules.mfish_ome_zarr import (
    convert_mfish_to_ome_zarr)

from neuroglancer_interface.modules.cell_types_ome_zarr import (
    convert_cell_types_to_ome_zarr)

from neuroglancer_interface.utils.census_utils import (
    get_structure_name_lookup,
    get_mask_lookup)


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
        format_ccf_annotations(
            annotation_path=config_data["ccf"]["labels"],
            segmentation_path=config_data["ccf"]["segmentation"],
            output_dir=output_dir/"ccf_annotations",
            clobber=False)
        print_status("Done formatting CCF annotations")

    if "template" in config_data:
        print_status("Formatting avg template image")
        write_nii_file_list_to_ome_zarr(
            file_path_list=[pathlib.Path(config_data["template"]["template"])],
            group_name_list=[None],
            output_dir=output_dir/"avg_template",
            downscale=config_data["downscale"],
            n_processors=1,
            clobber=False)
        print_status("Done formatting avg template image")

    if "census" in config_data:
        print_status("Reading structure masks for census")
        #this_dir = pathlib.Path(__file__).parent
        #onto_dir = this_dir / "data/ontology_parcellation"
        #structure_name_lookup = get_structure_name_lookup(
        #    path_list = [
        #        onto_dir/"1_adult_mouse_brain_graph.json",
        #        onto_dir/"structure_sets.csv"])

        structure_set_masks = get_mask_lookup(
                mask_dir=config_data["census"]["structure_set_masks"],
                n_processors=args.n_processors)

        structure_masks = get_mask_lookup(
                mask_dir=config_data["census"]["structure_masks"],
                n_processors=args.n_processors)

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
            structure_masks=structure_masks)
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
            structure_masks=structure_masks)
        print_status("Done formatting cell types data")

    print_status("Done formatting all data")
    print(f"written to\n{config_data['output_dir']}")

if __name__ == "__main__":
    main()
