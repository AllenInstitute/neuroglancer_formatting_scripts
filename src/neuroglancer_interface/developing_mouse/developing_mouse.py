# This script will attempt to load a unified dataset (template, CCF
# annotations, celltype heatmaps) into ome-zarr format

import argparse
import json
import pathlib
import shutil
import SimpleITK

from neuroglancer_interface.utils.data_utils import (
    create_root_group)

from neuroglancer_interface.modules.ccf_multiscale_annotations import (
    write_out_ccf)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    write_nii_to_group)

def print_status(msg):
    print(f"===={msg}====")

def  process_developing_mouse(
        config_path,
        n_processors,
        clobber,
        n_test,
        output_dir,
        transpose_ccf=False,
        tmp_dir=None,
        downscale_cutoff=64):

    with open(config_path, 'rb') as in_file:
        config_data = json.load(in_file)

    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)

    if output_dir is None:
        raise RuntimeError("must specify output_dir")

    if output_dir.exists():
        if not clobber:
            raise RuntimeError(
                f"{output_dir.resolve().absolute()} exists")
        else:
            print_status(f"Cleaning up {output_dir}")
            shutil.rmtree(output_dir)
            print_status("Done cleaning")

    output_dir.mkdir()

    if "ccf" in config_data:
        print_status("Formatting CCF annotations")

        if not isinstance(config_data['ccf']['segmentation'], list):
            raise RuntimeError("ccf.segmentation is not list")

        write_out_ccf(
            segmentation_path_list =[
                    pathlib.Path(p)
                    for p in config_data["ccf"]["segmentation"]],
            label_path=config_data["ccf"]["labels"],
            output_dir=output_dir/"ccf_annotations",
            use_compression=True,
            compression_blocksize=32,
            chunk_size=(64, 64, 64),
            do_transposition=transpose_ccf,
            tmp_dir=tmp_dir,
            downsampling_cutoff=downscale_cutoff)

        print_status("Done formatting CCF annotations")

    if "template" in config_data:
        print_status("Formatting avg template image")
        template_dir = output_dir/"avg_template"

        template_group = create_root_group(
           output_dir=template_dir)

        write_nii_to_group(
            root_group=template_group,
            group_name=None,
            nii_file_path=config_data['template']['template'],
            downscale_cutoff=downscale_cutoff,
            default_chunk=128,
            channel='red',
            do_transposition=transpose_ccf)

        print_status("Done formatting avg template image")

    if "boundary" in config_data:
        print_status("Formatting boundary image")
        template_dir = output_dir/"boundary"

        template_group = create_root_group(
           output_dir=template_dir)

        write_nii_to_group(
            root_group=template_group,
            group_name=None,
            nii_file_path=config_data['boundary']['template'],
            downscale_cutoff=downscale_cutoff,
            default_chunk=128,
            channel='red',
            do_transposition=transpose_ccf)

        print_status("Done formatting avg template image")



    if "cell_types" in config_data:
        print_status("Formatting mfish data")
        if n_test is not None:
            config_list = config_data["cell_types"][:n_test]
        else:
            config_list = config_data["cell_types"]
        cell_types_group = create_root_group(
            output_dir=output_dir/"cell_types")

        write_nii_file_list_to_ome_zarr(
            config_list=config_list,
            root_group=cell_types_group,
            n_processors=n_processors,
            downscale_cutoff=downscale_cutoff,
            default_chunk=128,
            do_transposition=False)

        print_status("Done formatting cell types data")

    print_status("Copying over config")
    dest_path = output_dir / 'config.json'
    assert not dest_path.exists()
    shutil.copy(config_path, dest_path)

    print_status("Done formatting all data")
    print(f"written to\n{output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=6)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--n_test', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--transpose_ccf', default=False, action='store_true')
    parser.add_argument('--tmp_dir', type=str, default=None)
    parser.add_argument('--downscale_cutoff', type=int, default=64)
    args = parser.parse_args()

    process_developing_mouse(
        config_path=args.config_path,
        n_processors=args.n_processors,
        clobber=args.clobber,
        n_test=args.n_test,
        output_dir=args.output_dir,
        transpose_ccf=args.transpose_ccf,
        tmp_dir=args.tmp_dir,
        downscale_cutoff=args.downscale_cutoff)

if __name__ == "__main__":
    main()
