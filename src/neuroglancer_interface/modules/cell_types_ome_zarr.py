import pathlib
import time
import numpy as np

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    create_root_group)

from neuroglancer_interface.utils.celltypes_utils import (
    read_manifest)


def convert_cell_types_to_ome_zarr(
        output_dir: str,
        input_list: list,
        downscale: int,
        clobber: bool,
        n_processors: int):
    """
    output_dir -- e.g. mouse_5/cell_types

    input_list -- list of dicts with
        {'output_prefix': 'Level_N',
         'input_dir': 'my/data/dir/level_n_id/'}

    downscale -- factor by which to downscale image at each step

    clobber -- should probably always be False in bundle
    """
    root_group = create_root_group(
                    output_dir=output_dir,
                    clobber=clobber)

    for input_config in input_list:
        input_dir = input_config["input_dir"]
        prefix = input_config["output_prefix"]
        write_sub_group(
            root_group=root_group,
            input_dir=input_dir,
            prefix=prefix,
            n_processors=args.n_processors,
            downscale=downscale)


def write_sub_group(
        root_group=None,
        input_dir=None,
        prefix=None,
        n_processors=4,
        downscale=2):

    input_dir = pathlib.Path(input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"{input_dir.resolve().absolute()}\n"
                           "is not a dir")

    fpath_list = [n for n in input_dir.rglob('*.nii.gz')
                  if n.is_file()][:10]

    manifest_path = input_dir / 'manifest.csv'
    if not manifest_path.is_file():
        raise RuntimeError(
            f"could not find\n{manifest_path.resolve().absolute()}")

    name_lookup = read_manifest(manifest_path)
    cluster_name_list = [name_lookup[n.name]["machine_readable"]
                         for n in fpath_list]

    print(f"writing {prefix}")
    root_group = write_nii_file_list_to_ome_zarr(
            file_path_list=fpath_list,
            group_name_list=cluster_name_list,
            output_dir=None,
            root_group=root_group,
            n_processors=n_processors,
            clobber=False,
            prefix=prefix,
            downscale=downscale)

    print(f"done writing {prefix}")
