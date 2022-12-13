import pathlib
import json
import time
import numpy as np
import shutil
import multiprocessing

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr,
    create_root_group)

from neuroglancer_interface.utils.celltypes_utils import (
    read_manifest,
    read_list_of_manifests,
    desanitizer_from_meta_manifest)

from neuroglancer_interface.classes.metadata_collectors import (
    CellTypeMetadataCollector)

def convert_cell_types_to_ome_zarr(
        output_dir: str,
        input_list: list,
        downscale: int,
        clobber: bool,
        n_processors: int,
        structure_set_masks=None,
        structure_masks=None,
        n_test=None,
        only_metadata=False):
    """
    output_dir -- e.g. mouse_5/cell_types

    input_list -- list of dicts with
        {'output_prefix': 'Level_N',
         'input_dir': 'my/data/dir/level_n_id/'}

    downscale -- factor by which to downscale image at each step

    clobber -- should probably always be False in bundle
    """

    list_of_manifests = []
    for input_config in input_list:
        input_dir = pathlib.Path(input_config["input_dir"])
        manifest_path = input_dir / "manifest.csv"
        list_of_manifests.append(manifest_path)

    meta_manifest = read_list_of_manifests(list_of_manifests)
    desanitizer = desanitizer_from_meta_manifest(meta_manifest)

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
            n_processors=n_processors,
            downscale=downscale,
            structure_set_masks=structure_set_masks,
            structure_masks=structure_masks,
            n_test=n_test,
            only_metadata=only_metadata)


def write_sub_group(
        root_group=None,
        input_dir=None,
        prefix=None,
        n_processors=4,
        downscale=2,
        structure_set_masks=None,
        structure_masks=None,
        n_test=None,
        only_metadata=False):


    input_dir = pathlib.Path(input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"{input_dir.resolve().absolute()}\n"
                           "is not a dir")

    fpath_list = [n for n in input_dir.rglob('*.nii.gz')
                  if n.is_file()]

    if n_test is not None:
        fpath_list = fpath_list[:n_test]

    manifest_path = input_dir / 'manifest.csv'
    if not manifest_path.is_file():
        raise RuntimeError(
            f"could not find\n{manifest_path.resolve().absolute()}")

    name_lookup = read_manifest(manifest_path)
    cluster_name_list = [name_lookup[n.name]["machine_readable"]
                         for n in fpath_list]


    output_dir = pathlib.Path(root_group.store.path)
    metadata_path = output_dir / f'{prefix}/metadata.json'
    metadata_h5_path = output_dir / f'{prefix}/per_slice_counts.h5'
    metadata_collector = CellTypeMetadataCollector(
                            metadata_output_path=metadata_path,
                            h5_output_path=metadata_h5_path,
                            structure_set_masks=structure_set_masks,
                            structure_masks=structure_masks)

    mgr = multiprocessing.Manager()
    metadata_collector.set_lock(mgr.Lock())
    metadata_collector.metadata = mgr.dict()

    print(f"writing {prefix}")
    root_group = write_nii_file_list_to_ome_zarr(
            file_path_list=fpath_list,
            group_name_list=cluster_name_list,
            output_dir=None,
            root_group=root_group,
            n_processors=n_processors,
            clobber=False,
            prefix=prefix,
            downscale=downscale,
            metadata_collector=metadata_collector,
            only_metadata=only_metadata)

    print("copying manifest over")
    output_dir = pathlib.Path(root_group.store.path)
    if prefix is not None:
        output_dir = output_dir / prefix
    new_manifest_path = output_dir / 'manifest.csv'
    if new_manifest_path.exists():
        raise RuntimeError(f"{new_manifest_path} already exists")
    shutil.copy(manifest_path, new_manifest_path)

    metadata_collector.write_to_file()

    print(f"done writing {prefix}")
