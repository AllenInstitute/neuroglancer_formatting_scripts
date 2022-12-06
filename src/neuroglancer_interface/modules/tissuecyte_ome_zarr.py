import shutil
import pathlib
import multiprocessing

from neuroglancer_interface.classes.metadata_collectors import (
    BasicMetadataCollector)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)

from neuroglancer_interface.classes.downscalers import (
    XYZScaler)


def convert_tissuecyte_to_ome_zarr(
        input_dir=None,
        output_dir=None,
        downscale=2,
        n_processors=6):

    output_dir = pathlib.Path(output_dir)
    input_dir = pathlib.Path(input_dir)

    if not input_dir.is_dir():
        raise RuntimeError(
            "In convert_tissuecyte_to_ome_zarr, input_dir\n"
            f"{input_dir.resolve().absolute()}\n"
            "is not a dir")

    metadata_path = output_dir / 'metadata.json'

    metadata_collector = BasicMetadataCollector(
            metadata_output_path=metadata_path)

    mgr = multiprocessing.Manager()
    metadata_collector.set_lock(mgr.Lock())
    metadata_collector.metadata = mgr.dict()

    sub_dir_list = [n for n in input_dir.iterdir()
                    if n.is_dir()]

    fname_list = []
    group_name_list = []
    for sub_dir in sub_dir_list:
        sub_file_list = [n for n in sub_dir.iterdir()
                         if n.is_file() and n.name.endswith('nii.gz')]

        for sub_file in sub_file_list:
            if "_red" in sub_file.name:
                channel_color = "red"
            elif "_green" in sub_file.name:
                channel_color = "green"
            else:
                raise RuntimeError(
                    "Cannot get channel color from "
                    f"{sub_file.name}")
            group_name = f"{sub_dir.name}/{channel_color}"
            fname_list.append(sub_file)
            if group_name in group_name_list:
                raise RuntimeError(
                    f"Group {group_name} appears more than once")
            group_name_list.append(group_name)

    root_group = write_nii_file_list_to_ome_zarr(
        file_path_list=fname_list,
        group_name_list=group_name_list,
        output_dir=output_dir,
        downscale=downscale,
        clobber=False,
        n_processors=n_processors,
        metadata_collector=metadata_collector,
        DownscalerClass=XYZScaler)

    print("copying image_series_information.csv over")
    for sub_dir in sub_dir_list:
        src_path = sub_dir / "image_series_information.csv"
        if src_path.exists():
            dest_path = output_dir / src_path.parent.name / src_path.name
            shutil.copy(src_path, dest_path)

    metadata_collector.write_to_file()
