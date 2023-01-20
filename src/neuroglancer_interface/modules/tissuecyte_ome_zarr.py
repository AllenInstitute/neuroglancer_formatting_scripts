import shutil
import json
import pathlib
import multiprocessing

from neuroglancer_interface.classes.metadata_collectors import (
    BasicMetadataCollector)

from neuroglancer_interface.utils.data_utils import (
    write_nii_file_list_to_ome_zarr)

from neuroglancer_interface.classes.downscalers import (
    XYZScaler)

from neuroglancer_interface.classes.nifti_array import (
    get_nifti_obj)


def convert_tissuecyte_to_ome_zarr(
        input_path=None,
        output_dir=None,
        downscale=2,
        n_processors=6,
        chunk_size=128):

    output_dir = pathlib.Path(output_dir)
    input_path = pathlib.Path(input_path)

    metadata_path = output_dir / 'metadata.json'

    metadata_collector = BasicMetadataCollector(
            metadata_output_path=metadata_path)

    mgr = multiprocessing.Manager()
    metadata_collector.set_lock(mgr.Lock())
    metadata_collector.metadata = mgr.dict()

    fname_list = []
    group_name_list = []
    channel_list = []
    sub_dir_list = None
    if input_path.is_dir():
        sub_dir_list = [n for n in input_path.iterdir()
                        if n.is_dir()]

        for sub_dir in sub_dir_list:
            sub_file_list = [n for n in sub_dir.iterdir()
                             if n.is_file() and n.name.endswith('nii.gz')]

            for sub_file in sub_file_list:
                if "_red" in sub_file.name:
                    channel_color = "red"
                elif "_green" in sub_file.name:
                    channel_color = "green"
                elif "_blue" in sub_file.name:
                    channel_color = "blue"
                else:
                    raise RuntimeError(
                        "Cannot get channel color from "
                        f"{sub_file.name}")
                group_name = f"{sub_dir.name}/{channel_color}"

                # we are now only getting the channels at the level
                # of the NiftiArrayCollection
                fname_list.append(sub_dir)
                if group_name in group_name_list:
                    raise RuntimeError(
                        f"Group {group_name} appears more than once")
                group_name_list.append(group_name)
                channel_list.append(channel_color)
    elif input_path.is_file():
        fname_list = [input_path, input_path]
        group_name_list = ["red", "green"]
        channel_list = ["red", "green"]
    else:
        raise RuntimeError(
            f"{input_path} is neither dir nor file")

    root_group = write_nii_file_list_to_ome_zarr(
        file_path_list=fname_list,
        group_name_list=group_name_list,
        channel_list=channel_list,
        output_dir=output_dir,
        downscale=downscale,
        clobber=False,
        n_processors=n_processors,
        metadata_collector=metadata_collector,
        DownscalerClass=XYZScaler,
        default_chunk=chunk_size)

    print("copying image_series_information.csv over")
    if sub_dir_list is not None:
        for sub_dir in sub_dir_list:
            src_path = sub_dir / "image_series_information.csv"
            if src_path.exists():
                dest_path = output_dir / src_path.parent.name / src_path.name
                shutil.copy(src_path, dest_path)

    if input_path.is_dir():
        copy_over_image_series_metadata(
            input_dir=input_path,
            output_dir=output_dir)

    metadata_collector.write_to_file()


def copy_over_image_series_metadata(
        input_dir,
        output_dir):
    """
    input_dir is the directory where the resampled .nii.gz files originate

    output_dir is the dir where the metadata file will be written
    """

    sub_dir_list = [n for n in input_dir.iterdir()
                    if n.is_dir()]

    k = "image_series_metadata.json"
    input_metadata_path = input_dir / k
    if input_metadata_path.exists():
        print("copying image_series_metadata.json over")
        with open(input_metadata_path, 'rb') as in_file:
            input_metadata = json.load(in_file)
        to_pop = []
        image_series_id_set = set([int(sub_dir.name)
                                   for sub_dir in sub_dir_list])
        for ii in range(len(input_metadata)):
            this_id = int(input_metadata[ii]['image_series_id'])
            if this_id not in image_series_id_set:
                to_pop.append(ii)
        to_pop.reverse()
        for ii in to_pop:
            input_metadata.pop(ii)

        output_metadata_path = output_dir / k
        with open(output_metadata_path, 'w') as out_file:
            out_file.write(json.dumps(input_metadata, indent=2))
