from typing import List, Any
import json
import pandas as pd
import numpy as np
import SimpleITK
import pathlib
import shutil
import time
import zarr
from numcodecs import blosc
import multiprocessing
from ome_zarr.io import parse_url

from ome_zarr.writer import write_image
from neuroglancer_interface.utils.multiprocessing_utils import (
    _winnow_process_list)

from neuroglancer_interface.classes.downscalers import (
    XYZScaler)

from neuroglancer_interface.classes.nifti_array import (
    get_nifti_obj)


blosc.use_threads = False


def create_root_group(
        output_dir,
        clobber=False):
    """
    Create an OME-ZARR group at output_dir

    Parameters
    ----------
    output_dir:
        The path at twhich to create the OME-ZARR group

    clobber:
        If True, delete whatever is at output_dir before
        proceeding. If False and  output_dir exists,
        raise a RuntimeError

    Returns
    -------
    A dict
        "group": An OME-ZARR group object pointing to output_dir
        "path": path to that parent directory
    """

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    if output_dir.exists():
        if not clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            print(f"cleaning out {output_dir}")
            shutil.rmtree(output_dir)
            print("done cleaning")

    assert not output_dir.exists()
    output_dir.mkdir()
    assert output_dir.is_dir()

    store = parse_url(output_dir, mode="w").store
    root_group = zarr.group(store=store)
    obj = {"group": root_group, "path": output_dir}
    return obj


def write_nii_file_list_to_ome_zarr(
        config_list,
        root_group,
        n_processors,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        default_chunk=64,
        do_transposition=False):
    """
    Parameters
    ----------
    config_list:
        List of dicts like
             "path": -> path to .nii file
             "group": -> group under root_group where data will be written
             "channel": -> optional channel to choose
    root_group:
        Dict created by create_root_group
    n_processors:
        Number of independent workers to spin up
    DownscalerClass:
        Class that will handle downsampling the images
    downscale_cutoff:
        Stop downscaling a dimension when it gets this large
    default_chunk:
        Guess at ome-zarr data chunk size
    do_transposition:
        If true, swap X, Z axes and map Z -> -X
    """
    # check uniqueness of group
    group_set = set()
    for config in config_list:
        g = config['group']
        if g in group_set:
            raise RuntimeError(
                f"group {g} occurs more than once in config_list")
        group_set.add(g)

    batches = []
    for i_batch in range(n_processors):
        batches.append([])
    for i_config, config in enumerate(config_list):
        i_batch = i_config % n_processors
        batches[i_batch].append(config)

    process_list = []
    for i_batch in range(n_processors):
        p = multiprocessing.Process(
                target=_write_nii_file_list_worker,
                kwargs={
                    'config_list': batches[i_batch],
                    'root_group': root_group,
                    'downscale_cutoff': downscale_cutoff,
                    'default_chunk': default_chunk,
                    'DownscalerClass': DownscalerClass,
                    'do_transposition': do_transposition})
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    return root_group


def _write_nii_file_list_worker(
        config_list,
        root_group,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        default_chunk=64,
        do_transposition=False):

    for config in config_list:

        if 'channel' in config:
            channel = config['channel']
        else:
            channel = 'red'

        write_nii_to_group(
            root_group=root_group,
            group_name=config['group'],
            nii_file_path=config['path'],
            DownscalerClass=DownscalerClass,
            downscale_cutoff=downscale_cutoff,
            default_chunk=default_chunk,
            channel=channel,
            do_transposition=do_transposition)


def write_nii_to_group(
        root_group,
        group_name,
        nii_file_path,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        default_chunk=64,
        channel='red',
        do_transposition=False):
    """
    Write a single nifti file to an ome_zarr group

    Parameters
    ----------
    root_group:
        dict created by create_root_group

    group_name: str
        is the name of the group being created for this data

    nii_file_path: Pathlib.path
        is the path to the nii file being written

    do_transposition:
        If True, transpose the NIFTI volumes so that
        (x, y, z) -> (z, y, x)
    """
    nii_file_path = pathlib.Path(nii_file_path)

    if group_name is not None:
        this_group = root_group["group"].create_group(f"{group_name}")
        zattr_path = root_group["path"] / this_group.path
    else:
        this_group = root_group["group"]
        zattr_path = root_group["path"]
    zattr_path = zattr_path / '.zattrs'

    nii_obj = get_nifti_obj(nii_file_path,
                            do_transposition=do_transposition)

    nii_results = nii_obj.get_channel(
                    channel=channel)

    x_scale = nii_results['scales'][0]
    y_scale = nii_results['scales'][1]
    z_scale = nii_results['scales'][2]

    arr = nii_results['channel']

    max_x = np.argmax(np.sum(arr, axis=(1, 2)))
    max_y = np.argmax(np.sum(arr, axis=(0, 2)))
    max_z = np.argmax(np.sum(arr, axis=(0, 1)))

    write_array_to_group(
        arr=arr,
        group=this_group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        DownscalerClass=DownscalerClass,
        downscale_cutoff=downscale_cutoff,
        default_chunk=default_chunk)

    # add max plane data to .zattrs
    zattr_data = json.load(open(zattr_path, 'rb'))
    assert 'max_planes' not in zattr_data
    zattr_data['max_planes'] = [int(max_x), int(max_y), int(max_z)]
    assert 'nii_file_path' not in zattr_data
    zattr_data['nii_file_path'] = str(nii_file_path.resolve().absolute())
    assert 'dtype' not in zattr_data
    zattr_data['dtype'] = str(arr.dtype)
    assert 'quantiles' not in zattr_data

    valid = (arr>0.0)
    q_values = ('0.25', '0.50', '0.75', '0.80', '0.90')
    q = np.quantile(arr[valid], [float(v) for v in q_values])
    quantiles = {
        k:v for k, v in zip(q_values, q)}
    zattr_data['quantiles'] = quantiles

    with open(zattr_path, 'w') as out_file:
        out_file.write(json.dumps(zattr_data, indent=2))

    print(f"wrote {nii_file_path} to {group_name}")


def write_summed_nii_files_to_group(
        file_path_list,
        group,
        downscale = 2,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        default_chunk=64,
        channel='red',
        do_transposition=False):
    """
    Sum the arrays in all of the files in file_path list
    into a single array and write that to the specified
    OME-zarr group

    downscale sets the amount by which to downscale the
    image at each level of zoom
    """

    main_array = None
    for file_path in file_path_list:
        nii_obj = get_nifti_obj(file_path,
                                do_transposition=do_transposition)

        nii_results = nii_obj.get_channel(
                        channel=channel)

        this_array = nii_results['channel']

        (this_x_scale,
         this_y_scale,
         this_z_scale) = nii_results['scales']

        if main_array is None:
            main_array = this_array
            x_scale = this_x_scale
            y_scale = this_y_scale
            z_scale = this_z_scale
            main_pth = file_path
            continue

        if this_array.shape != main_array.shape:
            msg = f"\n{main_path} has shape {main_array.shape}\n"
            msg += f"{file_path} has shape {this_array.shape}\n"
            msg += "cannot sum"
            raise RuntimeError(msg)

        if not np.allclose([x_scale, y_scale, z_scale],
                           [this_x_scale, this_y_scale, this_z_scale]):
            msg = f"\n{main_path} has scales ("
            msg += f"{x_scale}, {y_scale}, {z_scale})\n"
            msg += f"{file_path} has scales ("
            msg += f"{this_x_scale}, {this_y_scale}, {this_z_scale})\n"
            msg += "cannot sum"
            raise RuntimeError

        main_array += this_array

    write_array_to_group(
        arr=main_array,
        group=group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        downscale=downscale,
        DownscalerClass=DownscalerClass,
        downscale_cutoff=downscale_cutoff,
        default_chunk=default_chunk)


def write_array_to_group(
        arr: np.ndarray,
        group: Any,
        x_scale: float,
        y_scale: float,
        z_scale: float,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        default_chunk=64,
        storage_options=None):
    """
    Write a numpy array to an ome-zarr group

    Parameters
    ----------
    arr:
        the 3D numpy array of data to be converted to
        OME-ZARR

    group:
        The ome_zarr group object to which the data will
        be written

    x_scale: float
        The physical scale of one x pixel in millimeters

    y_scale: float
        The physical scale of one y pixel in millimeters

    z_scale: float
        The physical scale of one z pixel in millimeters

    DownscalerClass:
        The class to be used for downscaling the image
        (if None, there will be no downscaling)

    downscale_cutoff:
        Stop downscaling a dimension before it ends up smaller
        than this.

    default_chunk:
        Attempt to guess chunk size of data
    """

    # neuroglancer does not support 64 bit floats
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)

    shape = arr.shape

    coord_transform = [[
        {'scale': [x_scale,
                   y_scale,
                   z_scale],
         'type': 'scale'}]]

    if DownscalerClass is not None:
        scaler = DownscalerClass(
                   method='gaussian',
                   downscale=1,
                   downscale_cutoff=downscale_cutoff)

        list_of_nx_ny = scaler.create_empty_pyramid(base=arr)

        for nxny in list_of_nx_ny:
            this_coord = [{'scale': [x_scale*arr.shape[0]/nxny[0],
                                     y_scale*arr.shape[1]/nxny[1],
                                     z_scale*arr.shape[2]/nxny[2]],
                           'type': 'scale'}]
            coord_transform.append(this_coord)
    else:
        scaler = None

    axes = [
        {"name": "x",
         "type": "space",
         "unit": "millimeter"},
        {"name": "y",
         "type": "space",
         "unit": "millimeter"},
        {"name": "z",
         "type": "space",
         "unit": "millimeter"}]

    chunk_x = max(1, min(shape[0]//4, default_chunk))
    chunk_y = max(1, min(shape[1]//4, default_chunk))
    chunk_z = max(1, min(shape[2]//4, default_chunk))

    these_storage_opts = {'chunks': (chunk_x, chunk_y, chunk_z)}
    if storage_options is not None:
        for k in storage_options:
            if k == 'chunks':
                continue
            these_storage_opts[k] = storage_options[k]

    write_image(
        image=arr,
        scaler=scaler,
        group=group,
        coordinate_transformations=coord_transform,
        axes=axes,
        storage_options=these_storage_opts)


def get_celltype_lookups_from_rda_df(
        csv_path):
    """
    Read a lookup mapping the integer index from a cell type
    name to its human readable form

    useful only if reading directly from the dataframe produced
    from Zizhen's .rda file. That is currently out of scope
    """
    df = pd.read_csv(csv_path)
    cluster = dict()
    level1 = dict()
    level2 = dict()
    for id_arr, label_arr, dest in [(df.Level1_id.values,
                                     df.Level1_label.values,
                                     level1),
                                    (df.Level2_id.values,
                                     df.Level2_label.values,
                                     level2),
                                    (df.cluster_id.values,
                                     df.cluster_label.values,
                                     cluster)]:
        for id_val, label_val in zip(id_arr, label_arr):
            if np.isnan(id_val):
                continue
            id_val = int(id_val)
            if id_val in dest:
                if dest[id_val] != label_val:
                    raise RuntimeError(
                        f"Multiple values for {id_val}\n"
                        f"{label_val}\n{dest[id_val]}\n"
                        f"{line}")
            else:
                dest[id_val] = label_val

    return {"cluster": cluster,
            "Level1": level1,
            "Level2": level2}
