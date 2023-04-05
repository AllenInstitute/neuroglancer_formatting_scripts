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
    An OME-ZARR group object pointing to output_dir
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
    root_group.absolute_path = output_dir

    return root_group


def write_nii_file_list_to_ome_zarr(
        file_path_list,
        group_name_list,
        output_dir,
        downscale=2,
        n_processors=4,
        clobber=False,
        prefix=None,
        root_group=None,
        metadata_collector=None,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        only_metadata=False,
        default_chunk=128,
        channel_list=None,
        do_transposition=False):
    """
    Convert a list of nifti files into OME-zarr format

    Parameters
    -----------
    file_path_list: List[pathlib.Path]
        list of paths to files to be written

    group_name_list: List[str]
        list of the names of the OME-zarr groups to be written
        (same order as file_path_list)

    output_dir: pathlib.Path
        The directory where the parent OME-zarr group will be
        written

    downscale: int
        Factor by which to downscale images as you are writing
        them (i.e. when you zoom out one level, by how much
        are you downsampling the pixels)

    n_processors: int
        Number of independent processes to use.

    clobber: bool
        If False, do not overwrite an existing group (throw
        an exception instead)

    prefix: str
        optional sub-group in which all data is written
        (i.e. option to write groups to output_dir/prefix/)

    root_group:
        Optional root group into which to write ome-zarr
        group. If None, will be created.

    default_chunk: int
        default size for single dimension of chunk when writing
        data to disk

    do_transposition:
        If True, transpose the NIFTI volumes so that
        (x, y, z) -> (z, y, x)

    Returns
    -------
    the root group

    Notes
    -----
    Importing zarr causes multiprocessing to emit a warning about
    leaked semaphore objects. *Probably* this is fine. It's just
    scary. The zarr developers are working on this

    https://github.com/zarr-developers/numcodecs/issues/230
    """
    t0 = time.time()
    if not isinstance(file_path_list, list):
        file_path_list = [file_path_list,]
    if not isinstance(group_name_list, list):
        group_name_list = [group_name_list,]

    if len(file_path_list) != len(group_name_list):
        msg = f"\ngave {len(file_path_list)} file paths but\n"
        msg += f"{len(group_name_list)} group names"

    if root_group is None:
        root_group = create_root_group(
                        output_dir=output_dir,
                        clobber=clobber)

    if prefix is not None:
        parent_group = root_group.create_group(prefix)
    else:
        parent_group = root_group

    if len(file_path_list) == 1:

        _write_nii_file_list_worker(
            file_path_list=file_path_list,
            group_name_list=group_name_list,
            root_group=parent_group,
            downscale=downscale,
            metadata_collector=metadata_collector,
            DownscalerClass=DownscalerClass,
            downscale_cutoff=downscale_cutoff,
            only_metadata=only_metadata,
            default_chunk=default_chunk,
            channel_list=channel_list,
            do_transposition=do_transposition)

    else:
        n_workers = max(1, n_processors-1)
        n_workers = min(n_workers, len(file_path_list))
        file_lists = []
        group_lists = []
        channel_sub_lists = []
        for ii in range(n_workers):
            file_lists.append([])
            group_lists.append([])
            if channel_list is not None:
                channel_sub_lists.append([])
            else:
                channel_sub_lists.append(None)

        for ii in range(len(file_path_list)):
            jj = ii % n_workers
            file_lists[jj].append(file_path_list[ii])
            group_lists[jj].append(group_name_list[ii])
            if channel_list is not None:
                channel_sub_lists[jj].append(channel_list[ii])

        process_list = []
        for ii in range(n_workers):
            p = multiprocessing.Process(
                    target=_write_nii_file_list_worker,
                    kwargs={'file_path_list': file_lists[ii],
                            'group_name_list': group_lists[ii],
                            'channel_list': channel_sub_lists[ii],
                            'root_group': parent_group,
                            'downscale': downscale,
                            'metadata_collector': metadata_collector,
                            'DownscalerClass': DownscalerClass,
                            'downscale_cutoff': downscale_cutoff,
                            'only_metadata': only_metadata,
                            'default_chunk': default_chunk,
                            'do_transposition': do_transposition})
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

    duration = time.time() - t0
    if prefix is not None:
        print(f"{prefix} took {duration:.2e} seconds")

    return root_group


def _write_nii_file_list_worker(
        file_path_list,
        group_name_list,
        channel_list,
        root_group,
        downscale,
        metadata_collector=None,
        DownscalerClass=XYZScaler,
        downscale_cutoff=64,
        only_metadata=False,
        default_chunk=64,
        do_transposition=False):
    """
    Worker function to actually convert a subset of nifti
    files to OME-zarr

    Parameters
    ----------
    file_path_list: List[pathlib.Path]
        List of paths to nifti files to convert

    group_name_list: List[str]
        List of the names of the OME-zarr groups to
        write the nifti files to

    root_group:
        The parent group object as created by ome_zarr

    downscale: int
        The factor by which to downscale the images at each
        level of zoom.

    do_transposition:
        If True, transpose the NIFTI volumes so that
        (x, y, z) -> (z, y, x)
    """

    for idx in range(len(file_path_list)):
        f_path = file_path_list[idx]
        grp_name = group_name_list[idx]
        if channel_list is not None:
            channel = channel_list=channel_list[idx]
        else:
            channel = None

        write_nii_to_group(
            root_group=root_group,
            group_name=grp_name,
            channel=channel,
            nii_file_path=f_path,
            downscale=downscale,
            metadata_collector=metadata_collector,
            DownscalerClass=DownscalerClass,
            downscale_cutoff=downscale_cutoff,
            only_metadata=only_metadata,
            default_chunk=default_chunk,
            do_transposition=do_transposition)


def write_nii_to_group(
        root_group,
        group_name,
        nii_file_path,
        transpose=True,
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
        the ome_zarr group that the new group will
        be a child of (an object created using the ome_zarr
        library)

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
        this_group = root_group.create_group(f"{group_name}")
        zattr_path = root_group.absolute_path / this_group.path
    else:
        this_group = root_group
        zattr_path = root_group.absolute_path
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
