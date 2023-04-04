from typing import Any

import dask.array
import h5py
from ome_zarr.dask_utils import resize as dask_resize
from ome_zarr.writer import write_multiscales_metadata
from ome_zarr.format import CurrentFormat
import pathlib
import multiprocessing
from neuroglancer_interface.classes.downscalers import XYScaler


def write_array_to_group_from_dask(
        h5_path,
        h5_key,
        group: Any,
        x_scale: float,
        y_scale: float,
        z_scale: float,
        downscale: int = 1,
        DownscalerClass=XYScaler,
        downscale_cutoff=64,
        default_chunk=64,
        axis_order=('x', 'y', 'z'),
        storage_options=None,
        n_processors=4):
    """
    Write a numpy array to an ome-zarr group

    Parameters
    ----------
    group:
        The ome_zarr group object to which the data will
        be written

    x_scale: float
        The physical scale of one x pixel in millimeters

    y_scale: float
        The physical scale of one y pixel in millimeters

    z_scale: float
        The physical scale of one z pixel in millimeters

    downscale: int
        The amount by which to downscale the image at each
        level of zoom

    axis_order:
        controls the order in which axes are written out to .zattrs
        (note x_scale, y_scale, z_scale will correspond to the 0th,
        1st, and 2nd dimensions in the data, without regard to what
        the axis names are; this needs to be fixed later)
    """

    with h5py.File(h5_path, 'r') as in_file:
        arr_shape = in_file[h5_key].shape
        arr_dtype = in_file[h5_key].dtype

    coord_transform = [[
        {'scale': [x_scale,
                   y_scale,
                   z_scale],
         'type': 'scale'}]]

    scaler = DownscalerClass(
               method='gaussian',
               downscale=downscale,
               downscale_cutoff=downscale_cutoff)

    list_of_nx_ny = scaler.create_empty_pyramid(base_shape=arr_shape)

    for nxny in list_of_nx_ny:
        this_coord = [{'scale': [x_scale*arr_shape[0]/nxny[0],
                                 y_scale*arr_shape[1]/nxny[1],
                                 z_scale*arr_shape[2]/nxny[2]],
                       'type': 'scale'}]
        coord_transform.append(this_coord)

    assert len(coord_transform) == len(list_of_nx_ny) + 1

    axes = [
        {"name": axis_order[0],
         "type": "space",
         "unit": "millimeter"},
        {"name": axis_order[1],
         "type": "space",
         "unit": "millimeter"},
        {"name": axis_order[2],
         "type": "space",
         "unit": "millimeter"}]

    chunk_x = max(1, min(arr_shape[0]//4, default_chunk))
    chunk_y = max(1, min(arr_shape[1]//4, default_chunk))
    chunk_z = max(1, min(arr_shape[2]//4, default_chunk))

    these_storage_opts = {'chunks': (chunk_x, chunk_y, chunk_z)}
    if storage_options is not None:
        for k in storage_options:
            if k == 'chunks':
                continue
            these_storage_opts[k] = storage_options[k]

    write_dask_image(
        h5_path,
        h5_key,
        scaler=scaler,
        root_group=group,
        coordinate_transformations=coord_transform,
        axes=axes,
        storage_options=these_storage_opts,
        n_processors=n_processors)



def write_dask_image(
        h5_path,
        h5_key,
        scaler,
        root_group,
        coordinate_transformations,
        axes,
        storage_options,
        n_processors=4):
    """
    Write data from an HDF5 file to an OME-zarr
    group using a dask array for parallelization.

    Parameters
    ----------
    h5_path:
        the path to the HDF5 containing the data
        to be written.

    h5_key:
        The name of the dataset in the HDF5 file

    scaler:
        The instance of a subclass of ome-zarr-py's
        Scaler class that is used to downsample the data

    root_group:
        The ome-zarr-py group where the data will be
        written

    coordinate_transformations:
        The dict of coordinate transformations that will
        be written out to the .zattrs file in
        root_group

    axes:
        The dict of axis metadata that will be written
        out to the .zattrs file in root_group

    storage_options:
        The dict of storage options to be passed along
        to dask.array.Array.to_zarr

    n_processors:
        the number of independent processes to start up
        (ome-zarr-py's dask utils never seemed to use more
        than one core, so I had to implement a by-hand
        parallelization using python's multiprocessing module).

    Returns
    -------
    None
        the data from the HDF5 file is writt out in OME-zarr
        format to root_group.
    """
    list_of_shapes = scaler._list_of_nx_ny

    mgr = multiprocessing.Manager()
    dataset_lookup = mgr.dict()
    process_list = []

    for idx in range(len(list_of_shapes)+1):
        p = multiprocessing.Process(
                target=_write_dask_image_worker,
                args=(h5_path, h5_key,
                      scaler, root_group, idx, storage_options))
        p.start()
        process_list.append(p)
        while len(process_list) >= n_processors:
            process_list = _winnow_process_list(process_list)

    for p in process_list:
        p.join()

    datasets = [{'path': str(idx),
                 'coordinateTransformations': coordinate_transformations[idx]}
                for idx in range(len(list_of_shapes)+1)]

    write_multiscales_metadata(
        root_group,
        datasets,
        fmt=CurrentFormat(),
        axes=axes,
        name=None)


def _write_dask_image_worker(
        h5_path,
        h5_key,
        scaler,
        root_group,
        sub_group_idx,
        storage_options):
    """
    Worker function for one process writing data
    from an HDF5 file to an OME-Zarr group.

    This function writes a particular downsampling of
    the data to root_group.path / sub_group_idx

    Parameters
    ----------
    h5_path:
        the path to the HDF5 containing the data
        to be written.

    h5_key:
        The name of the dataset in the HDF5 file

    scaler:
        The instance of asubclass of ome-zarr-py's
        Scaler class that is used to downsample the data

    root_group:
        The ome-zarr-py group where the data will be
        written

    sub_group_idx:
        An integer used to find the array shape for this
        particular downsampling in scaler._list_of_nx_ny.
        Also the name of the OME-zarr sub group where this
        downsampling of the data is to be written.

    storage_options:
        The dict of storage options to be passed along
        to dask.array.Array.to_zarr

    Returns
    -------
    None
        the data from the HDF5 file is writt out in OME-zarr
        format to root_group.

    """

    with h5py.File(h5_path, mode='r', swmr=True) as in_file:
        image = dask.array.from_array(in_file[h5_key])

        list_of_nx_ny = scaler._list_of_nx_ny

        if sub_group_idx > 0:
            image = dask_resize(
                        image,
                        list_of_nx_ny[sub_group_idx-1],
                        preserve_range=True).astype(image.dtype)

        print(storage_options)
        job = image.rechunk(storage_options['chunks']).to_zarr(
                url=root_group.store,
                component=str(pathlib.Path(root_group.path) / str(sub_group_idx)),
                storage_options=storage_options,
                compressor=storage_options["compressor"])


def _winnow_process_list(
        process_list):
    """
    Loop over a list of processes, popping out any that have
    been completed. Return the winnowed list of processes.
    Parameters
    ----------
    process_list: List[multiprocessing.Process]
    Returns
    -------
    process_list: List[multiprocessing.Process]
    """
    to_pop = []
    for ii in range(len(process_list)-1, -1, -1):
        if process_list[ii].exitcode is not None:
            to_pop.append(ii)
    for ii in to_pop:
        process_list.pop(ii)
    return process_list
