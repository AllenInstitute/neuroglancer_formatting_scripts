from typing import List, Union, Any, Optional, Tuple

import time
import h5py
import pathlib
import numpy as np
import dask.array

from numcodecs import Blosc

from neuroglancer_interface.utils.jp2_utils import (
    write_data_to_hdf5)

from neuroglancer_interface.utils.data_utils import (
    create_root_group)

from neuroglancer_interface.utils.dask_utils import (
    write_array_to_group_from_dask)

from neuroglancer_interface.utils.utils import get_prime_factors
from neuroglancer_interface.classes.downscalers import XYZScaler


class HighResScaler(XYZScaler):

    def create_empty_pyramid(
            self,
            base):
        """
        Create a list of (nx, ny, nz) tuples representing
        the shapes of the downsamplings of the data that
        need to be computed.

        Parameters
        ----------
        base:
            Either an array representing the base image or
            a tuple representing its shape.

        Returns
        -------
        list_of_nx_ny
            A list of (nx, ny, nz) tuples representing the shapes
            of the downscalings that will be written to OME-zarr
        """
        if not hasattr(self, '_list_of_nx_ny'):

            if isinstance(base, tuple):
                base_shape = base.shape
            else:
                base_shape = base.shape
            nx = base_shape[0]
            ny = base_shape[1]
            nz = base_shape[2]

            nx_factor_list = get_prime_factors(nx)
            ny_factor_list = get_prime_factors(ny)
            nz_factor_list = get_prime_factors(nz)

            list_of_nx_ny = []

            keep_going = True
            while keep_going:
                keep_going = False
                nx_factor = nx_factor_list[0]
                ny_factor = ny_factor_list[0]
                nz_factor = nz_factor_list[0]
                if len(nx_factor_list) > 1:
                    if nx // nx_factor >= self.downscale_cutoff:
                        nx = nx // nx_factor
                        nx_factor_list.pop(0)
                        keep_going = True
                if len(ny_factor_list) > 1:
                    if ny//ny_factor >= self.downscale_cutoff:
                        ny = ny // ny_factor
                        ny_factor_list.pop(0)
                        keep_going = True
                if len(nz_factor_list) > 1:
                    if nz // nz_factor >= self.downscale_cutoff:
                        nz = nz // nz_factor
                        nz_factor_list.pop(0)
                        keep_going = True

                if keep_going:
                    list_of_nx_ny.append((nx, ny, nz))

            self._list_of_nx_ny = list_of_nx_ny
            self.max_layer = len(self._list_of_nx_ny)

            print(f"list_of_nx_ny {self._list_of_nx_ny}")

        return self._list_of_nx_ny


def convert_jp2_to_ome_zarr(
        config_list: List[dict],
        output_dir: pathlib.Path,
        clobber: bool = False,
        x_scale: float = 0.0003,
        y_scale: float = 0.0003,
        z_scale: float = 1.0,
        tmp_dir: Union[str, pathlib.Path]=None,
        nz_slice: Optional[Tuple[int, int]] = None,
        downscaler_class=HighResScaler,
        downscale_cutoff: int = 2501,
        default_chunk: int = 512) -> None:
    """
    Write an image stack represented by a series of JP2 files
    into an OME-zarr group.

    Data is first gathered into a single HDF5 file in a temporary directory
    and then written to OME-zarr. Both of these steps are time-consuming.
    A (30000, 40000, 140) volume took 8 hours to write to HDF5 and
    10 hours to write from HDF5 to OME-zarr (this was using 32 cores
    and 200 GB of memory).

    Parameters
    ----------
    config_list:
        The list of dicts pointing to the jp2 files to be stacked.
        Each dict should have a 'specimen_tissue_index' indicating
        it's z position in the stack and an 'image_path'

    output_dir:
        The OME-zarr group to be written out

    clobber:
        boolean indicating whether or not to overwrite an existing
        output_dir

    x_scale:
        Size in mm of the voxels in the x direction

    y_scale:
        Size in mm of the voxels in the y direction

    z_scale:
        Size in mm of the voxels in the z direction

    tmp_dir:
        Directory where temporary HDF5 file will be written.

    nz_slice:
        For testing. A tuple (z_min, z_max) indicating the config
        files in config_list to use. Set to None if you want to
        write the whole stack.

    downscaler_class:
        Subclass of ome-zarr-py's Scaler to use to downsample the
        data to different scales.

    downscale_cutoff:
        Do not write downsamplings in which one dimension has fewer
        voxels than downscale_cutoff.

    default_chunk:
        Data will be written out in chunks of size
        (default_chunk, default_chunk, default_chunk)

    Returns
    -------
    None
        Data is written out to an OME-zarr group in output_dir.
    """

    root_group = create_root_group(
                    output_dir=output_dir,
                    clobber=clobber)

    idx_list = np.array([r['specimen_tissue_index'] for r in config_list])
    sorted_dex = np.argsort(idx_list)
    config_list = [config_list[idx] for idx in sorted_dex]

    if nz_slice is not None:
        config_list = config_list[nz_slice[0]:nz_slice[1]]

    # write the data to a temporary HDF5 file so that it can
    # be manipulated as a dask array.
    h5_path = write_data_to_hdf5(
                config_list=config_list,
                tmp_dir=tmp_dir)

    # write the data from HDF5 to OME-zarr
    try:
        _convert_hdf5_to_ome_zarr(
            h5_path=h5_path,
            root_group=root_group,
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=z_scale,
            downscale_cutoff=downscale_cutoff,
            default_chunk=default_chunk,
            downscaler_class=downscaler_class)
    finally:
        if h5_path.exists():
            h5_path.unlink()


def _convert_hdf5_to_ome_zarr(
        h5_path: pathlib.Path,
        root_group: Any,
        x_scale: float = 0.0003,
        y_scale: float = 0.0003,
        z_scale: float = 0.1,
        downscaler_class=HighResScaler,
        downscale_cutoff=2501,
        default_chunk=128,
        n_processors=4) -> None:
    """
    Write data from an HDF5 file to an OME-zarr group.

    Parameters
    ----------
    h5_path:
        Path to the HDF5 file containing the data to be
        written out.

    root_group:
        The OME-zarr group to which the data will be written


    x_scale: float
        The physical scale of one x pixel in millimeters

    y_scale: float
        The physical scale of one y pixel in millimeters

    z_scale: float
        The physical scale of one z pixel in millimeters


    downscaler_class:
        Subclass of ome-zarr-py's Scaler to use to downsample the
        data to different scales.

    downscale_cutoff:
        Do not write downsamplings in which one dimension has fewer
        voxels than downscale_cutoff.

    default_chunk:
        Data will be written out in chunks of size
        (default_chunk, default_chunk, default_chunk)

    n_processors:
        the number of independent processes to start up
        (ome-zarr-py's dask utils never seemed to use more
        than one core, so I had to implement a by-hand
        parallelization using python's multiprocessing module).

    Returns
    -------
    None
        Data is written out to an OME-zarr group in output_dir.
    """

    storage_options = {'compressor':
                        Blosc(cname='lz4',
                              clevel=5,
                              shuffle=Blosc.SHUFFLE)}


    for data_key in ('green', 'red'):
        print(f"writing {data_key} channel")
        t0 = time.time()

        write_array_to_group_from_dask(
            h5_path=h5_path,
            h5_key=data_key,
            group=root_group.create_group(data_key),
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=z_scale,
            downscale=2,
            DownscalerClass=downscaler_class,
            downscale_cutoff=downscale_cutoff,
            default_chunk=default_chunk,
            axis_order=('y', 'x', 'z'),
            storage_options=storage_options,
            n_processors=n_processors)

        duration = (time.time()-t0)/3600.0
        print(f"{data_key} channel took "
              f"{duration:.2e} hours")



