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
            base_shape):
        """
        Create a lookup table of empty arrays for an
        image/volume pyramid

        Parameters
        ----------
        base: np.ndarray
            The array that will be converted into an image/volume
            pyramid

        downscale: int
            The factor by which to downscale base at each level of
            zoom

        Returns
        -------
        results: dict
            A dict mapping an image shape (nx, ny) to
            an empty array of size (nx, ny, nz)

            NOTE: we are not downsampling nz in this setup

        list_of_nx_ny:
            List of valid keys of results
        """
        if not hasattr(self, '_list_of_nx_ny'):
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
    Result is just written to the specified group.
    """

    root_group = create_root_group(
                    output_dir=output_dir,
                    clobber=clobber)

    idx_list = np.array([r['specimen_tissue_index'] for r in config_list])
    sorted_dex = np.argsort(idx_list)
    config_list = [config_list[idx] for idx in sorted_dex]

    if nz_slice is not None:
        config_list = config_list[nz_slice[0]:nz_slice[1]]

    h5_path = write_data_to_hdf5(
                config_list=config_list,
                tmp_dir=tmp_dir)

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
        downscale_cutoff=2501,
        default_chunk=128,
        downscaler_class=HighResScaler) -> None:

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
            storage_options=storage_options)

        duration = (time.time()-t0)/3600.0
        print(f"{data_key} channel took "
              f"{duration:.2e} hours")



