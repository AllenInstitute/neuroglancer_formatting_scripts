from typing import List, Union, Any, Optional, Tuple

import time
import h5py
import pathlib
import numpy as np
import dask.array

from neuroglancer_interface.utils.jp2_utils import (
    write_data_to_hdf5)

from neuroglancer_interface.utils.data_utils import (
    write_array_to_group,
    create_root_group)

from neuroglancer_interface.classes.downscalers import XYScaler


def convert_jp2_to_ome_zarr(
        config_list: List[dict],
        output_dir: pathlib.Path,
        clobber: bool = False,
        x_scale: float = 0.0003,
        y_scale: float = 0.0003,
        z_scale: float = 1.0,
        tmp_dir: Union[str, pathlib.Path]=None,
        nz_slice: Optional[Tuple[int, int]] = None) -> None:
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
        _convert_jp2_to_ome_zarr(
            h5_path=h5_path,
            root_group=root_group,
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=z_scale)
    finally:
        if h5_path.exists():
            h5_path.unlink()

def _convert_jp2_to_ome_zarr(
        h5_path: pathlib.Path,
        root_group: Any,
        x_scale: float = 0.0003,
        y_scale: float = 0.0003,
        z_scale: float = 1.0) -> None:


    with h5py.File(h5_path, 'r') as in_file:
        for data_key in ('green', 'red'):
            print(f"writing {data_key} channel")
            t0 = time.time()
            data = dask.array.from_array(in_file[data_key])

            write_array_to_group(
                arr=data.transpose(2, 1, 0),
                group=root_group.create_group(data_key),
                x_scale=x_scale,
                y_scale=y_scale,
                z_scale=z_scale,
                downscale=2,
                DownscalerClass=XYScaler,
                downscale_cutoff=60,
                default_chunk=256)

            duration = (time.time()-t0)/3600.0
            print(f"{data_key} channel took "
                  f"{duration:.2e} hours")
