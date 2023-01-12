import glymur
import pathlib

from neuroglancer_interface.utils.data_utils import (
    write_array_to_group,
    create_root_group)

from neuroglancer_interface.classes.downscalers import XYScaler


def convert_jp2_to_ome_zarr(
        jp2_path: pathlib.Path,
        output_dir: pathlib.Path,
        clobber: bool = False,
        x_scale: float = 0.0003,
        y_scale: float = 0.0003,
        z_scale: float = 1.0) -> None:
    """
    Result is just written to the specified group.
    """

    if not jp2_path.is_file():
        raise RuntimeError(f"{jp2_path} is not a file")

    root_group = create_root_group(
                    output_dir=output_dir,
                    clobber=clobber)

    data_src = glymur.Jp2k(data_file)
    write_array_to_group(
        arr=data_src[:, :, :],
        group=root_group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        downscale=2,
        DownscalerClass=XYScaler,
        downscale_cutoff=2048,
        default_chunk=1024)
