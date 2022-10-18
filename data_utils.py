from typing import List, Any
import numpy as np
from ome_zarr.scale import Scaler
from skimage.transform import pyramid_gaussian
from skimage.transform import resize as skimage_resize
from ome_zarr.writer import write_image


def write_array_to_group(
        arr: np.ndarray,
        group: Any,
        x_scale: float,
        y_scale: float,
        z_scale: float,
        downscale: int = 1):

    shape = arr.shape

    coord_transform = [[
        {'scale': [x_scale,
                   y_scale,
                   z_scale],
         'type': 'scale'}]]

    if downscale > 1:
        (_,
         list_of_nx_ny) = _create_empty_pyramid(
                              base=arr,
                              downscale=downscale)

        del _

        for nxny in list_of_nx_ny:
            this_coord = [{'scale': [x_scale*arr.shape[0]/nxny[0],
                                     y_scale*arr.shape[1]/nxny[1],
                                     z_scale],
                           'type': 'scale'}]
            coord_transform.append(this_coord)

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

    if downscale > 1:
        scaler = XYScaler(
                   method='gaussian',
                   downscale=downscale)
    else:
        scaler = None

    write_image(
        image=arr,
        scaler=scaler,
        group=group,
        coordinate_transformations=coord_transform,
        axes=axes,
        storage_options={'chunks':(shape[0]//4,
                                   shape[1]//4,
                                   shape[2]//4)})



def _create_empty_pyramid(base, downscale=2):
    """
    Returns space for downsampled images
    """
    result = []
    nx = base.shape[0]
    ny = base.shape[1]
    results = dict()
    list_of_nx_ny = []
    while nx > base.shape[2] or ny > base.shape[2]:
        nx = nx//downscale
        ny = ny//downscale
        data = np.zeros((nx, ny, base.shape[0]), dtype=float)
        key = (nx, ny)
        results[key] = data
        list_of_nx_ny.append(key)

    return results, list_of_nx_ny



class XYScaler(Scaler):
    """
    A scaler that ignores the z dimension, since it is
    so small relative to the other two dimensions in this initial
    dataset
    """

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError("did not expect to run resize_image")

    def laplacian(self, base: np.ndarray) -> List[np.ndarray]:
        raise RuntimeError("did not expect to run laplacian")

    def local_mean(self, base: np.ndarray) -> List[np.ndarray]:
        raise RuntimeError("did not expect to run local_mean")

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        assert len(base.shape) == 3

        (results,
         list_of_nx_ny) = _create_empty_pyramid(
                               base,
                               downscale=self.downscale)

        for iz in range(base.shape[2]):
            for nxny in list_of_nx_ny:
                img = skimage_resize(base[:, :, iz], nxny)
                results[nxny][:, :, iz] = img

        output = [base]
        return output + [np.round(results[key]).astype(base.dtype)
                         for key in list_of_nx_ny]


    def gaussian(self, base: np.ndarray) -> List[np.ndarray]:

        (results,
         list_of_nx_ny) = _create_empty_pyramid(
                              base,
                              downscale=self.downscale)

        for iz in range(base.shape[2]):
            gen = pyramid_gaussian(
                    base[:, :, iz],
                    downscale=self.downscale,
                    max_layer=-1,
                    multichannel=False)
            for layer in gen:
                nx = layer.shape[0]
                ny = layer.shape[1]
                key = (nx, ny)
                if key not in results:
                    break
                results[key][:, :, iz] = layer

        print(results)
        output = [base]
        return output + [np.round(results[key]).astype(base.dtype)
                         for key in list_of_nx_ny]


