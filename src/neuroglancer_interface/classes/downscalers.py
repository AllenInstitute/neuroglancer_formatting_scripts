from typing import List, Any
import numpy as np
from ome_zarr.scale import Scaler
from skimage.transform import pyramid_gaussian
from skimage.transform import resize as skimage_resize

class ScalerBase(Scaler):

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError("Base nearest")

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        raise RuntimeError("did not expect to run resize_image")

    def laplacian(self, base: np.ndarray) -> List[np.ndarray]:
        raise RuntimeError("did not expect to run laplacian")

    def local_mean(self, base: np.ndarray) -> List[np.ndarray]:
        raise RuntimeError("did not expect to run local_mean")

    def gaussian(self, base: np.ndarray) -> List[np.ndarray]:

        # I wouldn't expect this to be okay, but apparently
        # this code never actually executes
        raise RuntimeError("gaussian")

    @classmethod
    def create_empty_pyramid(cls, base, downscale=2):
        raise NotImplementedError("base create_empty_pyramid")


class XYScaler(ScalerBase):
    """
    A scaler that ignores the z dimension, since it is
    so small relative to the other two dimensions in this initial
    dataset
    """

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        assert len(base.shape) == 3

        (results,
         list_of_nx_ny) = self.create_empty_pyramid(
                               base,
                               downscale=self.downscale)

        for iz in range(base.shape[2]):
            for nxny in list_of_nx_ny:
                img = skimage_resize(base[:, :, iz], (nxny[0], nxny[1]))
                results[nxny][:, :, iz] = img

        output = [base]
        return output + [results[key].astype(base.dtype)
                         for key in list_of_nx_ny]



    @classmethod
    def create_empty_pyramid(cls, base, downscale=2):
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
        nx = base.shape[0]
        ny = base.shape[1]
        nz = base.shape[2]
        results = dict()
        list_of_nx_ny = []
        while nx > base.shape[2] or ny > base.shape[2]:
            nx = nx//downscale
            ny = ny//downscale
            data = np.zeros((nx, ny, base.shape[2]), dtype=float)
            key = (nx, ny, nz)
            results[key] = data
            list_of_nx_ny.append(key)

        return results, list_of_nx_ny


class XYZScaler(ScalerBase):
    """
    A scaler that ignores the z dimension, since it is
    so small relative to the other two dimensions in this initial
    dataset
    """

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        assert len(base.shape) == 3

        (results,
         list_of_nx_ny) = self.create_empty_pyramid(
                               base,
                               downscale=self.downscale)

        for nxnynz in list_of_nx_ny:
            img = skimage_resize(base[:, :, :], nxnynz)
            results[nxny][:, :, :] = img

        output = [base]
        return output + [results[key].astype(base.dtype)
                         for key in list_of_nx_ny]


    @classmethod
    def create_empty_pyramid(cls, base, downscale=2):
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

        list_of_nx_ny:
            List of valid keys of results
        """
        nx = base.shape[0]
        ny = base.shape[1]
        nz = base.shape[2]
        results = dict()
        list_of_nx_ny = []
        while max(nx, ny, nz) > 32:
            nx = nx//downscale
            ny = ny//downscale
            nz = nz//downscale
            data = np.zeros((nx, ny, nz), dtype=float)
            key = (nx, ny, nz)
            results[key] = data
            list_of_nx_ny.append(key)

        return results, list_of_nx_ny
