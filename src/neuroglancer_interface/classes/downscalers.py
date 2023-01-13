from typing import List, Any, Union, Tuple
import numpy as np
from ome_zarr.scale import Scaler
from dataclasses import dataclass
import dask.array
from ome_zarr.dask_utils import resize as dask_resize
from skimage.transform import pyramid_gaussian
from skimage.transform import resize as skimage_resize

@dataclass
class ScalerBase(Scaler):

    downscale_cutoff: int = 128

    def nearest(self, base: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError("Base nearest")

    def resize_image(self, image: Any) -> Any:
        if not isinstance(image, dask.array.Array):
            raise RuntimeError("did not expect to run resize_image  on np.ndarray")
        list_of_nx_ny = self.create_empty_pyramid(
                    base=None,
                    downscale=None,
                    downscale_cutoff=None)

        this_idx = None
        for idx in range(len(list_of_nx_ny)):
            if list_of_nx_ny[idx] == image.shape:
                this_idx = idx
                break

        if this_idx is None:
            if image.shape[0] > list_of_nx_ny[0][0]:
                this_idx = -1

        if this_idx is None:
            raise RuntimeError(f"could not find shape {image.shape} in\n"
                               f"{list_of_nx_ny}")
        this_dtype = image.dtype
        new_img = dask_resize(image,
                              output_shape=list_of_nx_ny[this_idx+1],
                              preserve_range=True).astype(this_dtype)
        return new_img

    def laplacian(self, base: np.ndarray) -> List[np.ndarray]:
        raise RuntimeError("did not expect to run laplacian")

    def local_mean(self, base: np.ndarray) -> List[np.ndarray]:
        raise RuntimeError("did not expect to run local_mean")

    def gaussian(self, base: np.ndarray) -> List[np.ndarray]:

        # I wouldn't expect this to be okay, but apparently
        # this code never actually executes
        raise RuntimeError("gaussian")

    @classmethod
    def create_empty_pyramid(cls, base, downscale=2, donwscale_cutoff=128):
        raise NotImplementedError("base create_empty_pyramid")


class XYScaler(ScalerBase):
    """
    A scaler that ignores the z dimension, since it is
    so small relative to the other two dimensions in this initial
    dataset
    """

    def nearest(
            self,
            base: Union[np.ndarray, dask.array.Array]
    ) -> List[Union[np.ndarray, dask.array.Array]]:

        assert len(base.shape) == 3

        if isinstance(base, dask.array.Array):
            resize_func = dask_resize
            is_dask = True
        else:
            resize_func = skimage_resize
            is_dask = False

        list_of_nx_ny = self.create_empty_pyramid(
                               base,
                               downscale=self.downscale,
                               downscale_cutoff=self.downscale_cutoff)

        print(f"downscaling to {list_of_nx_ny} -- is_dask {is_dask}")

        results = dict()
        for nxny in list_of_nx_ny:
            if is_dask:
                chunks = tuple([max(1, n//4) for n in nxny])
                results[nxny] = dask.array.empty_like(
                                    None,
                                    name=f"{nxny}",
                                    dtype=float,
                                    chunks=chunks,
                                    shape=(nxny))
            else:
                results[nxny] = np.zeros(nxny, dtype=float)

        for iz in range(base.shape[2]):
            for nxny in list_of_nx_ny:
                img = resize_func(base[:, :, iz],
                                  (nxny[0], nxny[1]),
                                  preserve_range=True)
                results[nxny][:, :, iz] = img

        output = [base]
        print("done downscaling")
        return output + [results[key].astype(base.dtype)
                         for key in list_of_nx_ny]

    def create_empty_pyramid(
            self,
            base,
            downscale=2,
            downscale_cutoff=128):
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
            nx = base.shape[0]
            ny = base.shape[1]
            nz = base.shape[2]
            list_of_nx_ny = []

            cutoff = max(downscale_cutoff, base.shape[2])

            while nx > cutoff or ny > cutoff:
                nx = nx//downscale
                ny = ny//downscale
                key = (nx, ny, nz)
                list_of_nx_ny.append(key)

            self._list_of_nx_ny = list_of_nx_ny
            self.max_layer = len(self._list_of_nx_ny)

        return self._list_of_nx_ny


class XYZScaler(ScalerBase):
    """
    A scaler that ignores the z dimension, since it is
    so small relative to the other two dimensions in this initial
    dataset
    """

    def nearest(
            self,
            base: Union[np.ndarray, dask.array.Array]
    ) -> List[Union[np.ndarray, dask.array.Array]]:

        assert len(base.shape) == 3

        if isinstance(base, dask.array.Array):
            resize_func = dask_resize
        else:
            resize_func = skimage_resize

        list_of_nx_ny = self.create_empty_pyramid(
                               base,
                               downscale=self.downscale,
                               downscale_cutoff=self.downscale_cutoff)

        print(f"downscaling to {list_of_nx_ny}")

        results = dict()
        for nxyz in list_of_nx_ny:
            img = resize_func(base[:, :, :],
                              nxyz,
                              preserve_range=True)
            results[nxyz] = img

        output = [base]
        print("done downscaling")
        return output + [results[key].astype(base.dtype)
                         for key in list_of_nx_ny]


    @classmethod
    def create_empty_pyramid(
            cls,
            base,
            downscale=2,
            downscale_cutoff=128):
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
        if not hasattr(self, '_list_of_nx_ny'):
            nx = base.shape[0]
            ny = base.shape[1]
            nz = base.shape[2]
            list_of_nx_ny = []
            while max(nx, ny, nz) > downscale_cutoff:
                nx = nx//downscale
                ny = ny//downscale
                nz = nz//downscale
                key = (nx, ny, nz)
                list_of_nx_ny.append(key)

            self._list_of_nx_ny = list_of_nx_ny
            self.max_layer = len(list_of_nx_ny)

        return self._list_of_nx_ny
