from typing import List
import pathlib
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler
import zarr
import numpy as np
import SimpleITK
import argparse
import shutil
from skimage.transform import pyramid_gaussian
from skimage.transform import resize as skimage_resize

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



def gene_from_fname(fname):
    gene = fname.name.split('_')[1]
    return gene

def main():
    default_input = '/allen/programs/celltypes/workgroups/rnaseqanalysis/'
    default_input += 'mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_1/'
    default_input += 'alignment/mouse1_warpedToCCF_20220906'

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=default_input)
    parser.add_argument('--clobber', default=False, action='store_true')
    args = parser.parse_args()

    assert args.output_dir is not None
    assert args.input_dir is not None

    output_dir = pathlib.Path(args.output_dir)
    input_dir = pathlib.Path(args.input_dir)
    if output_dir.exists():
        if not args.clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            shutil.rmtree(output_dir)
    assert not output_dir.exists()
    output_dir.mkdir()
    assert output_dir.is_dir()

    assert input_dir.is_dir()

    fname_list = [n for n in input_dir.rglob('*nii.gz')]

    store = parse_url(output_dir, mode="w").store
    root_group = zarr.group(store=store)

    fname_list.sort()

    gene_to_path = dict()
    for fname in fname_list:
        gene = gene_from_fname(fname)
        if gene in gene_to_path:
            msg = f"{gene} appears twice\n"
            msg += f"{fname}\n"
            msg += f"{gene_to_path[gene]}"
            raise RuntimeError(msg)
        gene_to_path[gene] = fname

    gene_list = list(gene_to_path.keys())
    gene_list.sort()
    downscale = 2

    for gene in gene_list:
        fname = gene_to_path[gene]
        print(f"processing {fname.name}")
        this_group = root_group.create_group(f"{gene}")
        img = SimpleITK.ReadImage(fname)

        arr = SimpleITK.GetArrayFromImage(img)
        arr = arr.transpose(2, 1, 0)
        shape = arr.shape

        x_scale = float(img.GetMetaData('pixdim[1]'))
        y_scale = float(img.GetMetaData('pixdim[2]'))
        z_scale = float(img.GetMetaData('pixdim[3]'))

        (_,
         list_of_nx_ny) = _create_empty_pyramid(
                              base=arr,
                              downscale=downscale)

        coord_transform = [[
            {'scale': [float(x_scale),
                       float(y_scale),
                       float(z_scale)],
             'type': 'scale'}]]

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

        write_image(
            image=arr,
            scaler=XYScaler(method='gaussian'),
            group=this_group,
            coordinate_transformations=coord_transform,
            axes=axes,
            storage_options={'chunks':(shape[0]//4,
                                       shape[1]//4,
                                       shape[2]//4)})

if __name__ == "__main__":
    main()
