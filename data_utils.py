from typing import List, Any
import numpy as np
import SimpleITK
import pathlib
import shutil
import zarr
from ome_zarr.scale import Scaler
from ome_zarr.io import parse_url
from skimage.transform import pyramid_gaussian
from skimage.transform import resize as skimage_resize
from ome_zarr.writer import write_image


def write_nii_file_list_to_ome_zarr(
        file_path_list,
        group_name_list,
        output_dir,
        downscale=2,
        clobber=False):
    """
    file_path_list -- list of paths to files to be written

    group_name_list -- list of names of groups for files

    output_dir -- dir for parent ome-zarr group

    clobber -- if False, do not overwrite
    """
    if not isinstance(file_path_list, list):
        file_path_list = [file_path_list,]
    if not isinstance(group_name_list, list):
        group_name_list = [group_name_list,]

    if len(file_path_list) != len(group_name_list):
        msg = f"\ngave {len(file_path_list)} file paths but\n"
        msg += f"{len(group_name_list)} group names"

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    if output_dir.exists():
        if not clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            shutil.rmtree(output_dir)
    assert not output_dir.exists()
    output_dir.mkdir()
    assert output_dir.is_dir()

    store = parse_url(output_dir, mode="w").store
    root_group = zarr.group(store=store)

    for f_path, grp_name in zip(file_path_list,
                                group_name_list):
        write_nii_to_group(
            root_group=root_group,
            group_name=grp_name,
            nii_file_path=f_path,
            downscale=downscale)


def write_nii_to_group(
        root_group,
        group_name,
        nii_file_path,
        downscale):
    """
    root_group is the ome_zarr group that the new group will
    be a child of.

    group_name is the name of the group being created for this data

    nii_file_path is the path to the nii file being written

    downscale is an int controlling downscaling
    """

    this_group = root_group.create_group(f"{group_name}")
    img = SimpleITK.ReadImage(nii_file_path)

    arr = get_array_from_img(img)

    (x_scale,
     y_scale,
     z_scale) = get_scales_from_img(img)

    write_array_to_group(
        arr=arr,
        group=this_group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        downscale=downscale)

    print(f"wrote {nii_file_path} to {group_name}")

def get_array_from_img(img):
    """
    Takes a SimpleITK img;
    Returns numpy arry with axes transposed as we want them
    """
    arr = SimpleITK.GetArrayFromImage(img)
    arr = arr.transpose(2, 1, 0)
    return arr

def get_scales_from_img(img):
    """
    Takes in a SimpleITK image;
    returns (x_scale, y_scale, z_scale)
    """
    x_scale = float(img.GetMetaData('pixdim[1]'))
    y_scale = float(img.GetMetaData('pixdim[2]'))
    z_scale = float(img.GetMetaData('pixdim[3]'))
    return (x_scale, y_scale, z_scale)


def write_summed_array_to_group(
        file_path_list,
        group,
        downscale = 2):
    """
    Sum the arrays in all of the files in file_path list
    into a single array and write that to the specified group
    """

    main_array = None
    for file_path in file_path_list:
        img = SimpleITK.ReadImage(file_path)

        this_array = get_array_from_img(img)

        (this_x_scale,
         this_y_scale,
         this_z_scale) = get_scales_from_img(img)

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
            msg += f"{this_x_scale}, {this_y_scale}, this_z_scale})\n"
            msg += "cannot sum"
            raise RuntimeError

        main_array += this_array

    write_array_to_group(
        arr=main_array,
        group=group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        downscale=downscale)


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
        return output + [results[key].astype(base.dtype)
                         for key in list_of_nx_ny]


    def gaussian(self, base: np.ndarray) -> List[np.ndarray]:

        # I wouldn't expect this to be okay, but apparently
        # this code never actually executes
        raise RuntimeError("gaussian")

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


