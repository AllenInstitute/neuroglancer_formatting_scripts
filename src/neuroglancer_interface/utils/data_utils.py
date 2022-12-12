from typing import List, Any
import pandas as pd
import numpy as np
import SimpleITK
import pathlib
import shutil
import time
import zarr
from numcodecs import blosc
import multiprocessing
from ome_zarr.io import parse_url

from ome_zarr.writer import write_image
from neuroglancer_interface.utils.multiprocessing_utils import (
    _winnow_process_list)
from neuroglancer_interface.classes.downscalers import (
    XYScaler)


blosc.use_threads = False


def create_root_group(
        output_dir,
        clobber=False):

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    if output_dir.exists():
        if not clobber:
            raise RuntimeError(f"{output_dir} exists")
        else:
            print(f"cleaning out {output_dir}")
            shutil.rmtree(output_dir)
            print("done cleaning")

    assert not output_dir.exists()
    output_dir.mkdir()
    assert output_dir.is_dir()

    store = parse_url(output_dir, mode="w").store
    root_group = zarr.group(store=store)

    return root_group


def write_nii_file_list_to_ome_zarr(
        file_path_list,
        group_name_list,
        output_dir,
        downscale=2,
        n_processors=4,
        clobber=False,
        prefix=None,
        root_group=None,
        metadata_collector=None,
        DownscalerClass=XYScaler):
    """
    Convert a list of nifti files into OME-zarr format

    Parameters
    -----------
    file_path_list: List[pathlib.Path]
        list of paths to files to be written

    group_name_list: List[str]
        list of the names of the OME-zarr groups to be written
        (same order as file_path_list)

    output_dir: pathlib.Path
        The directory where the parent OME-zarr group will be
        written

    downscale: int
        Factor by which to downscale images as you are writing
        them (i.e. when you zoom out one level, by how much
        are you downsampling the pixels)

    n_processors: int
        Number of independent processes to use.

    clobber: bool
        If False, do not overwrite an existing group (throw
        an exception instead)

    prefix: str
        optional sub-group in which all data is written
        (i.e. option to write groups to output_dir/prefix/)

    root_group:
        Optional root group into which to write ome-zarr
        group. If None, will be created.

    Returns
    -------
    the root group

    Notes
    -----
    Importing zarr causes multiprocessing to emit a warning about
    leaked semaphore objects. *Probably* this is fine. It's just
    scary. The zarr developers are working on this

    https://github.com/zarr-developers/numcodecs/issues/230
    """
    t0 = time.time()
    if not isinstance(file_path_list, list):
        file_path_list = [file_path_list,]
    if not isinstance(group_name_list, list):
        group_name_list = [group_name_list,]

    if len(file_path_list) != len(group_name_list):
        msg = f"\ngave {len(file_path_list)} file paths but\n"
        msg += f"{len(group_name_list)} group names"

    if root_group is None:
        root_group = create_root_group(
                        output_dir=output_dir,
                        clobber=clobber)

    if prefix is not None:
        parent_group = root_group.create_group(prefix)
    else:
        parent_group = root_group

    if len(file_path_list) == 1:

        _write_nii_file_list_worker(
            file_path_list=file_path_list,
            group_name_list=group_name_list,
            root_group=parent_group,
            downscale=downscale,
            metadata_collector=metadata_collector,
            DownscalerClass=DownscalerClass)

    else:
        n_workers = max(1, n_processors-1)
        n_workers = min(n_workers, len(file_path_list))
        file_lists = []
        group_lists = []
        for ii in range(n_workers):
            file_lists.append([])
            group_lists.append([])

        for ii in range(len(file_path_list)):
            jj = ii % n_workers
            file_lists[jj].append(file_path_list[ii])
            group_lists[jj].append(group_name_list[ii])

        process_list = []
        for ii in range(n_workers):
            p = multiprocessing.Process(
                    target=_write_nii_file_list_worker,
                    kwargs={'file_path_list': file_lists[ii],
                            'group_name_list': group_lists[ii],
                            'root_group': parent_group,
                            'downscale': downscale,
                            'metadata_collector': metadata_collector,
                            'DownscalerClass': DownscalerClass})
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

    duration = time.time() - t0
    if prefix is not None:
        print(f"{prefix} took {duration:.2e} seconds")

    return root_group


def _write_nii_file_list_worker(
        file_path_list,
        group_name_list,
        root_group,
        downscale,
        metadata_collector=None,
        DownscalerClass=XYScaler):
    """
    Worker function to actually convert a subset of nifti
    files to OME-zarr

    Parameters
    ----------
    file_path_list: List[pathlib.Path]
        List of paths to nifti files to convert

    group_name_list: List[str]
        List of the names of the OME-zarr groups to
        write the nifti files to

    root_group:
        The parent group object as created by ome_zarr

    downscale: int
        The factor by which to downscale the images at each
        level of zoom.
    """

    for f_path, grp_name in zip(file_path_list,
                                group_name_list):
        write_nii_to_group(
            root_group=root_group,
            group_name=grp_name,
            nii_file_path=f_path,
            downscale=downscale,
            metadata_collector=metadata_collector,
            DownscalerClass=DownscalerClass)


def write_nii_to_group(
        root_group,
        group_name,
        nii_file_path,
        downscale,
        transpose=True,
        metadata_collector=None,
        metadata_key=None,
        DownscalerClass=XYScaler):
    """
    Write a single nifti file to an ome_zarr group

    Parameters
    ----------
    root_group:
        the ome_zarr group that the new group will
        be a child of (an object created using the ome_zarr
        library)

    group_name: str
        is the name of the group being created for this data

    nii_file_path: Pathlib.path
        is the path to the nii file being written

    downscale: int
        How much to downscale the image by at each level
        of zoom.
    """
    global_t0 = time.time()
    if group_name is not None:
        this_group = root_group.create_group(f"{group_name}")
    else:
        this_group = root_group
    t0 = time.time()
    img = SimpleITK.ReadImage(nii_file_path)
    arr = get_array_from_img(
                img,
                transpose=transpose)
    dur_read = time.time()-t0

    (x_scale,
     y_scale,
     z_scale) = get_scales_from_img(img)

    t0 = time.time()
    if metadata_collector is not None:

        other_metadata = {
            'x_mm': x_scale,
            'y_mm': y_scale,
            'z_mm': z_scale,
            'path': str(nii_file_path.resolve().absolute())}

        metadata_collector.collect_metadata(
            data_array=arr,
            other_metadata=other_metadata,
            metadata_key=group_name)
    dur_meta = time.time()-t0


    write_array_to_group(
        arr=arr,
        group=this_group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        downscale=downscale,
        DownscalerClass=DownscalerClass)

    dur_all = time.time()-global_t0
    print(f"wrote {nii_file_path} to {group_name}; timing: "
          f"read {dur_read:.2e} meta {dur_meta:.2e} "
          f"all {dur_all:.2e}")

def get_array_from_img(img, transpose=True):
    """
    Takes a SimpleITK img;
    Returns numpy arry with axes transposed as we want them

    """
    arr = SimpleITK.GetArrayFromImage(img)
    if transpose:
        arr = arr.transpose(2, 1, 0)
    return arr

def get_scales_from_img(img):
    """
    Takes in a SimpleITK image;
    returns (x_scale, y_scale, z_scale)
    """
    if 'pixdim[1]' in img.GetMetaDataKeys():
        x_scale = float(img.GetMetaData('pixdim[1]'))
        y_scale = float(img.GetMetaData('pixdim[2]'))
        z_scale = float(img.GetMetaData('pixdim[3]'))
    else:
        x_scale = 0.01
        y_scale = 0.01
        z_scale = 0.01
    return (x_scale, y_scale, z_scale)


def write_summed_nii_files_to_group(
        file_path_list,
        group,
        downscale = 2,
        transpose=True,
        DownscalerClass=XYScaler):
    """
    Sum the arrays in all of the files in file_path list
    into a single array and write that to the specified
    OME-zarr group

    downscale sets the amount by which to downscale the
    image at each level of zoom
    """

    main_array = None
    for file_path in file_path_list:
        img = SimpleITK.ReadImage(file_path)

        this_array = get_array_from_img(
                        img,
                        transpose=transpose)

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
            msg += f"{this_x_scale}, {this_y_scale}, {this_z_scale})\n"
            msg += "cannot sum"
            raise RuntimeError

        main_array += this_array

    write_array_to_group(
        arr=main_array,
        group=group,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=z_scale,
        downscale=downscale,
        DownscalerClass=DownscalerClass)


def write_array_to_group(
        arr: np.ndarray,
        group: Any,
        x_scale: float,
        y_scale: float,
        z_scale: float,
        downscale: int = 1,
        DownscalerClass=XYScaler):
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
    """

    # neuroglancer does not support 64 bit floats
    arr = arr.astype(np.float32)

    shape = arr.shape

    coord_transform = [[
        {'scale': [x_scale,
                   y_scale,
                   z_scale],
         'type': 'scale'}]]

    if downscale > 1:
        (_,
         list_of_nx_ny) = DownscalerClass.create_empty_pyramid(
                              base=arr,
                              downscale=downscale)

        del _

        for nxny in list_of_nx_ny:
            this_coord = [{'scale': [x_scale*arr.shape[0]/nxny[0],
                                     y_scale*arr.shape[1]/nxny[1],
                                     z_scale*arr.shape[2]/nxny[2]],
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
        scaler = DownscalerClass(
                   method='gaussian',
                   downscale=downscale)
    else:
        scaler = None

    chunk_x = min(shape[0]//4, 64)
    chunk_y = min(shape[1]//4, 64)
    chunk_z = min(shape[2]//4, 64)

    write_image(
        image=arr,
        scaler=scaler,
        group=group,
        coordinate_transformations=coord_transform,
        axes=axes,
        storage_options={'chunks':(chunk_x,
                                   chunk_y,
                                   chunk_z)})



def get_celltype_lookups_from_rda_df(
        csv_path):
    """
    Read a lookup mapping the integer index from a cell type
    name to its human readable form

    useful only if reading directly from the dataframe produced
    from Zizhen's .rda file. That is currently out of scope
    """
    df = pd.read_csv(csv_path)
    cluster = dict()
    level1 = dict()
    level2 = dict()
    for id_arr, label_arr, dest in [(df.Level1_id.values,
                                     df.Level1_label.values,
                                     level1),
                                    (df.Level2_id.values,
                                     df.Level2_label.values,
                                     level2),
                                    (df.cluster_id.values,
                                     df.cluster_label.values,
                                     cluster)]:
        for id_val, label_val in zip(id_arr, label_arr):
            if np.isnan(id_val):
                continue
            id_val = int(id_val)
            if id_val in dest:
                if dest[id_val] != label_val:
                    raise RuntimeError(
                        f"Multiple values for {id_val}\n"
                        f"{label_val}\n{dest[id_val]}\n"
                        f"{line}")
            else:
                dest[id_val] = label_val

    return {"cluster": cluster,
            "Level1": level1,
            "Level2": level2}
