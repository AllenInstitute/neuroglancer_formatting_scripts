from typing import List
import json
import pathlib
import SimpleITK
import numpy as np
from neuroglancer_interface.utils.data_utils import get_scales_from_img
from neuroglancer_interface.utils.ccf_utils import (
    get_labels,
    format_labels)

def write_out_ccf(
        segmentation_path_list: List[pathlib.Path],
        label_path: pathlib.Path,
        output_dir: pathlib.Path) -> None:
    """
    Write CCF annotations to disk in neuroglancer-friendly format

    Parameters
    ----------
    segmentation_path_list:
        List of paths to CCF annotations at different scales (the nii.gz files)

    label_path:
        Path to the text file mapping uint16 to region name

    output_dir:
        The directory where output will be written.

    Returns
    -------
    None
        Data is written to output_dir in correct format
    """

    if not output_dir.exists():
        output_dir.mkdir()

    parent_info = create_info_dict(
            segmentation_path_list=segmentation_path_list)

    for scale_metadata in parent_info['scales']:
        do_chunking(metadata=scale_metadata,
                    parent_output_dir=output_dir)

    labels = format_labels(get_labels(label_path))
    seg_dir = output_dir / parent_info['segment_properties']
    seg_dir.mkdir(exist_ok=True)
    with open(seg_dir / 'info', 'w') as out_file:
        out_file.write(json.dumps(labels, indent=2))

    with open(output_dir / 'info', 'w') as out_file:
        out_file.write(json.dumps(parent_info, indent=2))

def do_chunking(
        metadata: dict,
        parent_output_dir: pathlib.Path) -> None:
    """
    Take the metadata created by get_scale_metadata and actually
    do the chunking of the CCF annotation file.

    Parameters
    ----------
    metadata:
        The info['scales'] element corresponding to the scale of CCF
        atlas being written

    parent_output_dir:
        The output dir for the entire CCF annotation. A sub-directory
        will be created where the chunked version of this annotation
        will be written.

    Returns
    -------
    None
    """

    file_path = pathlib.Path(metadata['local_file_path'])
    if not file_path.is_file():
        raise RuntimeError(f"{file_path} is not a file")

    sitk_img = SimpleITK.ReadImage(file_path)
    sitk_arr = SimpleITK.GetArrayFromImage(sitk_img).transpose(2, 1, 0)
    sitk_arr = np.round(sitk_arr).astype(np.uint16)

    if not sitk_arr.shape == metadata['size']:
        raise RuntimeError(
            f"array shape {sitk_arr.shape}\n"
            f"metadata says {metadata['size']}")

    output_dir = parent_output_dir / metadata['key']
    output_dir.mkdir(exist_ok=True)
    if not output_dir.is_dir():
        raise RuntimeError(f"{output_dir} is not a dir")

    dx = metadata['chunk_sizes'][0][0]
    dy = metadata['chunk_sizes'][0][1]
    dz = metadata['chunk_sizes'][0][2]
    for x0 in range(0, sitk_arr.shape[0], dx):
        x1 = min(sitk_arr.shape[0], x0+dx)
        for y0 in range(0, sitk_arr.shape[1], dy):
            y1 = min(sitk_arr.shape[1], y0+dy)
            for z0 in range(0, sitk_arr.shape[2], dz):
                z1 = min(sitk_arr.shape[2], z0+dz)
                name = f"{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"
                this_file = output_dir / name
                with open(this_file, "wb") as out_file:
                    this_data = sitk_arr[x0:x1, y0:y1, z0:z1]
                    this_data = this_data.tobytes("F")
                    out_file.write(this_data)
             

def create_info_dict(
        segmentation_path_list: List[pathlib.Path]) -> dict:
    """
    Create the dict that will be JSONized to make the info file.
    Return that dict.
    """

    scale_list = []
    size_list = []
    for pth in segmentation_path_list:
        this = get_scale_metadata(segmentation_path=pth)
        scale_list.append(this)
        size_list.append(this['size'][0]*this['size'][1]*this['size'][2])

    # from finest to coarsest resolution
    size_list = np.array(size_list)
    sorted_dex = np.argsort(-1*size_list)
    scale_list = [scale_list[idx] for idx in sorted_dex]

    result = dict()
    result['type'] = 'segmentation'
    result['segment_properties'] = 'segment_properties'
    result['data_type'] = 'uint16'
    result['num_channels'] = 1
    result['scales'] = scale_list

    return result


def get_scale_metadata(
        segmentation_path,
        chunk_size=(128, 128, 128)) -> dict:
    """
    Get the dict representing a single scale of a segmentation volume

    These need to be ordered from native resolution to zoomed out resolution
    """
    sitk_img = SimpleITK.ReadImage(segmentation_path)
    scale_mm = get_scales_from_img(sitk_img)

    # use image shape, wich is the transpose(2, 1, 0) of the
    # resulting numpy array
    img_shape = sitk_img.GetSize()

    voxel_offset = (0, 0, 0)

    result = dict()
    result['chunk_sizes'] = [chunk_size]
    result['encoding'] = 'raw'

    mm_to_nm = 10**6
    x_nm = int(mm_to_nm*scale_mm[0])
    y_nm = int(mm_to_nm*scale_mm[1])
    z_nm = int(mm_to_nm*scale_mm[2])

    result['key'] = f"{x_nm}_{y_nm}_{z_nm}"

    result['resolution'] = (x_nm, y_nm, z_nm)
    result['size'] = img_shape
    result['local_file_path'] = str(segmentation_path.resolve().absolute())

    return result
