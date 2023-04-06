from typing import List
import h5py
import json
import pathlib
import SimpleITK
import numpy as np
import tempfile

from neuroglancer_interface.utils.utils import (
    mkstemp_clean,
    _clean_up,
    get_prime_factors)

from neuroglancer_interface.utils.ccf_utils import (
    get_labels,
    format_labels,
    get_dummy_labels,
    downsample_segmentation_array)

from neuroglancer_interface.compression.utils import (
    compress_ccf_data)

from neuroglancer_interface.classes.nifti_array import (
    get_nifti_obj)


def write_out_ccf(
        segmentation_path_list: List[pathlib.Path],
        label_path: pathlib.Path,
        output_dir: pathlib.Path,
        use_compression=False,
        compression_blocksize=64,
        chunk_size=(256, 256, 256),
        do_transposition=False,
        tmp_dir=None,
        downsampling_cutoff=64) -> None:
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

    compression_blocksize:
        The data will be compressed into blocks of size
        (compression_blocksize, compression_blocksize). These blocks
        are internal to individual compressed files.

    chunk_size:
        Individual files on disk will contain voxel chunks of this size

    do_transposition:
        If True, transpose the NIFTI volumes so that
        (x, y, z) -> (z, y, x)

    tmp_dir:
        Directory where temporary HDF5 files are written during
        downsampling

    downsampling_cutoff:
        if applicable, do not downsample so far that any dimension
        is smaller than this value

    Returns
    -------
    None
        Data is written to output_dir in correct format
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)

    label_path = pathlib.Path(label_path)
    segmentation_path_list = [
        pathlib.Path(s) for s in segmentation_path_list]

    if not output_dir.exists():
        output_dir.mkdir()

    parent_info = create_info_dict(
            segmentation_path_list=segmentation_path_list,
            use_compression=use_compression,
            compression_blocksize=compression_blocksize,
            chunk_size=chunk_size,
            do_transposition=do_transposition,
            tmp_dir=tmp_dir,
            downsampling_cutoff=downsampling_cutoff)

    for scale_metadata in parent_info['scales']:
        do_chunking(metadata=scale_metadata,
                    parent_output_dir=output_dir,
                    do_transposition=do_transposition)

    if label_path is not None:
        label_path = pathlib.Path(label_path)
        labels = format_labels(get_labels(label_path))
    else:
        labels = format_labels(get_dummy_labels(segmentation_path_list))

    seg_dir = output_dir / parent_info['segment_properties']
    seg_dir.mkdir(exist_ok=True)
    with open(seg_dir / 'info', 'w') as out_file:
        out_file.write(json.dumps(labels, indent=2))

    with open(output_dir / 'info', 'w') as out_file:
        out_file.write(json.dumps(parent_info, indent=2))

    _clean_up(tmp_dir)

def do_chunking(
        metadata: dict,
        parent_output_dir: pathlib.Path,
        do_transposition: bool = False) -> None:
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

    do_transposition:
        If True, transpose the NIFTI volumes so that
        (x, y, z) -> (z, y, x)

    Returns
    -------
    None
    """

    if metadata['encoding'] == 'raw':
        use_compression = False
    elif metadata['encoding'] == 'compressed_segmentation':
        use_compression = True
        blocksize = metadata['compressed_segmentation_block_size'][0]
    else:
        raise RuntimeError(
            f"cannot parse encoding: {metadata['encoding']}")

    file_path = pathlib.Path(metadata['local_file_path'])
    if not file_path.is_file():
        raise RuntimeError(f"{file_path} is not a file")

    if file_path.name.endswith('.nii') or file_path.name.endswith('.nii.gz'):
        sitk_arr = _get_array_from_sitk(
            file_path,
            do_transposition=do_transposition)
    elif file_path.name.endswith('.h5'):
        sitk_arr = _get_array_from_h5(file_path)
    else:
        raise RuntimeError(
            f"unclear how to get array from file {file_path}")

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
                this_data = sitk_arr[x0:x1, y0:y1, z0:z1]
                if use_compression:
                    compress_ccf_data(
                        data=this_data,
                        file_path=this_file,
                        blocksize=blocksize)
                else:
                    _write_chunk_uncompressed(
                            file_path=this_file,
                            data=this_data)


def _get_array_from_sitk(file_path, do_transposition=False):
    nii_obj = get_nifti_obj(file_path,
                            do_transposition=do_transposition)
    sitk_arr = nii_obj.get_channel('red')['channel']
    sitk_arr = np.round(sitk_arr).astype(np.uint16)
    return sitk_arr


def _get_array_from_h5(file_path):
    with h5py.File(file_path, 'r') as in_file:
        data = in_file['data'][()]
    return data


def _write_chunk_uncompressed(file_path, data):
    with open(file_path, "wb") as out_file:
        data = data.tobytes("F")
        out_file.write(data)


def create_info_dict(
        segmentation_path_list: List[pathlib.Path],
        use_compression=False,
        compression_blocksize=64,
        chunk_size=(256, 256, 256),
        do_transposition=False,
        downsampling_cutoff=64,
        tmp_dir=None) -> dict:
    """
    Create the dict that will be JSONized to make the info file.
    Return that dict.

    Note:
    -----
    do_transposition:
        If True, transpose the NIFTI volumes so that
        (x, y, z) -> (z, y, x)
    """

    scale_list = []
    size_list = []
    if len(segmentation_path_list) > 1:
        for pth in segmentation_path_list:
            this = get_scale_metadata(
                        segmentation_path=pth,
                        use_compression=use_compression,
                        compression_blocksize=compression_blocksize,
                        chunk_size=chunk_size,
                        do_transposition=do_transposition)
            scale_list.append(this)
    else:
        this = get_scale_metadata(
            segmentation_path=segmentation_path_list[0],
            use_compression=use_compression,
            compression_blocksize=compression_blocksize,
            chunk_size=chunk_size,
            do_transposition=do_transposition)

        scale_list.append(this)

        downsampled_metadata = get_scale_metadata_with_downsampling(
            segmentation_path=segmentation_path_list[0],
            tmp_dir=tmp_dir,
            downsample_min=downsampling_cutoff,
            use_compression=use_compression,
            chunk_size=chunk_size,
            do_transposition=do_transposition)

        scale_list += downsampled_metadata

    for this in scale_list:
        size_list.append(this['size'][0]*this['size'][1]*this['size'][2])

    # from finest to coarsest resolution
    size_list = np.array(size_list)
    sorted_dex = np.argsort(-1*size_list)
    scale_list = [scale_list[idx] for idx in sorted_dex]

    result = dict()
    result['type'] = 'segmentation'
    result['segment_properties'] = 'segment_properties'
    if use_compression:
        result['data_type'] = 'uint32'
    else:
        result['data_type'] = 'uint16'
    result['num_channels'] = 1
    result['scales'] = scale_list

    return result


def get_scale_metadata(
        segmentation_path,
        chunk_size=(256, 256, 256),
        use_compression=False,
        compression_blocksize=64,
        do_transposition=False) -> dict:
    """
    Get the dict representing a single scale of a segmentation volume

    These need to be ordered from native resolution to zoomed out resolution
    """
    segmentation_path  = pathlib.Path(segmentation_path)
    nii_obj = get_nifti_obj(segmentation_path,
                            do_transposition=do_transposition)
    scale_mm = nii_obj.scales
    img_shape = nii_obj.shape

    result = dict()
    result['chunk_sizes'] = [chunk_size]
    if not use_compression:
        result['encoding'] = 'raw'
    else:
        result['encoding'] = 'compressed_segmentation'
        result['compressed_segmentation_block_size'] = [
            compression_blocksize,
            compression_blocksize,
            compression_blocksize]

    mm_to_nm = 10**6
    x_nm = int(mm_to_nm*scale_mm[0])
    y_nm = int(mm_to_nm*scale_mm[1])
    z_nm = int(mm_to_nm*scale_mm[2])

    result['key'] = f"{x_nm}_{y_nm}_{z_nm}"

    result['resolution'] = (x_nm, y_nm, z_nm)
    result['size'] = img_shape
    result['local_file_path'] = str(segmentation_path.resolve().absolute())

    return result


def get_scale_metadata_with_downsampling(
        segmentation_path,
        tmp_dir,
        downsample_min=64,
        chunk_size=(256, 256, 256),
        use_compression=False,
        compression_blocksize=64,
        do_transposition=False) -> dict:
    """
    Get the dict representing a single scale of a segmentation volume
    generated by downscaling a baseline segmentation at segmentation_path

    These need to be ordered from native resolution to zoomed out resolution

    Return a list of dicts representing the downsampled arrays, which have
    been written to tempfile.
    """
    segmentation_path  = pathlib.Path(segmentation_path)
    nii_obj = get_nifti_obj(segmentation_path,
                            do_transposition=do_transposition)
    scale_mm = nii_obj.scales
    img_shape = nii_obj.shape

    downsample_pyramid = _create_pyramid_of_ccf_downsamples(
        baseline_shape=img_shape,
        downsample_cutoff=downsample_min)

    baseline_array = _get_array_from_sitk(
        segmentation_path,
        do_transposition=do_transposition)

    config_list = []
    for downsample_by in downsample_pyramid:

        new_array = downsample_segmentation_array(
            arr=baseline_array,
            downsample_by=downsample_by)

        tmp_path = mkstemp_clean(
            dir=tmp_dir,
            suffix='.h5')
        tmp_path = pathlib.Path(tmp_path)
        with h5py.File(tmp_path, 'w') as out_file:
            out_file.create_dataset(
                'data',
                data=new_array)

        result = dict()
        result['chunk_sizes'] = [chunk_size]
        if not use_compression:
            result['encoding'] = 'raw'
        else:
            result['encoding'] = 'compressed_segmentation'
            result['compressed_segmentation_block_size'] = [
                compression_blocksize,
                compression_blocksize,
                compression_blocksize]

        mm_to_nm = 10**6
        x_nm = int(mm_to_nm*scale_mm[0]*downsample_by[0])
        y_nm = int(mm_to_nm*scale_mm[1]*downsample_by[1])
        z_nm = int(mm_to_nm*scale_mm[2]*downsample_by[2])

        result['key'] = f"{x_nm}_{y_nm}_{z_nm}"

        result['resolution'] = (x_nm, y_nm, z_nm)
        result['size'] = new_array.shape
        result['local_file_path'] = str(tmp_path.resolve().absolute())
        config_list.append(result)

    return config_list


def _create_pyramid_of_ccf_downsamples(
        baseline_shape,
        downsample_cutoff):
    """
    Create list of downsample_by tuples provided that no dimension
    goes below downsample_cutoff
    """
    # grab odd prime factors of the dimensions of the array
    factors = []
    for ii in range(3):
        these = [n for n in get_prime_factors(baseline_shape[ii])
                 if n%2==1]
        factors.append(these)

    current_downsample = [1, 1, 1]
    keep_going = True
    pyramid = []
    while keep_going:
        keep_going = False

        this = []

        for ii in range(3):
            if len(factors[ii]) == 0:
                this.append(1)
                continue

            candidate = current_downsample[ii]*factors[ii][0]
            if baseline_shape[ii] // candidate >= downsample_cutoff:
                factors[ii].pop(0)
                current_downsample[ii] = candidate
                keep_going = True
            else:
                candidate = current_downsample[ii]
            this.append(candidate)
        if keep_going:
            pyramid.append(tuple(this))
    return pyramid
