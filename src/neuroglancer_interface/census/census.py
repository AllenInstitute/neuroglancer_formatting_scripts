import h5py
import json
import multiprocessing
import numpy as np
import SimpleITK
import time

from neuroglancer_interface.utils.utils import (
    print_timing)

from neuroglancer_interface.census.utils import (
    get_mask_lookup,
    census_from_NIFTI_and_mask)



def run_census(
        mask_config_list,
        data_config_list,
        h5_path,
        n_processors):
    """
    Parameters
    ----------
    mask_config_list:
        config list to be passed to get_mask_lookup
    data_config_list:
        config list for the data files
        List of dicts with
            "tag": useful name for this dataset
            "path": path to NIFTI file
    h5_path:
        Path to HDF5 file that needs to be created
    n_processors:
        Number of workers to spin up
    """

    census_dtype=float

    (structure_to_col,
     data_to_row) = create_h5_census_file(
                         h5_path=h5_path,
                         mask_config_list=mask_config_list,
                         data_config_list=data_config_list,
                         census_dtype=census_dtype)

    mask_lookup = get_mask_lookup(
        nifti_config_list=mask_config_list,
        n_processors=n_processors)

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    process_list = []

    n_per = max(1, len(data_config_list)//n_processors)
    for i_batch in range(n_processors):
        i0 = i_batch*n_per
        if i_batch == n_processors-1:
            i1 = len(data_config_list)
        else:
            i1 = i0 + n_per

        batch = data_config_list[i0:i1]

        p = multiprocessing.Process(
                target=census_worker,
                kwargs={
                    'mask_lookup': mask_lookup,
                    'nifti_config_list': batch,
                    'nifti_to_row': data_to_row,
                    'structure_to_col': structure_to_col,
                    'h5_path': h5_path,
                    'output_lock': output_lock,
                    'census_dtype': census_dtype})

        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


def create_h5_census_file(
        h5_path,
        mask_config_list,
        data_config_list,
        census_dtype=float):
    """
    Create empty HDF5 file to contain the census data

    Parameters
    ----------
    h5_path:
        Path to HDF5 file that needs to be created
    mask_config_list:
        config list to be passed to get_mask_lookup
    data_config_list:
        config list for the data files
        List of dicts with
            "tag": useful name for this dataset
            "path": path to NIFTI file
    census_dtype:
        Dtype for the counts produced by the census
        utils

    Returns
    -------
    structure_to_col
    tag_to_row
        dicts mapping the names of structures and datasets to their
        rows/columns in the arrays in the HDF5 files
    """
    # check uniqueness of structure names and data tags
    structure_names = set()
    for config in mask_config_list:
        s = config['structure']
        if s in structure_names:
            raise RuntimeError(
                f"structure {s} occurs multiple times in config list")
        structure_names.add(s)

    tag_names = set()
    for config in data_config_list:
        t = config['tag']
        if t in structure_names:
            raise RuntimeError(
                f"data tag {t} occurs multiple times in config list")
        tag_names.add(t)

    n_structures = len(mask_config_list)
    n_data = len(data_config_list)

    eg_data = data_config_list[0]
    eg_arr = SimpleITK.GetArrayFromImage(
        SimpleITK.ReadImage(data_config_list[0]['path']))
    n_planes = eg_arr.shape[0]

    structure_to_col = dict()
    for ii, config in enumerate(mask_config_list):
        structure_to_col[config['structure']] = ii
    data_to_row = dict()
    for ii, config in enumerate(data_config_list):
        data_to_row[config['tag']] = ii

    chunks = 1000

    with h5py.File(h5_path, 'w') as out_file:

        out_file.create_dataset(
            'structures',
            data=json.dumps(structure_to_col).encode('utf-8'))

        out_file.create_dataset(
            'datasets',
            data=json.dumps(data_to_row).encode('utf-8'))

        out_file.create_dataset(
            'counts',
            shape=(n_data, n_structures),
            chunks=(min(1000, n_data),
                    min(1000, n_structures)),
            compression='gzip',
            dtype=census_dtype)

        out_file.create_dataset(
            'max_voxel',
            shape=(n_data, n_structures, 3),
            chunks=(min(1000, n_data),
                    min(1000, n_structures), 3),
            compression='gzip',
            dtype=census_dtype)

        out_file.create_dataset(
            'per_slice',
            shape=(n_data, n_structures, n_planes),
            chunks=(min(1000, n_data),
                    min(1000, n_structures),
                    min(1000, n_planes)),
            compression='gzip',
            dtype=census_dtype)

    return structure_to_col, data_to_row


def census_worker(
        mask_lookup,
        nifti_config_list,
        nifti_to_row,
        structure_to_col,
        h5_path,
        output_lock,
        census_dtype=float):
    """
    Parameters
    ----------
    mask_lookup:
        result of get_mask_lookup
    nifti_config_list:
        List of dicts with
            "tag": useful name for this dataset
            "path": path to NIFTI file
    nifti_to_row:
        Dict mapping "tag" in nifti_config_list
        to row in the HDF5 data file
    structure_to_col:
        Dict mapping structure name to column in
        the HDF5 file
    h5_path:
        Path to HDF5 file where data will be stored
    output_lock:
        multiprocessing lock preventing multiple
        processes from writing to HDF5 at once
    census_dtype:
        dtype for census count data
    """
    t0 = time.time()

    n_structures = len(mask_lookup)

    counts_arr = np.zeros(n_structures, dtype=census_dtype)
    max_voxel_arr = np.zeros((n_structures, 3), dtype=np.uint)
    s = list(mask_lookup.keys())[0]
    n_planes = mask_lookup[s]['shape'][0]
    per_plane_arr = np.zeros((n_structures, n_planes), dtype=census_dtype)

    ct = 0
    n_tot = len(nifti_config_list)
    print_every = max(1, n_tot//10)

    for nifti_config in nifti_config_list:
        this_census = census_from_NIFTI_and_mask(
            nifti_path=nifti_config['path'],
            mask_lookup=mask_lookup)

        for structure in this_census:
            col = structure_to_col[structure]
            counts_arr[col] = this_census[structure]['counts']
            max_voxel_arr[col] = this_census[structure]['max_voxel']
            per_plane_arr[col] = this_census[structure]['per_plane']

        row = nifti_to_row[nifti_config["tag"]]
        with output_lock:
            with h5py.File(h5_path, 'a') as out_file:
                for structure in this_census:
                    col = structure_to_col[structure]
                    out_file['counts'][row, :] = counts_arr
                    out_file['max_voxel'][row, :, :] = max_voxel_arr
                    out_file['per_slice'][row, :, :] = per_plane_arr

        ct += 1
        if ct % print_every == 0:
            print_timing(
                t0=t0,
                i_chunk=ct,
                tot_chunks=n_tot,
                unit='min')
