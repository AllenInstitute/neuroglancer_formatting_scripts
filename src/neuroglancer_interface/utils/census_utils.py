import numpy as np
import json
import SimpleITK
import pathlib
import multiprocessing
from neuroglancer_interface.utils.data_utils import (
    get_array_from_img)

def census_from_mask_lookup_and_arr(
        mask_lookup,
        data_arr):
    """
    Parameters
    ----------
    mask_lookup: dict
        maps some key to mask pixels (the result
        of running np.where on the mask array)

    data_arr: np.ndarray
        array that is the count data for this structure

    Returns
    -------
    Dict mapping 'counts' and 'max_voxel' to the total
    number of counts and the "brightest" voxel
    """

    result = dict()
    for mask_key in mask_lookup:
        mask_pixels = mask_lookup[mask_key]['mask']
        valid = data_arr[mask_pixels]
        total = valid.sum()
        idx = np.argmax(valid)
        voxel = [int(mask_pixels[ii][idx])
                 for ii in range(len(mask_pixels))]

        # need to transpose because of the way we
        # are transposing the data for display
        # in neuroglancer (see data_utils.get_array_from_img)
        voxel = [voxel[2], voxel[1], voxel[0]]

        this_result = {'counts': float(total),
                       'max_voxel': voxel}
        result[mask_key] = this_result
    return result



def get_structure_name_lookup(
        path_list):
    """
    Get the dict mapping structure ID to the human readable
    name

    Parameters
    ----------
    path_list: list
        list of paths to read
    """

    result = dict()
    for pth in path_list:
        pth = pathlib.Path(pth)
        if not pth.is_file():
            raise RuntimeError(f"{pth.resolve().absolute()} not a file")
        elif pth.name.endswith('json'):
            this_lookup = _get_structure_name_from_json(pth)
        elif pth.name.endswith('csv'):
            this_lookup = _get_structure_name_from_csv(pth)
        else:
            raise RuntimeError("do not know how to parse "
                               f"{pth.resolve().absolute()}")

        for k in this_lookup:
            if k in result and this_lookup[k] != result[k]:
                raise RuntimeError(f"two results for {k}\n"
                                   f"{this_lookup[k]}\n"
                                   f"{result[k]}")
            result[k] = this_lookup[k]
    return result




def _get_structure_name_from_csv(filepath):
    result = dict()
    with open(filepath, 'r') as in_file:
        id_idx = None
        name_idx = None
        header = in_file.readline()
        header = header.strip().split(',')
        for ii in range(len(header)):
            if header[ii] == 'id':
                assert id_idx is None
                id_idx = ii
            elif header[ii] == 'name':
                assert name_idx is None
                name_idx = ii
        if name_idx is None or id_idx is None:
            raise RuntimeError(
                "could not find 'id' and 'name' in \n"
                f"{header}")
        for line in in_file:
            params = line.strip().split(',')
            id_val = int(params[id_idx])
            name_val = params[name_idx]
            assert id_val not in result
            result[id_val] = name_val
    return result


def _get_structure_name_from_json(filepath):
    with open(filepath, 'rb') as in_file:
        json_data = json.load(in_file)

    result = dict()
    for element in json_data:
        id_val = int(element['id'])
        name_val = element['acronym']
        assert id_val not in result
        result[id_val] = name_val
    return result


def reformat_census(census, structure_name_lookup):

    metadata = dict()
    result = dict()
    for gene_name in census['genes']:
        zarr_path = census['genes'][gene_name]['zarr_path']
        metadata[gene_name] = zarr_path
        for struct_name in census['genes'][gene_name]['census']:
            human_name = structure_name_lookup[struct_name]
            if human_name not in result:
                result[human_name] = dict()
                result[human_name]['genes'] = dict()
            this_census = census['genes'][gene_name]['census'][struct_name]
            result[human_name]['genes'][gene_name] = this_census

    for struct_name in census['genes'][gene_name]['census']:
        human_name = structure_name_lookup[struct_name]
        result[human_name]['celltypes'] = dict()
        for child in result[human_name].keys():
            result[human_name]['celltypes'][child] = dict()
            for class_name in census['celltypes'][child]:
                this_census = census['celltypes'][child][class_name]['census'][struct_name]
                result[human_name]['celltypes'][child][class_name] = this_census
                zarr_path = census['celltypes'][child][class_name]['zarr_path']
                metadata[f"{child}/{class_name}"] = zarr_path

    return result, metadata


def _get_mask_lookup_worker(file_path_list, output_dict, lock):
    """
    get a dict mapping integer ID to mask pixels

    Parametrs
    ---------
    mask_dir: pathlib.Path
        directory to scann for all nii.gz files

    Returns
    -------
    dict
    """

    result = dict()
    for file_path in file_path_list:
        id_val = int(file_path.name.split('_')[0])
        mask = get_array_from_img(
                    SimpleITK.ReadImage(file_path))
        mask_pixels = np.where(mask==1)
        result[id_val] = {'mask': mask_pixels,
                          'path': str(file_path.resolve().absolute())}

    with lock:
        for id_val in result:
            output_dict[id_val] = result[id_val]

def get_mask_lookup(mask_dir, n_processors):
    """
    get a dict mapping integer ID to mask pixels

    Parametrs
    ---------
    mask_dir: pathlib.Path
        directory to scann for all nii.gz files

    n_processors: int

    Returns
    -------
    dict
    """
    if not isinstance(mask_dir, pathlib.Path):
        mask_dir = pathlib.Path(mask_dir)
    file_path_list = [n for n in mask_dir.rglob('*nii.gz')][:10]
    id_set = set([int(f.name.split('_')[0])
                  for f in file_path_list])
    assert len(id_set) == len(file_path_list)

    file_path_list.sort()

    mgr = multiprocessing.Manager()
    result = mgr.dict()
    lock = mgr.Lock()

    sub_lists = []
    for ii in range(n_processors):
        sub_lists.append([])
    for ii in range(len(file_path_list)):
        sub_lists[ii%n_processors].append(file_path_list[ii])
    process_list = []
    for ii in range(n_processors):
        p = multiprocessing.Process(
                target=_get_mask_lookup_worker,
                args=(sub_lists[ii],
                      result,
                      lock))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    return dict(result)
