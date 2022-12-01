import os
import pathlib
import zarr
import numpy as np
import argparse
import json
import SimpleITK
import multiprocessing
import time

from neuroglancer_interface.utils.celltypes_utils import (
    read_all_manifests)

from neuroglancer_interface.utils.census_utils import (
    census_from_mask_lookup_and_arr,
    reformat_census,
    get_structure_name_lookup)


def census_from_structure_lookup(
        structure_mask_lookup,
        mfish_dir,
        celltypes_dir,
        celltypes_desanitizer,
        n_processors):
    """
    Parameters
    ----------
    structure_mask_lookup: dict
        dict mapping some key to mask pixels
        (the result of running np.where) on the mask arrays

    mfish_dir: pathlib.Path
        path to the ome-zarr-ified mFISH counts

    celltypes_dir: pathlib.Path
        path to the ome-zarr-ified cell types counts

    celltypes_desanitizer: dict
        dict to convert machine-readable cell type name back
        to human readable form

    Return
    ------
    Dict containing the results of the census
    """

    result = dict()
    result['genes'] = census_from_mask_and_zarr_dir(
                        mask_pixel_lookup=structure_mask_lookup,
                        zarr_dir=mfish_dir,
                        desanitizer=None,
                        n_processors=n_processors)

    celltype_census = dict()
    celltype_children = [n for n in celltypes_dir.iterdir()
                         if n.is_dir()]
    for child in celltype_children:
        celltype_census[child.name] = census_from_mask_and_zarr_dir(
                            mask_pixel_lookup=structure_mask_lookup,
                            zarr_dir=child,
                            desanitizer=celltypes_desanitizer,
                            n_processors=n_processors)

    result['celltypes'] = celltype_census

    return result


def census_from_mask_and_zarr_dir(
        mask_pixel_lookup,
        zarr_dir,
        desanitizer=None,
        n_processors=4):
    """
    Loop through the subdirectories of the
    ome-zarr-ified data, performing the structure
    census on the contents.

    Parameters
    ----------
    mask_pixel_lookup: dict
        maps some key to the mask pixels
        (the result of running np.where on the
        mask array)

    zarr_dir: pathlib.Path
        dir containing ome-zarr-ified count data

    desanitizer: dict
       optional dict mapping zarr_dir subdir name
       to human readable form

    Returns
    -------
    Dict mapping human readable name from desnanitizer
    to census results

    Notes
    -----
    Because we transposed the data (2, 1, 0) when writing
    to ome-zarr, we undo that transpose when reading back
    in with zarr.
    """
    if not zarr_dir.is_dir():
        msg = f"\n{zarr_dir.resolve().absolute()} is not dir"
        raise RuntimeError(msg)

    sub_dir_list = [n for n in zarr_dir.iterdir() if n.is_dir()]
    for s in sub_dir_list:
        assert s.is_dir()
    sub_dir_list.sort()

    sub_lists = []
    for ii in range(n_processors):
        sub_lists.append([])
    for ii in range(len(sub_dir_list)):
        sub_lists[ii%n_processors].append(sub_dir_list[ii])

    mgr = multiprocessing.Manager()
    result = mgr.dict()
    lock = mgr.Lock()

    process_list = []
    for ii in range(n_processors):
        p = multiprocessing.Process(
                target=_census_from_mask_and_zarr_dir_worker,
                kwargs={'sub_dir_list': sub_lists[ii],
                        'mask_pixel_lookup': mask_pixel_lookup,
                        'desanitizer': desanitizer,
                        'output_dict': result,
                        'lock': lock})
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    return dict(result)


def _census_from_mask_and_zarr_dir_worker(
        sub_dir_list,
        mask_pixel_lookup,
        desanitizer,
        output_dict,
        lock):

    t0 = time.time()
    n_tot = len(sub_dir_list)
    ct = 0
    result = dict()
    for sub_dir in sub_dir_list:
        if desanitizer is not None:
            human_name = desanitizer[sub_dir.name]
        else:
            human_name = sub_dir.name

        if human_name in result:
            msg = f"two results for {human_name}"
            raise RuntimeError(msg)

        data_arr = np.array(
                zarr.open(sub_dir, 'r')['0']).transpose(2, 1, 0)
        this_census = census_from_mask_lookup_and_arr(
            mask_lookup=mask_pixel_lookup,
            data_arr=data_arr)

        result[human_name] = {'census': this_census,
                              'zarr_path': str(sub_dir.resolve().absolute())}

        ct += 1
        if ct%10 == 0:
            pid = os.getpid()
            duration = time.time()-t0
            per = duration/ct
            pred = per*n_tot
            remain = pred-duration
            with lock:
                print(f"{pid} -- {ct} in {duration:.2e} "
                      f"-- {remain:.2e} of {pred:.2e} remain")

    with lock:
        for k in result:
            output_dict[k] = result[k]


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
        mask = SimpleITK.GetArrayFromImage(
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
    file_path_list = [n for n in mask_dir.rglob('*nii.gz')]
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


def get_desanitizer(celltypes_dir):
    cell_type_list = read_all_manifests(celltypes_dir)
    desanitizer = dict()
    for cell_type in cell_type_list:
        m = cell_type['machine_readable']
        h = cell_type['human_readable']
        if m in desanitizer:
            if h != desanitizer[m]:
                raise RuntimeError(f"{m} occurs more than once")
        desanitizer[m] = h
    return desanitizer


def main():

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    default_mask = "/allen/programs/celltypes/workgroups/"
    default_mask += "rnaseqanalysis/mFISH/michaelkunst/"
    default_mask += "MERSCOPES/mouse/atlas/mouse_1/alignment/"
    default_mask += "RegPrelimDefNN_mouse1/iter0/structure_masks"

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dir', type=str, default=default_mask)
    parser.add_argument('--celltypes_dir', type=str, default=None)
    parser.add_argument('--mfish_dir', type=str, default=None)
    parser.add_argument('--annotation_path', type=str, default=default_anno)
    parser.add_argument('--structure_lookup', type=str, default=None)
    parser.add_argument('--n_processors', type=int, default=4)
    parser.add_argument('--output_path', type=str, default='census.json')
    args = parser.parse_args()

    mask_dir = pathlib.Path(args.mask_dir)
    celltypes_dir = pathlib.Path(args.celltypes_dir)
    mfish_dir = pathlib.Path(args.mfish_dir)

    for d in (mask_dir, celltypes_dir, mfish_dir):
        if not d.is_dir():
            msg = f"{d.resolve().absolute()} is not dir"
            raise RuntimeError(msg)

    desanitizer = get_desanitizer(celltypes_dir)

    if not isinstance(args.structure_lookup, list):
        structure_lookup_list = [args.structure_lookup]
    else:
        structure_lookup_list = args.structure_lookup

    structure_name_lookup = get_structure_name_lookup(
                                path_list=structure_lookup_list)
    print("got structure name lookup")

    mask_pixel_lookup = get_mask_lookup(mask_dir,
                            n_processors=args.n_processors)
    print(f"got mask pixel lookup -- {len(mask_pixel_lookup)}")

    census = census_from_structure_lookup(
        structure_mask_lookup=mask_pixel_lookup,
        mfish_dir=mfish_dir,
        celltypes_dir=celltypes_dir,
        celltypes_desanitizer=desanitizer,
        n_processors=args.n_processors)

    (census,
     zarr_paths) = reformat_census(
                census=census,
                structure_name_lookup=structure_name_lookup)

    final_census = dict()
    final_census['census'] = census
    final_census['zarr_paths'] = zarr_paths
    mask_paths = dict()
    for k in mask_pixel_lookup.keys():
        human_name = structure_name_lookup[k]
        assert human_name not in mask_paths
        mask_paths[human_name] = mask_pixel_lookup[k]['path']
    final_census['mask_paths'] = mask_paths

    with open(args.output_path, 'w') as out_file:
        out_file.write(json.dumps(final_census, indent=2))
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
