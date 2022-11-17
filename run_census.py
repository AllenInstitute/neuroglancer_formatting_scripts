import os
from celltypes_utils import get_class_lookup
import pathlib
import zarr
import numpy as np
import argparse
import json
import SimpleITK
import multiprocessing
import time


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
    for child in ('classes', 'subclasses', 'clusters'):
        this_dir = celltypes_dir / child
        celltype_census[child] = census_from_mask_and_zarr_dir(
                            mask_pixel_lookup=structure_mask_lookup,
                            zarr_dir=this_dir,
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
                raise RuntimeError(f"two results for {k}")
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
        for child in ('classes', 'subclasses', 'clusters'):
            result[human_name]['celltypes'][child] = dict()
            for class_name in census['celltypes'][child]:
                this_census = census['celltypes'][child][class_name]['census'][struct_name]
                result[human_name]['celltypes'][child][class_name] = this_census
                zarr_path = census['celltypes'][child][class_name]['zarr_path']
                metadata[f"{child}/{class_name}"] = zarr_path

    return result, metadata


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

    (subclass_to_clusters,
     class_to_clusters,
     valid_clusters,
     desanitizer) = get_class_lookup(args.annotation_path)
    print("got class lookup")

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