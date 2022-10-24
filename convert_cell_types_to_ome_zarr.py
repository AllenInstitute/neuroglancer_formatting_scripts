import pathlib
import argparse
import time
from data_utils import (
    write_nii_file_list_to_ome_zarr,
    write_summed_nii_files_to_group)


def sanitize_cluster_name(name):
    for bad_char in (' ', '/'):
        name = name.replace(bad_char, '_')
    return name

def get_class_lookup(
        anno_path):
    """
    returns subclass_to_clusters and class_to_clusters which
    map the names of classes to lists of the names of clusters
    therein

    also return a set containing all of the valid cluster names
    """

    anno_path = pathlib.Path(anno_path)
    if not anno_path.is_file():
        raise RuntimeError(f"{anno_path} is not a file")

    subclass_to_clusters = dict()
    class_to_clusters = dict()
    valid_clusters = set()

    with open(anno_path, "r") as in_file:
        header = in_file.readline()
        for line in in_file:
            params = line.replace('"', '').strip().split(',')
            assert len(params) == 4
            cluster_name = sanitize_cluster_name(params[1])
            valid_clusters.add(cluster_name)
            subclass_name = sanitize_cluster_name(params[2])
            class_name = sanitize_cluster_name(params[3])

            if subclass_name not in subclass_to_clusters:
                subclass_to_clusters[subclass_name] = []
            if class_name not in class_to_clusters:
                class_to_clusters[class_name] = []

            subclass_to_clusters[subclass_name].append(cluster_name)
            class_to_clusters[class_name].append(cluster_name)

    return subclass_to_clusters, class_to_clusters, valid_clusters


def write_summed_object(
        cluster_to_path,
        obj_to_clusters,
        root_group,
        downscale=2,
        prefix=None):

    key_list = list(obj_to_clusters.keys())
    key_list.sort()

    if prefix is not None:
        parent_group = root_group.create_group(prefix)
    else:
        parent_group = root_group

    for key in key_list:
        cluster_list = obj_to_clusters[key]
        file_path_list = [cluster_to_path[c] for c in cluster_list
                          if c in cluster_to_path]

        if len(file_path_list) > 0:
            group_name = key
            this_group = parent_group.create_group(group_name)
            write_summed_nii_files_to_group(
                file_path_list=file_path_list,
                group=this_group,
                downscale=downscale)

            print(f"wrote group {key}")
    return root_group

def main():

    default_input = '/allen/programs/celltypes/workgroups/'
    default_input += 'rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/'
    default_input += 'mouse/atlas/mouse_1/alignment/warpedCellTypes_Mouse1'

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=default_input)
    parser.add_argument('--annotation_path', type=str, default=default_anno)
    parser.add_argument('--clobber', default=False, action='store_true')
    parser.add_argument('--downscale', type=int, default=2)
    args = parser.parse_args()

    assert args.output_dir is not None
    assert args.input_dir is not None

    output_dir = pathlib.Path(args.output_dir)
    input_dir = pathlib.Path(args.input_dir)

    assert input_dir.is_dir()

    suffix = "_AppliedWarpAllSlc.nii.gz"

    (subclass_to_clusters,
     class_to_clusters,
     valid_clusters) = get_class_lookup(args.annotation_path)

    fpath_list = [n for n in input_dir.rglob('*nii.gz')]
    fpath_list.sort()
    cluster_name_list = []
    cluster_to_path = dict()
    for fpath in fpath_list:
        fname = fpath.name
        params = fname.split('_')
        cluster_name = fname.replace(f"{params[0]}_", "")
        cluster_name = cluster_name.replace(suffix, "")
        assert cluster_name in valid_clusters
        cluster_name_list.append(cluster_name)
        cluster_to_path[cluster_name] = fpath

    t0 = time.time()
    root_group = write_nii_file_list_to_ome_zarr(
            file_path_list=fpath_list,
            group_name_list=cluster_name_list,
            output_dir=args.output_dir,
            downscale=args.downscale,
            clobber=args.clobber,
            prefix="clusters")
    duration = time.time()-t0
    print(f"clusters took {duration:.2e}")

    root_group = write_summed_object(
            cluster_to_path=cluster_to_path,
            obj_to_clusters=subclass_to_clusters,
            root_group=root_group,
            downscale=args.downscale,
            prefix="subclasses")

    root_group = write_summed_object(
            cluster_to_path=cluster_to_path,
            obj_to_clusters=class_to_clusters,
            root_group=root_group,
            downscale=args.downscale,
            prefix="classes")

if __name__ == "__main__":
    main()
