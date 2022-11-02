import dominate
import dominate.tags
import numpy as np
import json
import argparse
import zarr

import pathlib
from create_celltypes_url import create_celltypes_url
from celltypes_utils import get_class_lookup
from html_utils import write_basic_table

from botocore import UNSIGNED
from botocore.client import Config
import boto3


def get_ct_data(
        data_dir,
        celltype):

    print(data_dir)
    print(celltype)
    print((data_dir / celltype).is_dir())
    arr = np.array(zarr.open(data_dir / celltype, 'r')['0'])
    plane_sums = np.sum(arr, axis=(0,1))
    max_z = np.argmax(plane_sums)
    all_ct = np.sum(plane_sums)

    # units are counts per 10x10x25 slice
    # these voxels are each 10x10x100
    # so multiply each voxel's count value by 4
    all_ct *= 4.0

    print(f"read ct data for {celltype}")
    return (max_z, all_ct)

def find_valid_celltypes(
        bucket,
        subclass_list,
        class_list,
        cluster_list,
        pass_all=False,
        data_dir=None):
    """
    Determine which cell types have actually been loaded into S3
    """
    if not pass_all and data_dir is None:
        s3_client = boto3.client(
                        's3',
                        config=Config(signature_version=UNSIGNED))

    valid_celltypes = []

    for (child, child_list) in [('subclasses', subclass_list),
                                ('classes', class_list),
                                ('clusters', cluster_list)]:
        for this_type in child_list:
            type_key = f"{child}/{this_type}"
            if pass_all:
                valid_celltypes.append(type_key)
            elif data_dir is not None:
                full_dir = data_dir / type_key
                if full_dir.is_dir():
                    valid_celltypes.append(type_key)
            else: 
                test_key = f"{type_key}/.zattrs"
                response = s3_client.list_objects_v2(
                        Bucket=bucket,
                        Prefix=test_key)
                if response['KeyCount'] > 0:
                    valid_celltypes.append(type_key)
    return valid_celltypes


def idx_from_cluster_name(cluster_name):
    params = cluster_name.split('_')
    if len(params) == 0:
        return 0
    try:
        idx = int(params[0])
    except ValueError:
        idx = 0
    return idx


def write_celltypes_html(
        output_path=None,
        annotation_path=None,
        bucket="mouse1-celltypes-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        template_bucket="mouse1-template-prototype",
        range_max=0.1,
        color='green',
        pass_all=False,
        data_dir=None):

    (subclass_to_clusters,
     class_to_clusters,
     cluster_set,
     desanitizer) = get_class_lookup(annotation_path)

    subclass_list = list(subclass_to_clusters.keys())
    class_list = list(class_to_clusters.keys())
    cluster_list = list(cluster_set)
    for l in (subclass_list, class_list, cluster_list):
        l.sort()

    valid_celltypes = find_valid_celltypes(
                            bucket=bucket,
                            class_list=class_list,
                            subclass_list=subclass_list,
                            cluster_list=cluster_list,
                            pass_all=pass_all,
                            data_dir=data_dir)

    sort_by = []
    for celltype in valid_celltypes:
        this_type = celltype.split('/')[-1]
        actual_type = desanitizer[this_type]
        idx = idx_from_cluster_name(actual_type)
        sort_by.append(idx)
    sort_by = np.array(sort_by)
    valid_celltypes = np.array(valid_celltypes)
    sorted_dex = np.argsort(sort_by)
    valid_celltypes = valid_celltypes[sorted_dex]

    celltype_to_link = dict()
    celltype_to_cols = dict()

    for celltype in valid_celltypes:

        starting_position = None
        if data_dir is not None:
            (max_plane,
             total_cts) = get_ct_data(
                            data_dir=data_dir,
                            celltype=celltype)
            starting_position=[550, 550, max_plane]

        this_url = create_celltypes_url(
                        bucket=bucket,
                        celltype=celltype,
                        range_max=range_max,
                        color=color,
                        template_bucket=template_bucket,
                        segmentation_bucket=segmentation_bucket,
                        desanitizer=desanitizer,
                        starting_position=starting_position)

        hierarchy = celltype.split('/')[0]
        dirty = desanitizer[celltype.split('/')[-1]]

        celltype_to_link[celltype] = this_url
        these_cols = {'names': ['celltype_name', 'hierarchy'],
                      'values': [dirty, hierarchy]}

        if data_dir is not None:
            these_cols['names'].append('counts')
            these_cols['values'].append(f"{total_cts:.3e}")

        celltype_to_cols[celltype] = these_cols

    title = "Mouse1 cell type count maps"
    div_name = "celltype_maps"
    cls_name = "celltype_name"

    write_basic_table(
        output_path=output_path,
        title=title,
        key_to_link=celltype_to_link,
        key_order=valid_celltypes,
        div_name=div_name,
        key_to_other_cols=celltype_to_cols,
        search_by=['celltype_name',
                   'hierarchy'])


def main():

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, default=default_anno)
    parser.add_argument('--pass_all', default=False, action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = None
    if args.data_dir is not None:
        data_dir = pathlib.Path(args.data_dir)

    html_dir = pathlib.Path('html')
    write_celltypes_html(
        output_path=html_dir / 'mouse1_celltype_maps.html',
        annotation_path=pathlib.Path(args.annotation_path),
        pass_all=args.pass_all,
        data_dir=data_dir)
    print("wrote html")

if __name__ == "__main__":
    main()
