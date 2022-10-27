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


def find_valid_celltypes(
        bucket,
        subclass_list,
        class_list,
        cluster_list,
        pass_all=False):
    """
    Determine which cell types have actually been loaded into S3
    """
    if not pass_all:
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
                continue
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
                            pass_all=pass_all)

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
        this_url = create_celltypes_url(
                        bucket=bucket,
                        celltype=celltype,
                        range_max=range_max,
                        color=color,
                        template_bucket=template_bucket,
                        segmentation_bucket=segmentation_bucket,
                        desanitizer=desanitizer)

        hierarchy = celltype.split('/')[0]
        dirty = desanitizer[celltype.split('/')[-1]]

        celltype_to_link[celltype] = this_url
        these_cols = {'names': ['celltype_name', 'hierarchy'],
                      'values': [dirty, hierarchy]}
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
    default_data = '/allen/programs/celltypes/workgroups/'
    default_data += 'rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/'
    default_data += 'mouse/atlas/mouse_1/alignment/warpedCellTypes_Mouse1'

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, default=default_anno)
    parser.add_argument('--pass_all', default=False, action='store_true')
    parser.add_argument('--data_dir', type=str, default=default_data)
    args = parser.parse_args()

    data_dir = pathlib.Path(default_data)
    if not data_dir.is_dir():
        data_dir = None

    html_dir = pathlib.Path('html')
    write_celltypes_html(
        output_path=html_dir / 'mouse1_celltype_maps.html',
        annotation_path=pathlib.Path(args.annotation_path),
        pass_all=args.pass_all,
        data_dir=data_dir)
    print("wrote html")

if __name__ == "__main__":
    main()
