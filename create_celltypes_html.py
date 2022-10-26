import dominate
import dominate.tags
import json
import argparse

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
        cluster_list):
    """
    Determine which cell types have actually been loaded into S3
    """
    s3_client = boto3.client(
                    's3',
                    config=Config(signature_version=UNSIGNED))

    valid_celltypes = []

    for (child, child_list) in [('subclasses', subclass_list),
                                ('classes', class_list),
                                ('clusters', cluster_list)]:
        for this_type in child_list:
            type_key = f"{child}/{this_type}"
            test_key = f"{type_key}/.zattrs"
            response = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=test_key)
            if response['KeyCount'] > 0:
                valid_celltypes.append(type_key)
    return valid_celltypes


def write_celltypes_html(
        output_path=None,
        annotation_path=None,
        bucket="mouse1-celltypes-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        template_bucket="mouse1-template-prototype",
        range_max=0.1,
        color='green'):

    (subclass_to_clusters,
     class_to_clusters,
     cluster_set) = get_class_lookup(annotation_path)

    subclass_list = list(subclass_to_clusters.keys())
    class_list = list(class_to_clusters.keys())
    cluster_list = list(cluster_set)
    for l in (subclass_list, class_list, cluster_list):
        l.sort()



    valid_celltypes = find_valid_celltypes(
                            bucket=bucket,
                            class_list=class_list,
                            subclass_list=subclass_list,
                            cluster_list=cluster_list)

    celltype_to_link = dict()
    for celltype in valid_celltypes:
        this_url = create_celltypes_url(
                        bucket=bucket,
                        celltype=celltype,
                        range_max=range_max,
                        color=color,
                        template_bucket=template_bucket,
                        segmentation_bucket=segmentation_bucket)

        celltype_to_link[celltype] = this_url

    title = "Mouse1 MFISH cell type count maps"
    div_name = "celltype_maps"
    cls_name = "celltype_name"

    write_basic_table(
        output_path=output_path,
        title=title,
        key_to_link=celltype_to_link,
        div_name=div_name,
        cls_name=cls_name)


def main():

    default_anno = '/allen/programs/celltypes/'
    default_anno += 'workgroups/rnaseqanalysis/mFISH'
    default_anno += '/michaelkunst/MERSCOPES/mouse/cluster_anno.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, default=default_anno)
    args = parser.parse_args()

    html_dir = pathlib.Path('html')
    write_celltypes_html(
        output_path=html_dir / 'mouse1_celltype_maps.html',
        annotation_path=pathlib.Path(args.annotation_path))
    print("wrote html")

if __name__ == "__main__":
    main()
