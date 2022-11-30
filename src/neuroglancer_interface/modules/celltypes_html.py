from neuroglancer_interface.modules.celltypes_url import (
    create_celltypes_url)

from neuroglancer_interface.utils.celltypes_utils import (
    read_all_manifests)

from neuroglancer_interface.utils.html_utils import (
    write_basic_table)

from botocore import UNSIGNED
from botocore.client import Config
import boto3

import zarr
import numpy as np
import dominate
import dominate.tags

def write_celltypes_html(
        output_path=None,
        annotation_path=None,
        cell_types_bucket="mouse1-celltypes-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        template_bucket="mouse1-template-prototype",
        range_max=0.1,
        color='green',
        data_dir=None,
        title="Mouse1 cell type count maps",
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.1):
    """
    data_dir has children that are levels within the partonomy.
    Scan those children to assess the available cell types
    """

    full_manifest = read_all_manifests(data_dir)

    celltype_to_link = dict()
    celltype_to_cols = dict()
    key_order = list()
    numerical_list = list()

    for celltype in full_manifest:

        data_path = celltype['data_path']

        starting_position = None
        if data_path is not None:
            (max_plane,
             total_cts) = get_ct_data(
                            data_dir=data_path.parent,
                            celltype=data_path.name)
            starting_position=[550, 550, max_plane]

        s3_celltype = f"{celltype['hierarchy']}/{celltype['machine_readable']}"
        this_url = create_celltypes_url(
                        bucket=cell_types_bucket,
                        celltype=s3_celltype,
                        range_max=range_max,
                        color=color,
                        template_bucket=template_bucket,
                        segmentation_bucket=segmentation_bucket,
                        public_name=celltype['human_readable'],
                        starting_position=starting_position,
                        x_mm=x_mm,
                        y_mm=y_mm,
                        z_mm=z_mm)

        hierarchy = celltype['hierarchy']
        celltype_name = celltype['human_readable']
        celltype_key = celltype['unique']
        key_order.append(celltype_key)
        num_idx = int(celltype['machine_readable'].split('_')[0])
        numerical_list.append(num_idx)

        celltype_to_link[celltype_key] = this_url
        these_cols = {'names': ['celltype_name', 'hierarchy'],
                      'values': [celltype_name, hierarchy]}

        if data_dir is not None:
            these_cols['names'].append('counts (arbitrary)')
            these_cols['values'].append(f"{total_cts:.3e}")

        celltype_to_cols[celltype_key] = these_cols

    title = title
    div_name = "celltype_maps"
    cls_name = "celltype_name"

    key_order = np.array(key_order)
    numerical_list = np.array(numerical_list)
    sorted_dex = np.argsort(numerical_list)
    key_order = key_order[sorted_dex]

    metadata_lines = []
    metadata_lines.append(f"cell types src: {cell_types_bucket}")
    metadata_lines.append(f"template src: {template_bucket}")
    metadata_lines.append(f"segmentation src: {segmentation_bucket}")

    write_basic_table(
        output_path=output_path,
        title=title,
        key_to_link=celltype_to_link,
        key_order=key_order,
        div_name=div_name,
        key_to_other_cols=celltype_to_cols,
        search_by=['celltype_name',
                   'hierarchy'],
        metadata_lines=metadata_lines)


def idx_from_cluster_name(cluster_name):
    params = cluster_name.split('_')
    if len(params) == 0:
        return 0
    try:
        idx = int(params[0])
    except ValueError:
        idx = 0
    return idx

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


def get_ct_data(
        data_dir,
        celltype):

    if not (data_dir / celltype).is_dir():
        raise RuntimeError(
            f"ct data cannot parse {data_dir} {celltype}")
    arr = np.array(zarr.open(data_dir / celltype, 'r')['0'])
    plane_sums = np.sum(arr, axis=(0,1))
    max_z = np.argmax(plane_sums)
    all_ct = np.sum(plane_sums)

    print(f"read ct data for {celltype}")
    return (max_z, all_ct)
