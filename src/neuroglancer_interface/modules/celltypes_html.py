from neuroglancer_interface.modules.celltypes_url import (
    create_celltypes_url)

from neuroglancer_interface.utils.celltypes_utils import (
    read_all_manifests)

from neuroglancer_interface.utils.html_utils import (
    write_basic_table)

from botocore import UNSIGNED
from botocore.client import Config
import boto3

import json
import time
import multiprocessing
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
        cell_types_dir=None,
        title="Mouse1 cell type count maps"):
    """
    data_dir has children that are levels within the partonomy.
    Scan those children to assess the available cell types
    """

    full_manifest = read_all_manifests(cell_types_dir)

    print("getting starting position lookup")
    starting_position_lookup = get_starting_positions(
            cell_types_dir)
    print("got starting position lookup")

    celltype_to_link = dict()
    celltype_to_cols = dict()
    key_order = list()
    numerical_list = list()

    for celltype in full_manifest:

        s3_celltype = f"{celltype['hierarchy']}/{celltype['machine_readable']}"
        this_metadata = starting_position_lookup[s3_celltype]

        starting_position = this_metadata['starting_position']
        total_cts = this_metadata['total_cts']

        this_url = create_celltypes_url(
                        bucket=cell_types_bucket,
                        celltype=s3_celltype,
                        range_max=range_max,
                        color=color,
                        template_bucket=template_bucket,
                        segmentation_bucket=segmentation_bucket,
                        public_name=celltype['human_readable'],
                        starting_position=starting_position,
                        x_mm=this_metadata["x_mm"],
                        y_mm=this_metadata["y_mm"],
                        z_mm=this_metadata["z_mm"])

        hierarchy = celltype['hierarchy']
        celltype_name = celltype['human_readable']
        celltype_key = celltype['unique']
        key_order.append(celltype_key)
        num_idx = int(celltype['machine_readable'].split('_')[0])
        numerical_list.append(num_idx)

        celltype_to_link[celltype_key] = this_url
        these_cols = {'names': ['celltype_name', 'hierarchy'],
                      'values': [celltype_name, hierarchy]}

        if cell_types_dir is not None:
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


def get_starting_positions(
        cell_types_dir):

    lookup = dict()
    child_dir_list = [n for n in cell_types_dir.iterdir()
                      if n.is_dir()]

    for child_dir in child_dir_list:
        metadata_path = child_dir / "metadata.json"
        with open(metadata_path, "rb") as in_file:
            metadata = json.load(in_file)

        for element in metadata:
            if element == "masks":
                continue
            unq_key = f"{child_dir.name}/{element}"
            if unq_key in lookup:
                raise RuntimeError(
                    f"more than one metadata entry for {unq_key}")
            plane = metadata[element]["max_plane"]
            this = {"starting_position": [550, 550, int(plane)],
                    "total_cts": metadata[element]["total_cts"],
                    "x_mm": metadata[element]["x_mm"],
                    "y_mm": metadata[element]["y_mm"],
                    "z_mm": metadata[element]["z_mm"]}
            lookup[unq_key] = this
    return lookup
