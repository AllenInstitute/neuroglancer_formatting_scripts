from neuroglancer_interface.modules.mfish_url import (
    create_mfish_url)

from neuroglancer_interface.utils.html_utils import (
    write_basic_table)

import dominate
import dominate.tags
import json
import pathlib


def write_mfish_html(
        output_path=None,
        gene_list_path=pathlib.Path("data/mouse1_gene_list.json"),
        mfish_bucket="mouse1-mfish-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        range_max=10.0):

    with open(gene_list_path, 'rb') as in_file:
        gene_list = json.load(in_file)

    gene_to_link = dict()
    gene_to_cols = dict()
    for gene_name in gene_list:
        gene_url = create_mfish_url(
                        mfish_bucket=mfish_bucket,
                        genes=[gene_name,],
                        colors=['green', ],
                        range_max=[range_max, ],
                        segmentation_bucket=segmentation_bucket)
        gene_to_link[gene_name] = gene_url
        these_cols = {'names': ['gene_name'],
                      'values': [gene_name]}
        gene_to_cols[gene_name] = these_cols

    title = "Mouse1 MFISH transcript count maps"
    div_name = "mfish_maps"

    write_basic_table(
        output_path=output_path,
        title=title,
        key_to_link=gene_to_link,
        div_name=div_name,
        key_to_other_cols=gene_to_cols,
        search_by=['gene_name'])
