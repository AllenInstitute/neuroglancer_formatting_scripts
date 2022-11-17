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
        template_bucket="mouse1-template-prototype/template",
        range_max=10.0,
        html_title="Mouse1 MFISH transcript count maps",
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.1):

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
                        segmentation_bucket=segmentation_bucket,
                        template_bucket=template_bucket,
                        x_mm=x_mm,
                        y_mm=y_mm,
                        z_mm=z_mm)
        gene_to_link[gene_name] = gene_url
        these_cols = {'names': ['gene_name'],
                      'values': [gene_name]}
        gene_to_cols[gene_name] = these_cols

    title = html_title
    div_name = "mfish_maps"

    metadata_lines = []
    metadata_lines.append(f"MFISH src: {mfish_bucket}")
    metadata_lines.append(f"template src: {template_bucket}")
    metadata_lines.append(f"segmentation src: {segmentation_bucket}")

    write_basic_table(
        output_path=output_path,
        title=title,
        key_to_link=gene_to_link,
        div_name=div_name,
        key_to_other_cols=gene_to_cols,
        search_by=['gene_name'],
        metadata_lines=metadata_lines)
