import dominate
import dominate.tags
import json
import argparse

import pathlib
from create_mfish_url import create_mfish_url

from html_utils import write_basic_table


def write_mfish_html(
        output_path=None,
        gene_list_path=pathlib.Path("data/mouse1_gene_list.json"),
        mfish_bucket="mouse1-mfish-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        quantile_path="data/mouse1_gene_quantiles.json",
        range_max=10.0):

    with open(quantile_path, 'rb') as in_file:
        quantile_lookup = json.load(in_file)

    with open(gene_list_path, 'rb') as in_file:
        gene_list = json.load(in_file)

    gene_to_link = dict()
    for gene_name in gene_list:
        gene_url = create_mfish_url(
                        mfish_bucket=mfish_bucket,
                        genes=[gene_name,],
                        colors=['green', ],
                        range_max=[range_max, ],
                        segmentation_bucket=segmentation_bucket)
        gene_to_link[gene_name] = gene_url

    title = "Mouse1 MFISH transcript count maps"
    div_name = "mfish_maps"
    cls_name = "gene_name"

    write_basic_table(
        output_path=output_path,
        title=title,
        key_to_link=gene_to_link,
        div_name=div_name,
        cls_name=cls_name)


def main():
    html_dir = pathlib.Path('html')
    write_mfish_html(output_path=html_dir / 'mouse1_mfish_maps.html')


if __name__ == "__main__":
    main()
