import dominate
import dominate.tags
import json
import argparse

import pathlib
from create_mfish_url import create_mfish_url


def write_mfish_html(
        output_path=None,
        gene_list_path=pathlib.Path("data/mouse1_gene_list.json"),
        mfish_bucket="mouse1-mfish-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        quantile_path="data/mouse1_gene_quantiles.json"):

    with open(quantile_path, 'rb') as in_file:
        quantile_lookup = json.load(in_file)

    with open(gene_list_path, 'rb') as in_file:
        gene_list = json.load(in_file)
    gene_list.sort()

    title = "Mouse1 MFISH transcript count maps"

    doc = dominate.document(title=title)
    doc += dominate.tags.h1(title)
    doc.head += dominate.tags.link(
                    href="reconstruction_table.css",
                    rel="stylesheet",
                    type="text/css",
                    media="all")

    with dominate.tags.div(id='mfish_maps') as this_div:
        this_div += dominate.tags.input_(cls="search", placeholder="Search")
        with dominate.tags.table().add(dominate.tags.tbody(cls="list")) as this_table:

            for gene_name in gene_list:
                range_max = 10.0
                gene_url = create_mfish_url(
                                mfish_bucket=mfish_bucket,
                                genes=[gene_name,],
                                colors=['green', ],
                                range_max=[range_max, ],
                                segmentation_bucket=segmentation_bucket,
                                segmentation_name='segmentation')
                this_row = dominate.tags.tr()
                this_row += dominate.tags.td(dominate.tags.a(gene_name),
                                             cls='gene_name')
                this_row += dominate.tags.td(dominate.tags.a('link',
                                                             href=gene_url))
                this_table += this_row

        doc += this_div
    doc.body += dominate.tags.script(src="https://cdnjs.cloudflare.com/ajax/libs/list.js/1.5.0/list.min.js")

    jcode = \
    '''
    var options = {
      valueNames: [ 'gene_name', ]
    };

    var userList = new List('mfish_maps', options);
    '''

    doc.body += dominate.tags.script(jcode)

    with open(output_path, 'w') as out_file:
        out_file.write(doc.render())


def main():
    html_dir = pathlib.Path('html')
    write_mfish_html(output_path=html_dir / 'mouse1_mfish_maps.html')


if __name__ == "__main__":
    main()
