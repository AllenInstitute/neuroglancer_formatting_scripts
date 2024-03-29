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
        mfish_bucket="mouse1-mfish-prototype",
        segmentation_bucket="mouse1-atlas-prototype",
        template_bucket="mouse1-template-prototype/template",
        range_max=0.9,
        html_title="Mouse1 MFISH transcript count maps",
        data_dir=None,
        projection_scale=2048,
        cross_section_scale=2.6):
    """
    range_max is a fraction of that gene's max value
    """

    metadata_path = data_dir / "mfish_heatmaps/metadata.json"
    with open(metadata_path, "rb") as in_file:
        full_metadata = json.load(in_file)
    if "masks" in full_metadata:
        full_metadata.pop("masks")

    template_metadata_path = data_dir / "avg_template/metadata.json"
    with open(template_metadata_path, "rb") as in_file:
        template_metadata = json.load(in_file)
    template_range_max = 0.9*template_metadata["null"]["max_val"]

    gene_list = list(full_metadata.keys())
    gene_list.sort()

    gene_to_link = dict()
    gene_to_cols = dict()
    for gene_name in gene_list:
        actual_range_max = full_metadata[gene_name]["max_val"]*range_max

        start_x = full_metadata[gene_name]["volume_shape"][0]//2
        start_y = full_metadata[gene_name]["volume_shape"][1]//2

        starting_position = [start_x,
                             start_y,
                             full_metadata[gene_name]["max_plane"]]
        gene_url = create_mfish_url(
                        mfish_bucket=mfish_bucket,
                        genes=[gene_name,],
                        colors=['green', ],
                        range_max=[actual_range_max, ],
                        segmentation_bucket=segmentation_bucket,
                        template_bucket=template_bucket,
                        x_mm=full_metadata[gene_name]["x_mm"],
                        y_mm=full_metadata[gene_name]["y_mm"],
                        z_mm=full_metadata[gene_name]["z_mm"],
                        starting_position=starting_position,
                        template_range_max=template_range_max,
                        projection_scale=projection_scale,
                        cross_section_scale=cross_section_scale)

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
