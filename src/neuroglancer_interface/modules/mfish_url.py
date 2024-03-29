import json

from neuroglancer_interface.utils.url_utils import (
    url_to_json,
    get_final_url,
    get_heatmap_image_layer,
    get_template_layer,
    get_segmentation_layer)


def create_mfish_url(
       mfish_bucket,
       genes,
       colors,
       range_max,
       segmentation_bucket='mouse1-segmenation-prototype',
       template_bucket='mouse1-template-prototype/template',
       x_mm=0.01,
       y_mm=0.01,
       z_mm=0.1,
       starting_position=None,
       template_range_max=700,
       projection_scale=2048,
       cross_section_scale=2.6):

    if len(colors) != len(genes) or len(range_max) != len(genes):
        raise RuntimeError(
            "len mismatch")

    gene_layers = get_gene_layers(
                    mfish_bucket=mfish_bucket,
                    gene_list=genes,
                    color_list=colors,
                    range_max_list=range_max)

    template_layer = get_template_layer(
            template_bucket=template_bucket,
            range_max=template_range_max,
            is_uint=True)

    segmentation_layer = get_segmentation_layer(
            segmentation_bucket=segmentation_bucket,
            segmentation_name="CCF segmentation")

    url = get_final_url(
            image_layer_list=gene_layers,
            template_layer=template_layer,
            segmentation_layer=segmentation_layer,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            starting_position=starting_position,
            projection_scale=projection_scale,
            cross_section_scale=cross_section_scale)

    return url

def get_gene_layers(
        mfish_bucket,
        gene_list,
        color_list,
        range_max_list):

    #with open("data/mouse1_gene_list.json", "rb") as in_file:
    #    legal_genes = set(json.load(in_file))

    if len(gene_list) != len(color_list):
        raise RuntimeError(
             f"{len(gene_list)} genes but "
             f"{len(color_list)} colors")

    layers = []
    for gene, color, range_max in zip(gene_list,
                                      color_list,
                                      range_max_list):
        #if gene not in legal_genes:
        #    raise RuntimeError(
        #        f"{gene} is not a legal gene")
        layers.append(get_heatmap_image_layer(
                          bucket_name=mfish_bucket,
                          dataset_name=gene,
                          public_name=gene,
                          color=color,
                          range_max=range_max))
    return layers
