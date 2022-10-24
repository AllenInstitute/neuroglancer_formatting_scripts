import argparse
import json

from url_utils import (
    get_base_url,
    get_rgb_shader_code,
    get_grayscale_shader_code,
    get_segmentation,
    get_color_lookup,
    json_to_url,
    url_to_json,
    get_template_layer,
    get_image_layer)


def get_gene_layers(
        mfish_bucket,
        gene_list,
        color_list,
        range_max_list):

    with open("data/mouse1_gene_list.json", "rb") as in_file:
        legal_genes = set(json.load(in_file))

    if len(gene_list) != len(color_list):
        raise RuntimeError(
             f"{len(gene_list)} genes but "
             f"{len(color_list)} colors")

    layers = []
    for gene, color, range_max in zip(gene_list,
                                      color_list,
                                      range_max_list):
        if gene not in legal_genes:
            raise RuntimeError(
                f"{gene} is not a legal gene")
        layers.append(get_image_layer(
                          bucket_name=mfish_bucket,
                          dataset_name=gene,
                          public_name=gene,
                          color=color,
                          range_max=range_max))
    return layers


def create_mfish_url(
       mfish_bucket,
       genes,
       colors,
       range_max,
       segmentation_bucket,
       segmentation_name,
       template_bucket='mouse1-template-prototype'):

    if len(colors) != len(genes) or len(range_max) != len(genes):
        raise RuntimeError(
            "len mismatch")

    url = get_base_url()

    template_layer = get_template_layer(
                template_bucket=template_bucket,
                template_name="template",
                range_max=700)

    segmentation_layer = get_segmentation(
                            segmentation_bucket=segmentation_bucket,
                            segmentation_name="CCF segmentation")

    gene_layers = get_gene_layers(
                    mfish_bucket=mfish_bucket,
                    gene_list=genes,
                    color_list=colors,
                    range_max_list=range_max)

    layer_list = gene_layers + [template_layer, segmentation_layer]

    layers = {"layers": layer_list}
    layers["selectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"
    url = f"{url}{json_to_url(json.dumps(layers))}"

    return url

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_bucket',
                        type=str,
                        default='mouse1-atlas-prototype')

    parser.add_argument('--segmentation_name',
                        type=str,
                        default='segmentation')

    parser.add_argument('--mfish_bucket',
                        type=str,
                        default='mouse1-mfish-prototype')

    parser.add_argument('--genes',
                        type=str,
                        nargs='+',
                        default=None)

    parser.add_argument('--range_max',
                        type=float,
                        nargs='+',
                        default=20.0)

    parser.add_argument('--colors',
                        type=str,
                        nargs='+',
                        default=None)

    args = parser.parse_args()

    if isinstance(args.genes, str):
        genes = [args.genes]
    else:
        genes = args.genes

    if isinstance(args.colors, str):
        colors = [args.colors]
    else:
        colors = args.colors

    if isinstance(args.range_max, int):
        range_max = [args.range_max]*len(colors)
    else:
        range_max = args.range_max

    url = create_mfish_url(
            mfish_bucket=args.mfish_bucket,
            genes=genes,
            colors=colors,
            range_max=range_max,
            segmentation_bucket=args.segmentation_bucket,
            segmentation_name=args.segmentation_name)

    params = url.split('#!')
    print(url_to_json(params[1]))
    print('')

    print(url)


if __name__ == "__main__":
    main()
