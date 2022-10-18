import argparse
import json

from url_utils import (
    get_base_url,
    get_shader_code,
    get_segmentation,
    get_color_lookup,
    json_to_url)


def get_mfish(
        mfish_bucket,
        mfish_gene,
        mfish_color):

    rgb_color = get_color_lookup()[mfish_color]
    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{mfish_bucket}/{mfish_gene}",
    result["name"] = f"{mfish_gene} ({mfish_color})"
    result["blend"] = "default"
    result["shader"] = get_shader_code(rgb_color)
    result["opacity"] = 1
    return result


def get_gene_layers(
        mfish_bucket,
        gene_list,
        color_list):

    with open("mouse1_gene_list.json", "rb") as in_file:
        legal_genes = set(json.load(in_file))

    if len(gene_list) != len(color_list):
        raise RuntimeError(
             f"{len(gene_list)} genes but "
             f"{len(color_list)} colors")

    layers = []
    for gene, color in zip(gene_list, color_list):
        if gene not in legal_genes:
            raise RuntimeError(
                f"{gene} is not a legal gene")
        layers.append(get_mfish(
                          mfish_bucket=mfish_bucket,
                          mfish_gene=gene,
                          mfish_color=color))
    return layers

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

    url = get_base_url()

    segmentation_layer = get_segmentation(
                            segmentation_bucket=args.segmentation_bucket,
                            segmentation_name="segmentation")

    gene_layers = get_gene_layers(
                    mfish_bucket=args.mfish_bucket,
                    gene_list=genes,
                    color_list=colors)

    layers = {"layers": gene_layers + [segmentation_layer]}
    layers["selectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"
    url = f"{url}{json_to_url(json.dumps(layers))}"

    print(url)


if __name__ == "__main__":
    main()
