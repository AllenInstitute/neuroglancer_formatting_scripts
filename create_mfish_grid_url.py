import argparse
import json
import numbers

from url_utils import (
    url_to_json,
    get_final_url)

from create_mfish_url import (
    get_gene_layers)
    

def create_mfish_grid_url(
       mfish_bucket,
       genes,
       colors,
       range_max,
       segmentation_bucket='mouse1-segmenation-prototype',
       template_bucket='mouse1-template-prototype'):

    if len(colors) != len(genes) or len(range_max) != len(genes):
        raise RuntimeError(
            "\nlen mismatch\n"
            f"{len(genes)} genes\n"
            f"{len(colors)} colors\n"
            f"{len(range_max)} range_max\n")

    if len(genes) > 4:
        raise RuntimeError(
            f"you gave {len(genes)}; can only support 4")

    gene_layers = get_gene_layers(
                    mfish_bucket=mfish_bucket,
                    gene_list=genes,
                    color_list=colors,
                    range_max_list=range_max)

    window_list = []
    for gene in gene_layers:
        this_window = dict()
        this_window["type"] = "viewer"
        this_window["layers"] = [gene["name"],
                                 "CCF template",
                                 "CCF segmentation"]
        this_window["layout"] = "xy"
        window_list.append(this_window)

    first_row = dict()
    first_row["type"] = "row"
    first_row["children"] = window_list[:2]

    second_row = dict()
    second_row["type"] = "row"
    second_row["children"] = window_list[2:]

    layout = dict()
    layout["type"] = "column"
    layout["children"] = [first_row, second_row]

    url = get_final_url(
            image_layer_list=gene_layers,
            template_bucket=template_bucket,
            segmentation_bucket=segmentation_bucket,
            layout=layout)

    return url

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_bucket',
                        type=str,
                        default='mouse1-atlas-prototype')

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

    if isinstance(args.range_max, numbers.Number):
        range_max = [args.range_max]*len(colors)
    elif len(args.range_max) == 1:
        range_max = args.range_max*len(colors)
    else:
        range_max = args.range_max

    url = create_mfish_grid_url(
            mfish_bucket=args.mfish_bucket,
            genes=genes,
            colors=colors,
            range_max=range_max,
            segmentation_bucket=args.segmentation_bucket)

    params = url.split('#!')
    #print(url_to_json(params[1]))
    print('')

    print(url)


if __name__ == "__main__":
    main()
