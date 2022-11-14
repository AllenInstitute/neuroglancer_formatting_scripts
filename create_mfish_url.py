import argparse
import json
import numbers

from neuroglancer_interface.modules.mfish_url import (
    create_mfish_url)


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
    else:
        range_max = args.range_max

    url = create_mfish_url(
            mfish_bucket=args.mfish_bucket,
            genes=genes,
            colors=colors,
            range_max=range_max,
            segmentation_bucket=args.segmentation_bucket)

    params = url.split('#!')
    print(url_to_json(params[1]))
    print('')

    print(url)


if __name__ == "__main__":
    main()
