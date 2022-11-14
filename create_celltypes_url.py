import argparse
import json

from neuroglancer_interface.modules.celltypes_url import (
    create_celltypes_url)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--celltype_bucket',
                        type=str,
                        default='mouse1-mfish-prototype')

    parser.add_argument('--celltype',
                        type=str,
                        default=None)

    parser.add_argument('--range_max',
                        type=float,
                        default=20.0)

    parser.add_argument('--color',
                        type=str,
                        default=None)

    args = parser.parse_args()

    url = create_celltypes_url(
               bucket=args.celltype_bucket,
               celltype=args.celltype,
               range_max=args.range_max,
               color=args.color)

    print(url)

if __name__ == "__main__":
    main()
