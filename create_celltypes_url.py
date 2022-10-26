import argparse
import json

from url_utils import (
    get_image_layer,
    json_to_url,
    get_final_url)


def create_celltypes_url(
        bucket=None,
        celltype=None,
        range_max=0.1,
        color='green',
        template_bucket='mouse1-template-prototype',
        segmentation_bucket='mouse1-atlas-prototype'):

    image_layers = get_image_layer(
                       bucket_name=bucket,
                       dataset_name=celltype,
                       public_name=celltype.split('/')[-1],
                       color=color,
                       range_max=range_max)

    url = get_final_url(
            image_layer_list=image_layers,
            template_bucket=template_bucket,
            segmentation_bucket=segmentation_bucket)

    return url


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
