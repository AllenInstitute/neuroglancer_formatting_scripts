import argparse
import json

from url_utils import (
    get_image_layer,
    json_to_url,
    get_final_url)


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

    image_layers = get_image_layer(
                       bucket_name=args.celltype_bucket,
                       dataset_name=args.celltype,
                       public_name=args.celltype.split('/')[-1],
                       color=args.color,
                       range_max=args.range_max)

    url = get_final_url(
            image_layer_list=image_layers,
            template_bucket='mouse1-template-prototype',
            segmentation_bucket='mouse1-atlas-prototype')

    print(url)


if __name__ == "__main__":
    main()
