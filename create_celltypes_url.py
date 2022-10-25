import argparse
import json

from url_utils import (
    get_base_url,
    get_image_layer,
    json_to_url)


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

    url = get_base_url()

    layers_list = [get_image_layer(bucket_name=args.celltype_bucket,
                                  dataset_name=args.celltype,
                                  public_name=args.celltype.split('/')[-1],
                                  color=args.color,
                                  range_max=args.range_max)]

    layers = {"layers": layers_list}
    layers["SelectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"
    url = f"{url}{json_to_url(json.dumps(layers))}"
    print(url)


if __name__ == "__main__":
    main()
