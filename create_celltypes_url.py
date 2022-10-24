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
    get_image_layer)


def get_type_layer(
        bucket_name,
        dataset_name,
        color,
        range_max):

    rgb_color = get_color_lookup()[color]
    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{bucket_name}/{dataset_name}"
    result["name"] = f"{dataset_name} ({color})"
    result["blend"] = "default"
    result["shader"] = get_rgb_shader_code(rgb_color,
                                       transparent=False,
                                       range_max=range_max)
    result["opacity"] = 1.0
    result["visible"] = True
    return result


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
                                  public_name=args.celltype.split('/')[0],
                                  color=args.color,
                                  range_max=args.range_max)]

    layers = {"layers": layers_list}
    layers["SelectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"
    url = f"{url}{json_to_url(json.dumps(layers))}"
    print(url)


if __name__ == "__main__":
    main()
