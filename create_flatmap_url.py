import argparse
import json
import numbers

from neuroglancer_interface.utils.url_utils import (
    json_to_url,
    get_template_layer,
    get_heatmap_image_layer,
    get_base_url)


def _get_flatmap_url(
        image_layer,
        template_layer,
        starting_position=None):

    state = dict()
    layer_list = [image_layer, template_layer]

    state["dimensions"] = {"x": [1.0e-5, "m"],
                           "y": [1.0e-5, "m"],
                           "z": [0.0001, "m"]}
    state["crossSectionScale"] = 1.3
    state["projectionScale"] = 1024
    state["layers"] = layer_list
    state["selectedLayer"] = {"visible": True, "layer": "new layer"}
    state["layout"] = "4panel"

    if starting_position is not None:
        state["position"] = [float(x) for x in starting_position]
    url = get_base_url()
    url = f"{url}{json_to_url(json.dumps(state))}"

    return url


def get_flatmap_url(
    dataset_name):

    bucket_name = "sfd-flatmap-prototype"

    image_layer = get_heatmap_image_layer(
            bucket_name=bucket_name,
            dataset_name=dataset_name,
            public_name=dataset_name,
            color="green",
            range_max=1.0)

    template_layer = get_template_layer(
            template_bucket=bucket_name,
            template_name="avg_template",
            range_max=40000.0,
            public_name="avg_template")

    url = _get_flatmap_url(
            image_layer=image_layer,
            template_layer=template_layer)
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

    dataset = "e112672268_righthemisphere"

    url = get_flatmap_url(dataset_name=dataset)

    print(url)


if __name__ == "__main__":
    main()
