import argparse
import json
import numbers

from neuroglancer_interface.utils.url_utils import (
    json_to_url,
    get_template_layer,
    get_heatmap_image_layer,
    get_base_url)


from neuroglancer_interface.utils.html_utils import (
    write_basic_table)


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
            range_max=0.7)

    template_layer = get_template_layer(
            template_bucket=bucket_name,
            template_name="avg_template",
            range_max=40000.0,
            public_name="avg_template")

    url = _get_flatmap_url(
            image_layer=image_layer,
            template_layer=template_layer)
    return url


def write_flatmap_html(
        output_path):

    dataset_list = ["e176898557_lefthemisphere",
                    "e112672268_lefthemisphere",
                    "e176898557_righthemisphere",
                    "e112672268_righthemisphere",
                    "e182182936_lefthemisphere",
                    "e112745787_lefthemisphere",
                    "e182182936_righthemisphere",
                    "e112745787_righthemisphere",
                    "e603468246_lefthemisphere",
                    "e114430043_lefthemisphere",
                    "e603468246_righthemisphere",
                    "e114430043_righthemisphere"]

    key_to_link = dict()
    key_to_other_cols = dict()
    for dataset in dataset_list:
        url = get_flatmap_url(dataset_name=dataset)
        key_to_link[dataset] = url
        key_to_other_cols[dataset] = {'names': ['experiment'],
                                      'values': [dataset]}

    write_basic_table(
        output_path=output_path,
        title="Flatmap prototype",
        key_to_link=key_to_link,
        div_name="flatmaps",
        key_to_other_cols=key_to_other_cols,
        search_by=['experiment'])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path',
                        type=str,
                        default=None)
    args = parser.parse_args()
    write_flatmap_html(output_path=args.output_path)


if __name__ == "__main__":
    main()
