import pathlib
import json
import argparse

from neuroglancer_interface.utils.url_utils import (
    get_final_url,
    get_heatmap_image_layer,
    get_template_layer,
    get_segmentation_layer)

from neuroglancer_interface.utils.html_utils import (
    write_basic_table)


def create_tissuecyte_url(
        tissuecyte_s3,
        segmentation_s3,
        template_s3,
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.01,
        starting_position=None,
        red_max=17000,
        green_max=17000):

    template_layer = get_template_layer(
        template_bucket=template_s3,
        range_max=700)

    segmentation_layer = get_segmentation_layer(
        segmentation_bucket=segmentation_s3,
        segmentation_name="CCF segmentation")

    series_id = tissuecyte_s3.split("_")[-1]

    red_layer = get_heatmap_image_layer(
            bucket_name=f"{tissuecyte_s3}/red",
            dataset_name=None,
            public_name=series_id,
            color='red',
            range_max=red_max,
            visible=False,
            opacity=0.4,
            is_transparent=True)

    green_layer = get_heatmap_image_layer(
            bucket_name=f"{tissuecyte_s3}/green",
            dataset_name=None,
            public_name=series_id,
            color='green',
            range_max=green_max,
            visible=True,
            opacity=0.4,
            is_transparent=True)

    url = get_final_url(
            image_layer_list=[green_layer, red_layer],
            template_layer=template_layer,
            segmentation_layer=segmentation_layer,
            starting_position=starting_position,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm)

    return url


def create_tissuecyte_html(
        dataset_dir,
        s3_location,
        table_title,
        output_path):

    dataset_dir = pathlib.Path(dataset_dir)

    template_s3 = f"{s3_location}/avg_template"
    segmentation_s3 = f"{s3_location}/ccf_annotations"

    metadata_path = dataset_dir / "tissuecyte/metadata.json"
    with open(metadata_path, "rb") as in_file:
        metadata = json.load(in_file)

    tissuecyte_dir = dataset_dir / "tissuecyte"
    tissuecyte_dir_list = [n for n in tissuecyte_dir.iterdir()
                           if n.is_dir()]

    tissuecyte_dir_list.sort()
    key_to_link = dict()
    key_to_other_cols = dict()
    key_order = []
    for sub_dir in tissuecyte_dir_list:
        series_id = sub_dir.name
        tissuecyte_s3 = f"{s3_location}/tissuecyte/{series_id}"
        csv_path = sub_dir / "image_series_information.csv"
        range_lookup = dict()
        with open(csv_path, "r") as in_file:
            header = in_file.readline().strip().split(',')
            data = in_file.readline().strip().split(',')
            assert len(header) == len(data)
            for h, d in zip(header, data):
                range_lookup[h] = d

        red_max = float(range_lookup['red_upper'])
        green_max = float(range_lookup['green_upper'])

        this_metadata = metadata[f"{series_id}/green"]
        x_mm = this_metadata["x_mm"]
        y_mm = this_metadata["y_mm"]
        z_mm = this_metadata["z_mm"]
        url = create_tissuecyte_url(
            tissuecyte_s3=tissuecyte_s3,
            segmentation_s3=segmentation_s3,
            template_s3=template_s3,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            red_max=red_max,
            green_max=green_max)

        key_to_link[series_id] = url
        key_order.append(series_id)
        key_to_other_cols[series_id] = {'names': ['image_series_id'],
                                        'values': [str(series_id)]}

    metadata_lines = [
        f"tissuecyte src: {tissuecyte_s3}",
        f"template src: {template_s3}",
        f"segmentation src: {segmentation_s3}"]

    write_basic_table(
        output_path=output_path,
        title=table_title,
        key_to_link=key_to_link,
        key_to_other_cols=key_to_other_cols,
        key_order=key_order,
        div_name="tissuecyte",
        search_by=['image_series_id'],
        metadata_lines=metadata_lines)

    print(f"wrote {output_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_location", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--table_title", type=str, default=None)
    args = parser.parse_args()

    create_tissuecyte_html(
        dataset_dir=args.data_dir,
        s3_location=args.s3_location,
        table_title=args.table_title,
        output_path=args.output_path)


if __name__ == "__main__":
    main()
