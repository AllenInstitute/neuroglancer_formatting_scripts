import pathlib
import json
import argparse

from botocore import UNSIGNED
from botocore.client import Config as botocore_Config
import boto3
import io

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
        range_max=700,
        is_uint=False)

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
            is_transparent=True,
            is_uint=False)

    green_layer = get_heatmap_image_layer(
            bucket_name=f"{tissuecyte_s3}/green",
            dataset_name=None,
            public_name=series_id,
            color='green',
            range_max=green_max,
            visible=True,
            opacity=0.4,
            is_transparent=True,
            is_uint=False)

    url = get_final_url(
            image_layer_list=[green_layer, red_layer],
            template_layer=template_layer,
            segmentation_layer=segmentation_layer,
            starting_position=starting_position,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm)

    return url


def _get_tissuecyte_metadata(
        bucket_name,
        data_prefix,
        filename='metadata.json'):

    s3_config = botocore_Config(signature_version=UNSIGNED)
    s3_client = boto3.client('s3', config=s3_config)
    metadata_key = f"data/{data_prefix}/tissuecyte/{filename}"
    metadata_response = s3_client.get_object(
                            Bucket=bucket_name,
                            Key=metadata_key)
    metadata_stream = io.BytesIO()
    for chunk in metadata_response['Body'].iter_chunks():
        metadata_stream.write(chunk)
    metadata_stream.seek(0)
    metadata = json.load(metadata_stream)
    return metadata


def create_tissuecyte_lookup(
        bucket_name,
        data_prefix):
    """
    Data prefix is the directory under bucket_name/data/
    """

    metadata = _get_tissuecyte_metadata(
                    bucket_name=bucket_name,
                    data_prefix=data_prefix,
                    filename='metadata.json')

    image_series_metadata = _get_tissuecyte_metadata(
                    bucket_name=bucket_name,
                    data_prefix=data_prefix,
                    filename='image_series_metadata.json')

    # convert to a lookup table
    image_series_metadata = {
        this['image_series_id']: this
        for this in image_series_metadata}


    series_id_list = set()
    for k in metadata:
        series_id = k.split('/')[0]
        series_id_list.add(int(series_id))
    series_id_list = list(series_id_list)
    series_id_list.sort()

    s3_location = f"{bucket_name}/data/{data_prefix}"
    template_s3 = f"{s3_location}/avg_template"
    segmentation_s3 = f"{s3_location}/ccf_annotations"

    key_to_link = dict()
    key_to_other_cols = dict()
    key_order = []
    for series_id in series_id_list:
        tissuecyte_s3 = f"{s3_location}/tissuecyte/{series_id}"
        this_metadata = image_series_metadata[int(series_id)]

        red_max = float(this_metadata['red_upper'])
        green_max = float(this_metadata['green_upper'])
        mouse_id = this_metadata['mouse_id']

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
        key_to_other_cols[series_id] = {'names': ['image_series_id', 'labtracks ID'],
                                        'values': [str(series_id), int(mouse_id)]}

    return {'key_to_link': key_to_link,
            'key_order': key_order,
            'key_to_other_cols': key_to_other_cols}


def create_tissuecyte_html(
        bucket_name,
        data_prefix_list,
        table_title,
        output_path):

    key_to_link = dict()
    key_to_other_cols = dict()
    key_order = []

    for data_prefix in data_prefix_list:
        result = create_tissuecyte_lookup(
                bucket_name=bucket_name,
                data_prefix=data_prefix)

        for k in result['key_to_link']:
            key_to_link[k] = result['key_to_link'][k]
        for k in result['key_order']:
            key_order.append(k)
        for k in result['key_to_other_cols']:
            key_to_other_cols[k]= result['key_to_other_cols'][k]

    key_order.sort()

    #metadata_lines = [
    #    f"tissuecyte src: {tissuecyte_s3}",
    #    f"template src: {template_s3}",
    #    f"segmentation src: {segmentation_s3}"]

    write_basic_table(
        output_path=output_path,
        title=table_title,
        key_to_link=key_to_link,
        key_to_other_cols=key_to_other_cols,
        key_order=key_order,
        div_name="tissuecyte",
        search_by=['image_series_id', 'labtracks ID'],
        metadata_lines=None)

    print(f"wrote {output_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", type=str,
                        default='tissuecyte-visualizations')
    parser.add_argument("--data_prefix", type=str, default=None, nargs='+')
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--table_title", type=str, default=None)
    args = parser.parse_args()

    if not isinstance(args.data_prefix, list):
        data_prefix = [args.data_prefix]
    else:
        data_prefix = args.data_prefix

    create_tissuecyte_html(
        bucket_name=args.bucket_name,
        data_prefix_list=data_prefix,
        table_title=args.table_title,
        output_path=args.output_path)


if __name__ == "__main__":
    main()
