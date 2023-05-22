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
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.01,
        starting_position=None,
        red_max=17000,
        green_max=17000,
        vector_name=None,
        image_series_id=None):

    # tissuecyte_s3 = 'neuroglancer-vis-prototype/tissuecyte/230519_final_test'
    template_s3 = 'tissuecyte-visualizations/data/230105/avg_template'
    segmentation_s3 = 'tissuecyte-visualizations/data/230105/ccf_annotations'

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
            public_name=f"{image_series_id}--{vector_name}--",
            color='red',
            range_max=red_max,
            visible=False,
            opacity=0.4,
            is_transparent=True,
            is_uint=False)

    green_layer = get_heatmap_image_layer(
            bucket_name=f"{tissuecyte_s3}/green",
            dataset_name=None,
            public_name=f"{image_series_id}--{vector_name}--",
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
            z_mm=z_mm,
            projection_scale=3072,
            cross_section_scale=3.9)

    return url


def _get_tissuecyte_metadata(
        bucket_name,
        data_prefix,
        filename='image_series_metadata.json'):

    s3_config = botocore_Config(signature_version=UNSIGNED)
    s3_client = boto3.client('s3', config=s3_config)
    metadata_key = f"data/{data_prefix}/{filename}"
    metadata_response = s3_client.get_object(
                            Bucket=bucket_name,
                            Key=metadata_key)
    metadata_stream = io.BytesIO()
    for chunk in metadata_response['Body'].iter_chunks():
        metadata_stream.write(chunk)
    metadata_stream.seek(0)
    metadata = json.load(metadata_stream)
    return metadata


def main():
    bucket_name = 'tissuecyte-visualizations'
    metadata = _get_tissuecyte_metadata(
        bucket_name=bucket_name,
        data_prefix='230518')
    skip_mice = set([661073, 669026, 658554])

    vector_lookup = dict()
    with open('tissuecyte_Neuroglancer_round2.csv', 'r') as in_file:
        header = in_file.readline()
        for line in in_file:
            line = line.strip().split(',')
            vector_lookup[int(line[0])] = line[1]

    results = []
    for element in metadata:
        if element['mouse_id'] in skip_mice:
            continue
        red_max = element['red_upper']
        green_max = element['green_upper']
        s3_tag = f"{bucket_name}/data/230518/{element['image_series_id']}"
        url = create_tissuecyte_url(
            tissuecyte_s3=s3_tag,
            red_max=red_max,
            green_max=green_max,
            vector_name=vector_lookup[element['mouse_id']],
            image_series_id=element['image_series_id'])
        results.append(
            {'mouse_id': element['mouse_id'],
             'image_series_id': element['image_series_id'],
             'url': url})


    with open('archive/index.html', 'r') as in_file:
        html_lines = in_file.readlines()
    cutoff_idx = 0
    for idx in range(len(html_lines)):
        if '</tbody>' in html_lines[idx]:
            cutoff_idx = idx
            break

    new_lines = []
    for idx in range(cutoff_idx):
        new_lines.append(html_lines[idx])
    for element in results:
        new_lines.append('        <tr>\n')
        new_lines.append(f'            <td class="image_series_id">'
                        f'<a>{element["image_series_id"]}</a></td>\n')
        new_lines.append(f'            <td class="labtracks ID">'
                        f'<a>{element["mouse_id"]}</a></td>\n')

        vector = vector_lookup[element['mouse_id']]
        new_lines.append(f'            <td class="vector ID">'
                        f'<a>{vector}</a></td>\n')
        url = element["url"]
        new_lines.append(f'            <td><a href="{url}">link</a></td>\n')
        new_lines.append(f'        </tr>\n')

    for idx in range(cutoff_idx, len(html_lines), 1):
        new_lines.append(html_lines[idx])
    with open('new_index.html', 'w') as out_file:
        for line in new_lines:
            out_file.write(line)

if __name__ == "__main__":
    main()
