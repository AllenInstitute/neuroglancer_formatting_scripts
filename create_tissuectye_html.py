from neuroglancer_interface.utils.url_utils import (
    url_to_json,
    get_final_url,
    get_heatmap_image_layer,
    get_template_layer,
    get_segmentation_layer)


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


def main():

    url = create_tissuecyte_url(
        tissuecyte_s3="tissuecyte-visualizations/data/221205/tissuecyte/1122763685",
        segmentation_s3="tissuecyte-visualizations/data/221205/ccf_annotations",
        template_s3="tissuecyte-visualizations/data/221205/avg_template",
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.01,
        starting_position=None,
        red_max=5000,
        green_max=17000)

    print(url)


if __name__ == "__main__":
    main()
