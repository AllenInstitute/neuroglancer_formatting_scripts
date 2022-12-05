from neuroglancer_interface.utils.url_utils import (
    get_heatmap_image_layer,
    json_to_url,
    get_final_url,
    get_template_layer,
    get_segmentation_layer)


def create_celltypes_url(
        bucket=None,
        celltype=None,
        range_max=0.1,
        color='green',
        template_bucket='mouse1-template-prototype/template',
        segmentation_bucket='mouse1-atlas-prototype',
        max_count_bucket=None,
        starting_position=None,
        public_name=None,
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.1):

    if public_name is None:
        public_name = celltype.split('/')[-1]

    image_layers = get_heatmap_image_layer(
                       bucket_name=bucket,
                       dataset_name=celltype,
                       public_name=public_name,
                       color=color,
                       range_max=range_max)

    if max_count_bucket is not None:

        max_count_layer = get_heatmap_imag_layer(
            bucket_name=max_count_bucket,
            dataset_name=None,
            public_name="max gene counts",
            color=color,
            range_max=10.0,
            visible=False)

        image_layers = [image_layers, max_count_layer]

    template_layer = get_template_layer(
            template_bucket=template_bucket,
            range_max=700)

    segmentation_layer = get_segmentation_layer(
            segmentation_bucket=segmentation_bucket,
            segmentation_name="CCF segmentation")

    url = get_final_url(
            image_layer_list=image_layers,
            template_layer=template_layer,
            segmentation_layer=segmentation_layer,
            starting_position=starting_position,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm)

    return url
