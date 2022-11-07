from neuroglancer_interface.utils.url_utils import (
    get_heatmap_image_layer,
    json_to_url,
    get_final_url)


def create_celltypes_url(
        bucket=None,
        celltype=None,
        range_max=0.1,
        color='green',
        template_bucket='mouse1-template-prototype',
        segmentation_bucket='mouse1-atlas-prototype',
        starting_position=None,
        desanitizer=None):

    public_name = celltype.split('/')[-1]
    if desanitizer is not None:
        public_name = desanitizer[public_name]

    image_layers = get_heatmap_image_layer(
                       bucket_name=bucket,
                       dataset_name=celltype,
                       public_name=public_name,
                       color=color,
                       range_max=range_max)

    url = get_final_url(
            image_layer_list=image_layers,
            template_bucket=template_bucket,
            segmentation_bucket=segmentation_bucket,
            starting_position=starting_position)

    return url
