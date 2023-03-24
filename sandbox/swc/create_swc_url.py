from neuroglancer_interface.utils.url_utils import (
    url_to_json,
    get_final_url,
    get_heatmap_image_layer,
    get_template_layer,
    get_segmentation_layer)


def main():
    template_bucket='neuroglancer-vis-prototype/mouse3/230320_XnegZ/avg_template'
    template_range_max=600
    template_layer = get_template_layer(
            template_bucket=template_bucket,
            range_max=template_range_max,
            is_uint=True)

    swc_bucket = 'neuroglancer-vis-prototype/swc/230324b'
    segmentation_layer = get_segmentation_layer(
         segmentation_bucket=swc_bucket,
         segmentation_name='example SWC')

    x_mm=0.05
    y_mm=0.05
    z_mm=0.05

    url = get_final_url(
        image_layer_list=[],
        template_layer=template_layer,
        segmentation_layer=segmentation_layer,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        starting_position=None,
        projection_scale=256,
        cross_section_scale=0.7)

    print(url)
    print('')
    print(len(url))


if __name__ == "__main__":
    main()
