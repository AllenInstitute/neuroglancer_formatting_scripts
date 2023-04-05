from neuroglancer_interface.utils.url_utils import (
    get_final_url,
    get_template_layer,
    get_heatmap_image_layer,
    get_segmentation_layer)


def main():
    bucket_name = "neuroglancer-vis-prototype/mouse3/230405_1301_test"
    
    template_layer = get_template_layer(
        template_bucket=f"{bucket_name}/avg_template",
        range_max=190)

    segmentation_layer = get_segmentation_layer(
        segmentation_bucket=f"{bucket_name}/ccf_annotations",
        segmentation_name="CCF annotations")
        
    img_layer = get_heatmap_image_layer(
        bucket_name=f"{bucket_name}/cell_types",
        dataset_name="5178",
        public_name="cell_type_5178",
        color="green",
        is_uint=False,
        range_max=0.1)

    url = get_final_url(
        image_layer_list=[img_layer],
        template_layer=template_layer,
        segmentation_layer=segmentation_layer,
        x_mm=0.2,
        y_mm=0.02,
        z_mm=0.02)
    print(url)


if __name__ == "__main__":
    main()
