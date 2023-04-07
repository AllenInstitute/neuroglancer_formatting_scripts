import json
import numpy as np

from neuroglancer_interface.utils.url_utils import (
    get_final_url,
    get_template_layer,
    get_heatmap_image_layer,
    get_segmentation_layer)

from neuroglancer_interface.utils.html_utils import (
    write_basic_table)

def create_url(
        cell_type_dir,
        bucket_name = "neuroglancer-vis-prototype/mouse3/handoff/230406"):

    template_layer = get_template_layer(
        template_bucket=f"{bucket_name}/avg_template",
        range_max=190)

    segmentation_layer = get_segmentation_layer(
        segmentation_bucket=f"{bucket_name}/ccf_annotations",
        segmentation_name="CCF annotations")
        
    img_layer = get_heatmap_image_layer(
        bucket_name=f"{bucket_name}/cell_types",
        dataset_name=cell_type_dir.name,
        public_name=cell_type_dir.name,
        color="green",
        is_uint=False,
        range_max=0.1)

    zattr_path = cell_type_dir / ".zattrs"
    zattr_data = json.load(open(zattr_path, "r"))
    max_planes = zattr_data["max_planes"]

    datasets = zattr_data["multiscales"][0]["datasets"][0]
    coord = datasets["coordinateTransformations"]
    x_mm = None
    for c in coord:
        if c["path"] == "0":
            scale = c["scale"]
            x_mm = float(scale[0])
            y_mm = float(scale[1])
            z_mm = float(scale[2])
            break
    assert x_mm is not None

    url = get_final_url(
        image_layer_list=[img_layer],
        template_layer=template_layer,
        segmentation_layer=segmentation_layer,
        starting_position=(int(max_planes[0]), 550, 550)
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm)

    return url

def write_html(
        cell_type_dir,
        output_path):
    cell_type_dir = pathlib.Path(cell_type_dir)
    key_to_link = dict()
    cell_type_list = [n for n in cell_type_dir.iterdir()
                      if n.is_file()]
    key_order = []
    numeric = []
    key_to_other_cols = dict()
    for cell_type in cell_type_list:
        url = create_url(cell_type_dir=cell_type)
        assert cell_type.name not in key_to_link
        key_to_link[cell_type.name] = url
        key_order.append(cell_type.name)
        numeric.append(int(cell_type.name.split('.')[0]))
        zattr_path = cell_type / ".zattrs"
        zattr_data = json.load(open(zattr_path, "r"))
        ct_sum = zattr_data["sum"]
        key_to_other_cols[cell_type.name] = {"names": ["counts"], "values": [ct_sum]}

    numeric = np.array(numeric)
    key_order = np.array(key_order)
    sorted_dex = np.argsort(numeric)
    key_order = key_order[sorted_dex]

    write_basic_table(
        output_path=output_path,
        title="Mouse 3 clusters",
        key_to_link=key_to_link,
        key_order=key_orer,
        div_name="cell_types",
        key_to_other_cols=key_to_other_cols)

def main():
    write_html(
        cell_type_dir="/allen/aibs/technology/danielsf/mouse3_230406",
        output_path="/home/scott.daniel/neuroglancer_formatting_scripts/html/handoff_230406.html")


if __name__ == "__main__":
    main()
