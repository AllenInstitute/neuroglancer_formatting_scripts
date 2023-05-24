from neuroglancer_interface.utils.url_utils import (
    get_final_url)

def get_heatmap_image_layer():

    result = dict()
    result["type"] = "image"
    result["source"] = "zarr://s3://neuroglancer-vis-prototype/rgb/230524_0021"
    result["name"] = f"RGB image"
    result["blend"] = "default"
    result["shader"] = get_rgb_shader_code()
    result["opacity"] = 1.0
    result["visible"] = True
    return result


def get_rgb_shader_code():
    """
    Return shader code for a 3-channel RGB image
    """

    code = f"#uicontrol invlerp normalized_r(range=[0, 4000])\n"
    code += f"#uicontrol invlerp normalized_g(range=[0, 4000])\n"
    code += f"#uicontrol invlerp normalized_b(range=[0, 4000])\n"
    code += "void main(){\n"
    code += "float r = normalized_r(getDataValue(0));\n"
    code += "float g = normalized_g(getDataValue(1));\n"
    code += "float b = normalized_b(getDataValue(2));\n"
    code += "emitRGB(vec3(r, g, b));\n}\n"
    return code


def main():
    img_layer = get_heatmap_image_layer()
    url = get_final_url(
       [img_layer], z_mm=0.01)
    print(url)


if __name__ == "__main__":
    main()
