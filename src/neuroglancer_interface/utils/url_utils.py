import json
import urllib.parse


def get_final_url(
        image_layer_list,
        template_layer=None,
        segmentation_layer=None,
        template_bucket='mouse1-template-prototype',
        segmentation_bucket='mouse1-segmentation-prototype',
        starting_position=None,
        x_mm=0.01,
        y_mm=0.01,
        z_mm=0.1,
        projection_scale=2048,
        cross_section_scale=2.6):
    """
    Image layers with template and segmentation layer
    """

    if not isinstance(image_layer_list, list):
        image_layer_list = [image_layer_list]

    url = get_base_url()

    layer_list = image_layer_list
    if template_layer is not None:
        layer_list.append(template_layer)
    if segmentation_layer is not None:
        layer_list.append(segmentation_layer)

    layers = dict()
    layers["dimensions"] = {"x": [float(x_mm*0.001), "m"],
                            "y": [float(y_mm*0.001), "m"],
                            "z": [float(z_mm*0.001), "m"]}
    layers["crossSectionScale"] = cross_section_scale
    layers["projectionScale"] = projection_scale
    layers["layers"] = layer_list
    layers["selectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"

    if starting_position is not None:
        layers["position"] = [float(x) for x in starting_position]
    url = f"{url}{json_to_url(json.dumps(layers))}"

    return url


def get_base_url():
    #return "https://neuromancer-seung-import.appspot.com/#!"
    return "https://neuroglancer-demo.appspot.com/#!"

def get_template_layer(
        template_bucket,
        range_max=700,
        public_name="CCF template"):

    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{template_bucket}"
    result["blend"] = "default"
    result["shader"] = get_grayscale_shader_code(
                           transparent=False,
                           range_max=range_max)
    result["opacity"] = 0.4
    result["visible"] = True
    result["name"] = public_name
    return result


def get_heatmap_image_layer(
        bucket_name,
        dataset_name,
        public_name,
        color,
        range_max,
        visible=True,
        opacity=1.0,
        is_transparent=False,
        is_uint=False):

    rgb_color = get_color_lookup()[color]
    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{bucket_name}"
    if dataset_name is not None:
        result["source"] += f"/{dataset_name}"
    result["name"] = f"{public_name} ({color})"
    result["blend"] = "default"
    result["shader"] = get_rgb_heat_map_shader_code(
                                       rgb_color,
                                       transparent=is_transparent,
                                       range_max=range_max,
                                       is_uint=is_uint)
    result["opacity"] = opacity
    result["visible"] = visible
    return result


def get_ish_image_layer(
        bucket_name,
        img_name):

    layer = dict()
    layer["type"] = "image"
    layer["blend"] = "default"
    if bucket_name.startswith('http'):
        bucket_url = bucket_name
    else:
        bucket_url = f"https://{bucket_name}.s3.amazonaws.com"

    layer["source"] = f"precomputed://{bucket_url}"
    if len(img_name) > 0:
        layer["source"] += f"/{img_name}"

    layer["shader"] = get_rgb_shader_code()
    layer["name"] = img_name
    return layer


def get_rgb_shader_code():
    """
    Return shader code for a 3-channel RGB image
    """

    code = "void main(){\n"
    code += "float r = toNormalized(getDataValue(0));\n"
    code += "float g = toNormalized(getDataValue(1));\n"
    code += "float b = toNormalized(getDataValue(2));\n"
    code += "emitRGB(vec3(r, g, b));\n}\n"
    return code

def get_rgb_heat_map_shader_code(
        color,
        transparent=True,
        range_max=20.0,
        threshold=0.0,
        is_uint=False):

    if transparent:
        default = 'emitTransparent()'
    else:
        default = 'emitRGB(vec3(0, 0, 0))'

    code = f"#uicontrol invlerp normalized(range=[0, {range_max}])\n"
    code += "void main()"
    code += " {\n  "

    if is_uint:
        code += f"    if(int(getDataValue(0).value)>{int(threshold):d})"

    else:
        code += f"    if(getDataValue(0)>{threshold})"

    code += "{\n"
    code += "        emitRGB(normalized()*"
    code += "vec3("
    code += f"{color[0]}, {color[1]}, {color[2]}"
    code += "));\n}"
    code += "    else{\n"
    code += f"{default}"
    code += ";}\n}"
    return code


def get_grayscale_shader_code(
        transparent=True,
        range_max=20.0,
        threshold=0.0):

    if transparent:
        default = 'emitTransparent()'
    else:
        default = 'emitRGB(vec3(0, 0, 0))'

    code = f"#uicontrol invlerp normalized(range=[0,{range_max}])\n"
    code += "void main()"
    code += " {\n  "
    #code += "emitGrayscale(normalized());\n}"
    code += f"    if(getDataValue(0)>{threshold})"
    code += "{\n"
    code += "        emitGrayscale(normalized())"
    code += ";\n}"
    code += "    else{\n"
    code += f"{default}"
    code += ";}\n}"
    return code


def get_segmentation_layer(
        segmentation_bucket,
        segmentation_name):

    return {"type": "segmentation",
            "source": f"precomputed://s3://{segmentation_bucket}",
            "tab": "source",
            "name": segmentation_name,
            "selectedAlpha": 0.25,
            "visible": False}


def get_color_lookup():

    return {"red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "orange": (252/255, 186/255, 3/255),
            "cyan": (3/255, 252/255, 227/255),
            "magenta": (252/255, 3/255, 252/255)}


def url_to_json(url):
    return urllib.parse.unquote(url)

def json_to_url(json_data):
    return urllib.parse.quote(json_data)

