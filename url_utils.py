import json

def get_final_url(
        image_layer_list,
        template_bucket='mouse1-template-prototype',
        segmentation_bucket='mouse1-segmentation-prototype',
        starting_position=None):
    """
    Image layers with template and segmentation layer
    """

    if not isinstance(image_layer_list, list):
        image_layer_list = [image_layer_list]

    url = get_base_url()

    template_layer = get_template_layer(
            template_bucket=template_bucket,
            template_name="template",
            range_max=700)

    segmentation_layer = get_segmentation_layer(
            segmentation_bucket=segmentation_bucket,
            segmentation_name="CCF segmentation")

    layer_list = image_layer_list + [template_layer, segmentation_layer]

    layers = dict()
    layers["dimensions"] = {"x": [1.0e-5, "m"],
                            "y": [1.0e-5, "m"],
                            "z": [0.0001, "m"]}
    layers["crossSectionScale"] = 2.6
    layers["projectionScale"] = 2048
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
        template_name='template',
        range_max=700):

    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{template_bucket}/{template_name}"
    result["blend"] = "default"
    result["shader"] = get_grayscale_shader_code(
                           transparent=False,
                           range_max=range_max)
    result["opacity"] = 0.4
    result["visible"] = True
    result["name"] = "CCF template"
    return result


def get_heatmap_image_layer(
        bucket_name,
        dataset_name,
        public_name,
        color,
        range_max):

    rgb_color = get_color_lookup()[color]
    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{bucket_name}/{dataset_name}"
    result["name"] = f"{public_name} ({color})"
    result["blend"] = "default"
    result["shader"] = get_rgb_heat_map_shader_code(
                                       rgb_color,
                                       transparent=False,
                                       range_max=range_max)
    result["opacity"] = 1.0
    result["visible"] = True
    return result


def get_ish_image_layer(
        bucket_name,
        img_name):

    layer = dict()
    layer["type"] = "image"
    layer["blend"] = "default"
    bucket_url = f"https://{bucket_name}.s3.amazonaws.com"
    layer["source"] = f"precomputed://{bucket_url}/{img_name}"
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
        threshold=0.0):

    if transparent:
        default = 'emitTransparent()'
    else:
        default = 'emitRGB(vec3(0, 0, 0))'

    code = f"#uicontrol invlerp normalized(range=[0, {range_max}])\n"
    code += "void main()"
    code += " {\n  "
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


def get_pct_to_char():
    result = {
        "%20": " ",
        "%5C": "\\",
        "%22": '"',
        "%2C": ",",
        "%7B": "{",
        "%7D": "}",
        "%5B": "[",
        "%5D": "]",
        "%28": "(",
        "%29": ")",
        "%2A": "*",
        "%3B": ";",
        "%3E": ">",
        "%3C": "<",
        "%27": "'"
        }

    return result

def get_char_to_pct():
    pct_to_char = get_pct_to_char()
    result = dict()
    for k in pct_to_char:
        result[pct_to_char[k]] = k
    return result


def url_to_json(url):
    result = ""
    ii = 0
    pct_to_char = get_pct_to_char()
    while ii < len(url):
        if url[ii] != '%':
            result += url[ii]
            ii += 1
        else:
            pct = url[ii:ii+3]
            result += pct_to_char[pct]
            ii += 3

    return result

def json_to_url(json_data):
    char_to_pct = get_char_to_pct()
    result = ""
    for c in json_data:
        if c in char_to_pct:
            result += char_to_pct[c]
        else:
            result += c
    return result

