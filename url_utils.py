
def get_base_url():
    return "https://neuroglancer-demo.appspot.com/#!"

def get_shader_code(color, transparent=True, range_max=20.0):

    if transparent:
        default = 'emitTransparent()'
    else:
        default = 'emitRGB(vec3(0, 0, 0))'

    code = f"#uicontrol invlerp normalized(range=[0, {range_max}])\n"
    code += "void main()"
    code += " {\n  "
    code += "    if(getDataValue(0)>0.0){\n"
    code += "        emitRGB(normalized()*"
    code += "vec3("
    code += f"{color[0]}, {color[1]}, {color[2]}"
    code += "));\n}"
    code += "    else{\n"
    code += f"{default}"
    code += ";}\n}"
    return code


def get_segmentation(
        segmentation_bucket,
        segmentation_name):

    return {"type": "segmentation",
            "source": f"precomputed://s3://{segmentation_bucket}",
            "tab": "source",
            "name": segmentation_name,
            "selectedAlpha": 0.25}


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
        "%3C": "<"
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

