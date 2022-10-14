import argparse
import json

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
        "%3B": ";"
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
            print(ii,pct[ii:ii+5])
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

def get_base_url():
    return "https://neuroglancer-demo.appspot.com/#!"

def get_segmentation(
        segmentation_bucket,
        segmentation_name):

    return {"type": "segmentation",
            "source": f"precomputed://s3://{segmentation_bucket}",
            "tab": "source",
            "name": segmentation_name}


def get_shader_code(color):

    code = "#uicontrol invlerp normalized\nvoid main()"
    code += " {\n  emitRGB(normalized()*"
    code += "vec3("
    code += f"{color[0]}, {color[1]}, {color[2]}"
    code += "));\n}\n"
    return code


def get_mfish(
        mfish_bucket,
        mfish_gene,
        mfish_color):

    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{mfish_bucket}/{mfish_gene}",
    result["name"] = mfish_gene
    result["blend"] = "additive"
    result["shader"] = get_shader_code(mfish_color)
    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_bucket',
                        type=str,
                        default='mouse1-atlas-prototype')
    parser.add_argument('--segmentation_name',
                        type=str,
                        default='segmentation')
    parser.add_argument('--mfish_bucket',
                        type=str,
                        default='mouse1-mfish-prototype')

    args = parser.parse_args()

    url = get_base_url()

    segmentation_layer = get_segmentation(
                            segmentation_bucket=args.segmentation_bucket,
                            segmentation_name="segmentation")

    mfish_layer = get_mfish(
                    mfish_bucket=args.mfish_bucket,
                    mfish_gene='Prox1',
                    mfish_color=(1, 0, 0))

    mfish_layer2 = get_mfish(
                    mfish_bucket=args.mfish_bucket,
                    mfish_gene='Cpne9',
                    mfish_color=(0, 1, 0))

    layers = {"layers": [segmentation_layer, mfish_layer, mfish_layer2]}
    layers["selectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"
    url = f"{url}{json_to_url(json.dumps(layers))}"

    print(url)


if __name__ == "__main__":
    main()
