import argparse
import json

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

def get_base_url():
    return "https://neuroglancer-demo.appspot.com/#!"

def get_segmentation(
        segmentation_bucket,
        segmentation_name):

    return {"type": "segmentation",
            "source": f"precomputed://s3://{segmentation_bucket}",
            "tab": "source",
            "name": segmentation_name}


def get_shader_code(color, transparent=True):

    if transparent:
        default = 'emitTransparent()'
    else:
        default = 'emitRGB(vec3(0, 0, 0))'

    code = "#uicontrol invlerp normalized\nvoid main()"
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


def get_mfish(
        mfish_bucket,
        mfish_gene,
        mfish_color):

    rgb_color = get_color_lookup()[mfish_color]
    result = dict()
    result["type"] = "image"
    result["source"] = f"zarr://s3://{mfish_bucket}/{mfish_gene}",
    result["name"] = f"{mfish_gene} ({mfish_color})"
    result["blend"] = "default"
    result["shader"] = get_shader_code(rgb_color)
    result["opacity"] = 1
    return result


def get_gene_layers(
        mfish_bucket,
        gene_list,
        color_list):

    with open("mouse1_gene_list.json", "rb") as in_file:
        legal_genes = set(json.load(in_file))

    if len(gene_list) != len(color_list):
        raise RuntimeError(
             f"{len(gene_list)} genes but "
             f"{len(color_list)} colors")

    layers = []
    for gene, color in zip(gene_list, color_list):
        if gene not in legal_genes:
            raise RuntimeError(
                f"{gene} is not a legal gene")
        layers.append(get_mfish(
                          mfish_bucket=mfish_bucket,
                          mfish_gene=gene,
                          mfish_color=color))
    return layers

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

    parser.add_argument('--genes',
                        type=str,
                        nargs='+',
                        default=None)

    parser.add_argument('--colors',
                        type=str,
                        nargs='+',
                        default=None)

    args = parser.parse_args()

    if isinstance(args.genes, str):
        genes = [args.genes]
    else:
        genes = args.genes

    if isinstance(args.colors, str):
        colors = [args.colors]
    else:
        colors = args.colors

    url = get_base_url()

    segmentation_layer = get_segmentation(
                            segmentation_bucket=args.segmentation_bucket,
                            segmentation_name="segmentation")

    gene_layers = get_gene_layers(
                    mfish_bucket=args.mfish_bucket,
                    gene_list=genes,
                    color_list=colors)

    layers = {"layers": [segmentation_layer] + gene_layers}
    layers["selectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"
    url = f"{url}{json_to_url(json.dumps(layers))}"

    print(url)


if __name__ == "__main__":
    main()
