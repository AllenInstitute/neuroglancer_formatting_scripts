import re
import numpy as np
import SimpleITK

def get_labels(annotation_path):
    name_lookup = {}
    name_set = set()
    name_pattern = re.compile('".*"')
    with open(annotation_path, 'r') as in_file:
        for line in in_file:
            idx = int(line.split()[0])
            name = name_pattern.findall(line)[0]
            name = name.replace('"','')
            name = name.split(' - ')[0]
            if name in name_set:
                raise RuntimeError(
                    f"{name} repeated")
            if idx in name_lookup:
                raise RuntimeError(
                    f"idx {idx} repeated")
            name_set.add(name)
            name_lookup[idx] = name
    return name_lookup


def get_dummy_labels(segmentation_path_list):
    """
    construct dict from int to str(int)
    """
    result = dict()
    for pth in segmentation_path_list:
        vals = np.unique(
                    SimpleITK.GetArrayFromImage(
                        SimpleITK.ReadImage(pth)))
        for v in vals:
            result[int(v)] = f"{str(int(v))}"

    return result

def format_labels(labels):
    """
    convert an idx -> label lookup into a dict conforming to the metadata
    schema expected by neuroglancer for a segmentation layer
    """
    output = dict()
    output["@type"] = "neuroglancer_segment_properties"
    inline = dict()
    inline["ids"] = []
    properties = dict()
    properties["id"] = "label"
    properties["type"] = "label"
    properties["values"] = []

    k_list = list(labels.keys())
    k_list.sort()
    for k in k_list:
        value = labels[k]
        properties["values"].append(value)
        inline["ids"].append(str(k))

    inline["properties"] = [properties]
    output["inline"] = inline
    return output
