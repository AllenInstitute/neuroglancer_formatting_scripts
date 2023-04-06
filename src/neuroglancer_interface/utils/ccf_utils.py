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


def downsample_segmentation_array(
        arr,
        downsample_by):
    """
    Downsample a CCF annotation array by dividing voxels
    into blocks of size downsample_by and assigning the
    value from the central voxel.

    Parameters
    ----------
    arr:
        Array of CCF annotation
    downsample_by:
        Tuple of ints indicating the factor by which
        to downsample each dimension.

    Returns
    -------
    Downsampled array
    """


    # check that every element in downsample_by is odd (so that
    # there is a center voxel)
    for d in downsample_by:
        if d <= 0:
            raise RuntimeError(
                f"downsample_by {downsample_by} not positive definite")

    for d in downsample_by:
        if d % 2 == 0:
            raise RuntimeError(
                f"downsample_by {downsample_by} are not all  odd")

    # check that downsample_by is an integer divisor
    # of array shape in all dimensions
    for s, d in zip(arr.shape, downsample_by):
        if s % d != 0:
            raise RuntimeError(
                f"array shape {arr.shape} is not integer divisible "
                f"by downsample factors {downsample_by}")

    new_shape = (arr.shape[0]//downsample_by[0],
                 arr.shape[1]//downsample_by[1],
                 arr.shape[2]//downsample_by[2])

    new_arr = np.zeros(new_shape, dtype=arr.dtype)

    for ix in range(new_shape[0]):
        ix0 = ix*downsample_by[0]
        ix_center = ix0 + (downsample_by[0]-1)//2
        for iy in range(new_shape[1]):
            iy0 = iy*downsample_by[1]
            iy_center = iy0 + (downsample_by[1]-1)//2
            for iz in range(new_shape[2]):
                iz0 = iz*downsample_by[2]
                iz_center = iz0 + (downsample_by[2]-1)//2
                new_arr[ix, iy, iz] = arr[ix_center,
                                          iy_center,
                                          iz_center]

    return new_arr
