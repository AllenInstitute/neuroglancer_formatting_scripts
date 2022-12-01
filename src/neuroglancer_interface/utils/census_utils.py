import numpy as np


def census_from_mask_lookup_and_arr(
        mask_lookup,
        data_arr):
    """
    Parameters
    ----------
    mask_lookup: dict
        maps some key to mask pixels (the result
        of running np.where on the mask array)

    data_arr: np.ndarray
        array that is the count data for this structure

    Returns
    -------
    Dict mapping 'counts' and 'max_voxel' to the total
    number of counts and the "brightest" voxel
    """

    result = dict()
    for mask_key in mask_lookup:
        mask_pixels = mask_lookup[mask_key]['mask']
        valid = data_arr[mask_pixels]
        total = valid.sum()
        idx = np.argmax(valid)
        voxel = [int(mask_pixels[ii][idx])
                 for ii in range(len(mask_pixels))]
        this_result = {'counts': float(total),
                       'max_voxel': voxel}
        result[mask_key] = this_result
    return result
