import numpy as np
from neuroglancer_interface.utils.census_utils import (
    census_from_mask_lookup_and_arr)


class CellTypeMetadataCollector(object):

    def __init__(
            self,
            structure_set_masks=None,
            structure_masks=None):
        self._metadata = None
        self.masks = {
            "structure_sets": structure_set_masks,
            "structures": structure_masks}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def collect_metadata(
            self,
            data_array,
            metadata_key):
        plane_sums = np.sum(data_array, axis=(0, 1))
        total_cts = plane_sums.sum()
        max_plane = np.argmax(plane_sums)

        this = {'total_cts': float(total_cts),
                'max_plane': int(max_plane)}
        this_census = dict()
        for mask_key in self.masks:
            sub_census = census_from_mask_lookup_and_arr(
                            mask_lookup=self.masks[mask_key],
                            data_arr=data_array)
            this_census[mask_key] = sub_census
        this['census'] = this_census
        self.metadata[metadata_key] = this
