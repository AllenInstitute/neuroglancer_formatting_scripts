import numpy as np


class CellTypeMetadataCollector(object):

    def __init__(self):
        self._metadata = None

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
        self.metadata[metadata_key] = this
