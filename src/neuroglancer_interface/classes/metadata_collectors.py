import numpy as np
import pathlib
import json
from neuroglancer_interface.utils.census_utils import (
    census_from_mask_lookup_and_arr)


class DummyLock(object):

    def __enter__(self):
        pass

    def __exit__(
            self,
            exception_type,
            exception_value,
            exception_traceback):
        pass


class MetadataCollectorABC(object):

    def collect_metadata(
            self,
            data_array,
            rotation_matrix,
            metadata_key,
            other_metadata=None):
        raise NotImplementedError("this is the base.collect_metadata")

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def set_lock(self, lock_obj):
        self._lock = lock_obj

    def add_final_metadata(self, metadata):
        raise NotImplementedError("base.add_final_metadata")

    def write_to_file(self):
        output_path = pathlib.Path(self.output_path)
        if output_path.exists():
            raise RuntimeError(f"{output_path} exists already")

        metadata = dict(self.metadata)
        metadata = self.add_final_metadata(metadata)
        with open(output_path, 'w') as out_file:
            out_file.write(json.dumps(metadata, indent=2))


class BasicMetadataCollector(MetadataCollectorABC):
    """
    This metadata collector just collects x_mm, y_mm, z_mm
    """
    def __init__(
            self,
            metadata_output_path=None):

        self._metadata=None
        self._lock = None
        self.output_path = metadata_output_path

    def add_final_metadata(self, metadata):
        return metadata

    def collect_metadata(
            self,
            data_array,
            rotation_matrix,
            metadata_key,
            other_metadata=None):

        with self._lock:
            if metadata_key in self.metadata:
                raise RuntimeError(
                    f"Trying to write {metadata_key} more than once")
            self.metadata[metadata_key] = other_metadata


class CellTypeMetadataCollector(MetadataCollectorABC):

    def __init__(
            self,
            metadata_output_path=None,
            structure_set_masks=None,
            structure_masks=None):

        self._metadata = None
        self._lock = None
        self.masks = None
        self.output_path = metadata_output_path
        if structure_set_masks is not None or structure_masks is not None:
            self.masks = {
                "structure_sets": structure_set_masks,
                "structures": structure_masks}


    def collect_metadata(
            self,
            data_array,
            rotation_matrix,
            metadata_key,
            other_metadata=None):
        plane_sums = np.sum(data_array, axis=(0, 1))
        total_cts = plane_sums.sum()
        max_plane = np.argmax(plane_sums)
        valid = (data_array > 0.0)

        this = {'total_cts': float(total_cts),
                'max_plane': int(max_plane),
                'max_val': float(data_array.max()),
                'volume_shape': [int(data_array.shape[0]),
                                 int(data_array.shape[1]),
                                 int(data_array.shape[2])]}

        if other_metadata is not None:
            this.update(other_metadata)

        this_census = dict()

        if self.masks is not None:
            for mask_key in self.masks:
                if self.masks[mask_key] is not None:
                    sub_census = census_from_mask_lookup_and_arr(
                                mask_lookup=self.masks[mask_key],
                                data_arr=data_array,
                                rotation_matrix=rotation_matrix)
                this_census[mask_key] = sub_census

        if len(this_census) > 0:
            this['census'] = this_census

        if self._lock is None:
            this_lock = DummyLock()
        else:
            this_lock = self._lock

        with this_lock:
            if metadata_key in self.metadata:
                raise RuntimeError(
                    f"Trying to write {metadata_key} more than once")
            self.metadata[metadata_key] = this

    def add_final_metadata(self, metadata):
        if hasattr(self, 'masks') and self.masks is not None:
            local_masks = dict()
            for k in self.masks:
                if self.masks[k] is not None:
                    local_masks[k] = dict()
                    for el in self.masks[k]:
                        local_masks[k][el] = {'path': self.masks[k][el]['path']}

            metadata['masks'] = local_masks
        return metadata
