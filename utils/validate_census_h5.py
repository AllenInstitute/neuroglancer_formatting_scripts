import json
import h5py
import pathlib
import numpy as np

def _compare_census_files(
        json_census,
        h5_census,
        rng):

    json_census = json_census['census']

    structure_lookup = json.loads(h5_census['structures'][()].decode('utf-8'))
    gene_lookup = json.loads(h5_census['genes'][()].decode('utf-8'))
    cell_types_lookup = json.loads(h5_census['cell_types'][()].decode('utf-8'))

    structure_names = list(json_census['structures'].keys())
    structure_names.sort()

    for k in cell_types_lookup:
        print(k)

    for super_structure in ('structures', 'structure_sets'):
        structure_names = list(json_census[super_structure].keys())
        structure_names.sort()

        if len(structure_names) > 20:
            chosen_structures = rng.choice(structure_names, 20, replace=False)
        else:
            chosen_structures = structure_names

        for structure in chosen_structures:
            structure_idx = structure_lookup[f'{super_structure}/{structure}']
            for obj_key, obj_lookup in zip(('genes', 'celltypes'),
                                             (gene_lookup, cell_types_lookup)):

                obj_names = list(
                    obj_lookup.keys())

                if len(obj_names) > 5:
                    chosen_obj = rng.choice(obj_names, 5, replace=False)
                else:
                    chosen_obj = obj_names

                for obj in chosen_obj:
                    obj_idx = obj_lookup[obj]
                    h5_count = h5_census['counts'][obj_idx, structure_idx]
                    h5_max_voxel = h5_census['max_voxel'][obj_idx, structure_idx, :]

                    if '/' not in obj:
                        this_json_obj = json_census[super_structure][structure][obj_key][obj]
                    else:
                        super_obj = obj.split('/')[0]
                        sub_obj = obj.replace(f'{super_obj}/','')
                        this_json_obj = json_census[super_structure][structure][obj_key][super_obj][sub_obj]

                    json_count = this_json_obj['counts']
                    json_max_voxel = this_json_obj['max_voxel']
                    print(super_structure, structure, h5_count, h5_max_voxel)
                    np.testing.assert_allclose(h5_count, json_count)
                    np.testing.assert_allclose(h5_max_voxel, json_max_voxel)

                    json_slices = this_json_obj['per_slice']
                    h5_slices = h5_census['per_slice'][obj_idx, structure_idx, :]
                    json_slice_vec = np.zeros(len(h5_slices), dtype=float)
                    for idx in json_slices:
                        json_slice_vec[int(idx)] = json_slices[idx]
                    np.testing.assert_allclose(h5_slices, json_slice_vec)


def compare_census_files(
        json_path,
        h5_path,
        rng):

    json_data = json.load(open(json_path, "rb"))
    with h5py.File(h5_path, 'r') as h5_data:
        _compare_census_files(
            json_census=json_data,
            h5_census=h5_data,
            rng=rng)


def main():
    census_dir = pathlib.Path('/allen/aibs/technology/danielsf/mouse3_census_221213')
    assert census_dir.is_dir()

    json_path = census_dir / 'census.json'
    h5_path = census_dir / 'census_h5.h5'
    rng = np.random.default_rng(2231)
    compare_census_files(
        json_path=json_path,
        h5_path=h5_path,
        rng=rng)


if __name__ == "__main__":
    main()
