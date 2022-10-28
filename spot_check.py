import pathlib
import zarr
import SimpleITK
import json
import numpy as np

def check_census(
    zarr_path,
    mask_path,
    census):

    data_arr = np.array(zarr.open(zarr_path,'r')[0])
    data_arr = data_arr.transpose(2, 1, 0)
    mask_arr = SimpleITK.GetArrayFromImage(
                    SimpleITK.ReadImage(mask_path)).astype(bool)


    true_ct = data_arr[mask_arr].sum()

    np.testing.assert_allclose(true_ct, census['counts'])

    mn = data_arr.min()
    data_copy = np.copy(data_arr)
    data_copy[np.logical_not(mask_arr)] = mn-1.0
    idx = np.argmax(data_copy)
    voxel = np.unravel_index(idx, data_arr.shape)
    np.testing.assert_allclose(voxel, census['max_voxel'])
    return data_arr[voxel[0], voxel[1], voxel[2]]

def main():
    with open('test_census.json', 'rb') as in_file:
        census_data = json.load(in_file)
    with open('test_mask.json', 'rb') as in_file:
        mask_lookup = json.load(in_file)

    rng = np.random.default_rng(22130)
    for parent in (census_data['genes'],
                   census_data['celltypes']['classes'],
                   census_data['celltypes']['subclasses'],
                   census_data['celltypes']['clusters']):
        k = list(parent.keys())[0]
        first_test = parent[k]
        zarr_path = first_test['zarr_path']
        struct_list = list(first_test['census'].keys())
        struct_list.sort()
        rng.shuffle(struct_list)
        for struct_id in struct_list[:10]:
            actual = first_test['census'][struct_id]
            struct_path = mask_lookup[struct_id]
            max_v = check_census(zarr_path=zarr_path,
                                 mask_path=struct_path,
                                 census=actual)
            print(f"{actual} -- {max_v:.2e}")

if __name__ == "__main__":
    main()

