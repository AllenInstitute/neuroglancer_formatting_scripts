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

    rng = np.random.default_rng(22130)
    census = census_data['census']
    zarr_paths = census_data['zarr_paths']
    mask_paths = census_data['mask_paths']
    for struct_name in census:
        this_census = census[struct_name]
        gene_names = list(this_census['genes'].keys())
        gene_names.sort()
        rng.shuffle(gene_names)
        print('genes')
        for gene in gene_names[:5]:
            actual = this_census['genes'][gene]
            max_v = check_census(zarr_path=zarr_paths[gene],
                                 mask_path=mask_paths[struct_name],
                                 census=actual)
            print(f"{actual} -- {max_v}")

        for child in ('classes', 'subclasses', 'clusters'):
            print(child)
            k_list = list(this_census['celltypes'][child].keys())
            k_list.sort()
            rng.shuffle(k_list)
            for k in k_list[:5]:
                actual = this_census['celltypes'][child][k]
                zarr_path = zarr_paths[f"{child}/{k}"]
                mask_path = mask_paths[struct_name]
                max_v = check_census(
                         zarr_path=zarr_path,
                         mask_path=mask_path,
                         census=actual)
                print(f"{actual} -- {max_v}")

    exit()
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

