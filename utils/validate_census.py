import SimpleITK
import json
import numpy as np

def check_census(heatmap_path, census_data, mask_lookup, rng):
    n_validated = 0
    is_bizarre = False
    heatmap_arr = SimpleITK.GetArrayFromImage(
                    SimpleITK.ReadImage(heatmap_path))

 
    sub_keys = list(census_data.keys())
    sub_keys.sort()
    rng.shuffle(sub_keys)
    non_zero = False
    zero = False
    for s in sub_keys:
        if zero and non_zero:
            break

        ct = census_data[s]["counts"]
        test_voxel = census_data[s]['max_voxel']

        if ct < 1.0e-10 and zero:
            continue

        if ct > 1.0e-3 and non_zero:
            continue

        n_validated += 1
        mask_path = mask_lookup[s]
        mask_arr = SimpleITK.GetArrayFromImage(
                     SimpleITK.ReadImage(mask_path))
        mask_arr = (mask_arr==1)
        expected_ct = np.sum(heatmap_arr[mask_arr])

        mask_where = np.where(mask_arr==1)
        expected_ct = float(heatmap_arr[mask_where].sum())

        np.testing.assert_allclose(expected_ct, ct, atol=0.0, rtol=0.0001)

        test_max_val = heatmap_arr[test_voxel[0], test_voxel[1], test_voxel[2]]
        this_max_val = -999.0
        if ct > 1.0e-10:
            this_heatmap = np.copy(heatmap_arr)
            this_heatmap[~mask_arr] = 0.0
            max_idx = np.argmax(this_heatmap)
            max_voxel = np.unravel_index(max_idx, heatmap_arr.shape)
            this_max_val = heatmap_arr[max_voxel[0], max_voxel[1], max_voxel[2]]

            if not np.allclose(max_voxel,
                               test_voxel):
                is_bizarre = True
                if not np.allclose(this_max_val, test_max_val):
                    raise RuntimeError("mismatch max voxel\n"
                                   f"{test_voxel} -- {this_max_val:.5e}\n"
                                   f"{max_voxel} -- {test_max_val:.5e}\n"
                                   f"{ct:.2e} {expected_ct:.2e}")
        else:
            max_voxel = None

        if ct < 1.0e-10:
            zero = True

        elif ct>1.0e-3:
            non_zero = True

        if is_bizarre:
            print(f"validated {ct} {test_voxel} {test_max_val:.5e} -- {expected_ct} {max_voxel} {this_max_val:.5e}"
                   f" {zero} {non_zero}")
    return n_validated

def validate_dir(dir_name, rng=None):
    with open(f"/allen/aibs/technology/danielsf/mouse3_test/{dir_name}/metadata.json", "rb") as in_file:
            json_data = json.load(in_file)

    for k in json_data:
        if k == 'masks':
            continue
        check_census(heatmap_path=k,
                     census_data=json_data[k]['census'],
                     masks=json_data['masks'],
                     rng=rng)

def validate_census(census_path, rng=None):
    n_validated = 0
    last_print = 0
    with open(census_path, "rb") as in_file:
        full_census = json.load(in_file)

    for structure_key in ("structures", "structure_sets"):
        mask_lookup = full_census[f'{structure_key[:-1]}_masks']
        census = full_census['census'][structure_key]
        zarr_path_lookup = full_census['zarr_paths']

        eg_key = list(census.keys())[0]
        eg_census = census[eg_key]       

        for hier in ("cluster", "Level_1", "Level_2"):
            cell_type_list = list(eg_census['celltypes'][hier].keys())
            cell_type_list.sort()
            for cell_type in cell_type_list:
                heatmap_path = zarr_path_lookup[f"{hier}/{cell_type}"]
                census_data = dict()
                for struct in census:
                    census_data[struct] = census[struct]['celltypes'][hier][cell_type] 
                n_validated += check_census(
                    heatmap_path=heatmap_path,
                    census_data=census_data,
                    mask_lookup=mask_lookup,
                    rng=rng)
                if n_validated >last_print+ 100:
                    print(f"validated {n_validated}")
                    last_print = n_validated
            print(f"done with {hier} {n_validated}")
        print(f"starting genes {n_validated}")
        for gene_key in eg_census['genes']:
            heatmap_path = zarr_path_lookup[gene_key]
            census_data = dict()
            for struct in census:
                census_data[struct] = census[struct]['genes'][gene_key]
            n_validated += check_census(
                heatmap_path=heatmap_path,
                census_data=census_data,
                mask_lookup=mask_lookup,
                rng=rng)
            if n_validated > last_print + 100:
               print(f"validated {n_validated}")
               last_print=n_validated

    print(f"validated {n_validated}")
    print("done")

def main():
    rng = np.random.default_rng(2231321)
    census_path = "/allen/aibs/technology/danielsf/mouse3_test/census.json"
    validate_census(census_path, rng=rng) 


if __name__ == "__main__":
    main()
