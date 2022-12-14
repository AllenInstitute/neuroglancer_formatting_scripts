import json
import h5py
import numpy as np
import time
import pathlib
import argparse


def convert_census_to_hdf5(
        input_path,
        output_path,
        clobber=False,
        n_slices=66):

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    if not input_path.is_file():
        raise RuntimeError(f"{input_path} is not a file")

    if output_path.exists():
        if clobber:
            output_path.unlink()
        else:
            raise RuntimeError(f"{output_path} exists")

    with open(input_path, "rb") as in_file:
        input_data = json.load(in_file)

    census_data = input_data["census"]

    structure_names = []
    prefix_list = list(census_data.keys())
    prefix_list.sort()
    for prefix in prefix_list:
        name_list = list(census_data[prefix].keys())
        name_list.sort()
        for name in name_list:
            structure_names.append(f"{prefix}${name}")

    eg = census_data[structure_names[0].split('$')[0]]
    eg = eg[structure_names[0].split('$')[-1]]

    count_names = []
    name_list = list(eg['genes'].keys())
    name_list.sort()
    for name in name_list:
        count_names.append(f"genes${name}")
    prefix_list = list(eg["celltypes"].keys())
    prefix_list.sort()
    for prefix in prefix_list:
        name_list = list(eg["celltypes"][prefix].keys())
        name_idx = [int(n.split()[0]) for n in name_list]
        name_list = np.array(name_list)
        name_idx = np.array(name_idx)
        sorted_dex = np.argsort(name_idx)
        name_list = name_list[sorted_dex]
        for name in name_list:
            count_names.append(f"celltypes${prefix}${name}")

    structure_to_idx = {n:ii for ii, n in enumerate(structure_names)}
    count_to_idx = {n:ii for ii, n in enumerate(count_names)}
    print(f"{len(structure_names)} structures")
    print(f"{len(count_names)} counts")

    n_structures = len(structure_names)
    n_counts = len(count_names)
    per_slice = np.zeros(n_slices, dtype=float)


    with h5py.File(output_path, "w") as out_file:
        out_file.create_dataset(
                "counts",
                shape=(n_counts, n_structures),
                dtype=float,
                chunks=(512, 512))

        out_file.create_dataset(
                "max_voxel",
                shape=(n_counts, n_structures, 3),
                dtype=int,
                chunks=(512, 512, 3))

        chunk_size=(min(n_counts, n_structures),
                    512//n_slices,
                    n_slices)

        out_file.create_dataset(
                "per_slice",
                shape=(n_counts, n_structures, n_slices),
                chunks=chunk_size,
                compression="gzip")

        t0 = time.time()
        s_ct = 0
        t_ct = 0
        for structure_name in structure_names:
            structure_idx = structure_to_idx[structure_name]
            s_key_list = structure_name.split('$')
            this_census = census_data
            for k in s_key_list:
                this_census = this_census[k]
            s_ct += 1
            for count_name in count_names:
                count_idx = count_to_idx[count_name]
                c_key_list = count_name.split('$')
                this_data = this_census
                for k in c_key_list:
                    this_data = this_data[k]
                t_ct += 1
                out_file["counts"][count_idx, structure_idx] = this_data["counts"]
                out_file["max_voxel"][count_idx,
                                      structure_idx,
                                      :] = np.array(this_data["max_voxel"])
                
                per_slice[:] = 0.0
                for idx in this_data["per_slice"]:
                    per_slice[int(idx)] = this_data["per_slice"][idx]
                
                out_file["per_slice"][count_idx,
                                      structure_idx,
                                      :] = per_slice

                if t_ct % 1000 == 0:
                    duration = time.time()-t0
                    per = duration/t_ct
                    pred = per*(n_structures*n_counts)
                    remain = pred-duration
                    print(f"{t_ct} in {duration:2e} -- "
                          f"{remain:.2e} of {pred:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    convert_census_to_hdf5(
        input_path=args.input_path,
        output_path=args.output_path)


if __name__ == "__main__":
    main()
