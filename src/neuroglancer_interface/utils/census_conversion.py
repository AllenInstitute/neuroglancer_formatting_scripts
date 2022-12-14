import json
import h5py
import numpy as np
import time
import pathlib


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

    print(f"{len(structure_names)} structures")
    print(f"{len(count_names)} counts")

    n_structures = len(structure_names)
    n_counts = len(count_names)

    with h5py.File(output_path, "w") as out_file:

        c_row = min(512, n_counts)
        c_col = min(512, n_structures)

        out_file.create_dataset(
                "counts",
                shape=(n_counts, n_structures),
                dtype=float,
                chunks=(c_row, c_col))

        out_file.create_dataset(
                "max_voxel",
                shape=(n_counts, n_structures, 3),
                dtype=int,
                chunks=(c_row, c_col, 3))

        chunk_size=(c_row,
                    min(c_col, max(1, 512//n_slices)),
                    n_slices)

        out_file.create_dataset(
                "per_slice",
                shape=(n_counts, n_structures, n_slices),
                chunks=chunk_size,
                compression="gzip")

    lookups = _write_data_in_chunks(
                structure_names=structure_names,
                count_names=count_names,
                output_path=output_path,
                census_data=census_data,
                n_slices=n_slices)

    _write_keys(output_path, lookups)


def _write_data_in_chunks(
        structure_names,
        count_names,
        output_path,
        census_data,
        n_slices):

    n_structures = len(structure_names)
    n_counts = len(count_names)

    idx_to_structure = {ii:n for ii, n in enumerate(structure_names)}
    idx_to_count = {ii:n for ii, n in enumerate(count_names)}

    dump_every = max(1, 10000000//(n_counts*n_slices))
    count_chunk = np.zeros((n_counts, dump_every), dtype=float)
    voxel_chunk = np.zeros((n_counts, dump_every, 3), dtype=int)
    per_slice_chunk = np.zeros((n_counts, dump_every, n_slices), dtype=float)

    t0 = time.time()
    s_ct = 0
    t_ct = 0

    min_struct = 0
    for structure_idx in range(n_structures):
        structure_name = idx_to_structure[structure_idx]
        s_key_list = structure_name.split('$')
        this_census = census_data
        for k in s_key_list:
            this_census = this_census[k]
        s_ct += 1
        for count_idx in range(n_counts):
            count_name = idx_to_count[count_idx]
            c_key_list = count_name.split('$')
            this_data = this_census
            for k in c_key_list:
                this_data = this_data[k]
            t_ct += 1
            count_chunk[count_idx,
                        structure_idx-min_struct] = this_data["counts"]

            voxel_chunk[count_idx,
                        structure_idx-min_struct,
                        :] = np.array(this_data["max_voxel"])

            for slice_idx in this_data["per_slice"]:
                per_slice_chunk[count_idx,
                                structure_idx-min_struct,
                                int(slice_idx)] = this_data["per_slice"][slice_idx]


        if structure_idx >= (min_struct+dump_every-1) or structure_idx == (n_structures-1):
            max_valid = structure_idx+1-min_struct
            with h5py.File(output_path, "a") as out_file:
                out_file["counts"][:, min_struct:structure_idx+1] = count_chunk[:, :max_valid]
                out_file["max_voxel"][:, min_struct:structure_idx+1, :] = voxel_chunk[:, :max_valid, :]
                out_file["per_slice"][:, min_struct:structure_idx+1, :] = per_slice_chunk[:, :max_valid, :]

            count_chunk[:, :] = 0.0
            voxel_chunk[:, :, :] = 0
            per_slice_chunk[:, :, :] = 0.0

            min_struct = structure_idx+1
            duration = time.time()-t0
            per = duration/t_ct
            pred = per*(n_structures*n_counts)
            remain = pred-duration
            print(f"{t_ct} in {duration:2e} -- "
                  f"{remain:.2e} of {pred:.2e} left")

    return {'idx_to_structure': idx_to_structure,
            'idx_to_count': idx_to_count}


def _write_keys(output_path, lookups):

    structure_lookup = {lookups['idx_to_structure'][ii].replace('$','/'): int(ii)
                        for ii in lookups['idx_to_structure']}

    gene_lookup = dict()
    cell_type_lookup = dict()
    for ii in lookups['idx_to_count']:
        name = lookups['idx_to_count'][ii]
        if name.startswith('genes$'):
            gene_lookup[name.replace('genes$','').replace('$','/')] = int(ii)
        elif name.startswith('celltypes$'):
            cell_type_lookup[name.replace('celltypes$','').replace('$','/')] = int(ii)
        else:
            raise RuntimeError(f"cannot parse name {name}")

    with h5py.File(output_path, 'a') as out_file:
        out_file.create_dataset(
            'structures',
            data=json.dumps(structure_lookup).encode('utf-8'))
        out_file.create_dataset(
            'genes',
            data=json.dumps(gene_lookup).encode('utf-8'))
        out_file.create_dataset(
            'cell_types',
            data=json.dumps(cell_type_lookup).encode('utf-8'))
