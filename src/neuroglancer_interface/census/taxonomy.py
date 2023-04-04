import h5py
import json
import numpy as np


def add_taxonomy_nodes(
        input_census_path,
        output_census_path,
        new_nodes):
    """
    Parameters
    ----------
    input_census_path:
        Path to the baseline census HDF5 file
    output_census_path:
        Path to the census HDF5 file to be written
    new_nodes:
        List of dicts. Each new dict is
        {
            "tag": name_of_node
            "children": [list of nodes from original census]}
    """
    with h5py.File(input_census_path, 'r') as in_file:
        datasets = json.loads(in_file['datasets'][()].decode('utf-8'))
        structures = in_file['structures'][()]
        counts = in_file['counts'][()]
        max_voxel = in_file['max_voxel'][()]
        per_slice = in_file['per_slice'][()]

    # check that dataset tags will still be unique
    dataset_names = set(datasets.keys())
    for node in new_nodes:
        if node['tag'] in dataset_names:
            raise RuntimeError(
                "Dataset tag "
                f"{node['tag']} occurs more than once")
        dataset_names.add(node['tag'])

    n_new_nodes = len(new_nodes)
    n_structures = counts.shape[1]
    new_counts = np.zeros((n_new_nodes, n_structures),
                          dtype=counts.dtype)
    new_max_voxel = np.zeros((n_new_nodes, n_structures, 3),
                             dtype=max_voxel.dtype)
    new_per_slice = np.zeros((n_new_nodes, n_structures, per_slice.shape[2]),
                             dtype=per_slice.dtype)

    row = counts.shape[0]
    for i_new_node, node in enumerate(new_nodes):
        datasets[node['tag']] = row
        row += 1
        child_rows = np.array([datasets[c] for c in node['children']])
        child_rows = np.sort(child_rows)
        child_counts = counts[child_rows, :]
        new_counts[i_new_node, :] = child_counts.sum(axis=0)

        child_per_slice = per_slice[child_rows, :, :]
        new_per_slice[i_new_node, :, :] = child_per_slice.sum(axis=0)

        # for each column, find the row with the largest counts value;
        # use this to arbitrarily select a max_voxel for the new nodes
        max_count_rows = np.argmax(child_counts, axis=0)
        chosen_children = child_rows[max_count_rows]
        max_voxel_row = max_voxel[chosen_children, np.arange(n_structures), :]
        new_max_voxel[i_new_node, :, :] = max_voxel_row

    n_all_nodes = counts.shape[0]+n_new_nodes

    with h5py.File(output_census_path, "w") as out_file:
        out_file.create_dataset(
            "structures",
            data=structures)

        out_file.create_dataset(
            "datasets",
            data=json.dumps(datasets).encode('utf-8'))

        out_file.create_dataset(
            "counts",
            data=np.vstack([counts, new_counts]),
            compression='gzip',
            chunks=(min(1000, n_all_nodes),
                    min(1000, n_structures)))

        out_file.create_dataset(
            "max_voxel",
            data=np.vstack([max_voxel, new_max_voxel]),
            dtype=max_voxel.dtype,
            compression='gzip',
            chunks=(min(1000, n_all_nodes),
                    min(1000, n_structures),
                    3))

        out_file.create_dataset(
            "per_slice",
            data=np.vstack([per_slice, new_per_slice]),
            compression='gzip',
            chunks=(min(1000, n_all_nodes),
                    min(1000, n_structures),
                    min(1000, per_slice.shape[2])))
