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

    chosen_structures = rng.choice(structure_names, 20, replace=False)
    for structure in chosen_structures:
        structure_idx = structure_lookup[f'structures/{structure}']
        gene_names = list(json_census['structures'][structure]['genes'].keys())
        chosen_genes = rng.choice(gene_names, 5, replace=False)
        for gene in chosen_genes:
            gene_idx = gene_lookup[gene]
            h5_count = h5_census['counts'][gene_idx, structure_idx]
            json_count = json_census['structures'][structure]['genes'][gene]['counts']
            print(h5_count, json_count)
            np.testing.assert_allclose(h5_count, json_count)


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
