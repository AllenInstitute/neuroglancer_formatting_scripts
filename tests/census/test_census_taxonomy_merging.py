import pytest

import h5py
import json
import numpy as np
import pathlib

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.census.taxonomy import (
    add_taxonomy_nodes)


@pytest.fixture
def temp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('census_taxonomy'))
    yield tmp_dir
    _clean_up(tmp_dir)

@pytest.fixture
def n_datasets_fixture():
    return 29

@pytest.fixture
def n_structures_fixture():
    return 7

@pytest.fixture
def n_slices_fixture():
    return 17

@pytest.fixture
def counts_fixture(
        n_datasets_fixture,
        n_structures_fixture):
    rng = np.random.default_rng(88931)
    return rng.random(
            (n_datasets_fixture,
             n_structures_fixture))

@pytest.fixture
def max_voxel_fixture(
        n_datasets_fixture,
        n_structures_fixture):
    rng = np.random.default_rng(776123)
    return rng.integers(
        10, 500,
        (n_datasets_fixture,
         n_structures_fixture,
         3))

@pytest.fixture
def per_slice_fixture(
        n_datasets_fixture,
        n_structures_fixture,
        n_slices_fixture):
    rng = np.random.default_rng(34343434)
    return rng.random(
        (n_datasets_fixture,
         n_structures_fixture,
         n_slices_fixture))

@pytest.fixture
def dataset_to_row_fixture(
        n_datasets_fixture):
    result = dict()
    for ii in range(n_datasets_fixture):
        result[f'data_{ii}'] = ii
    return result

@pytest.fixture
def structure_to_col_fixture(
        n_structures_fixture):
    result = dict()
    for ii in range(n_structures_fixture):
        result[f'structure_{ii}'] = ii
    return result

@pytest.fixture
def baseline_census_fixture(
        counts_fixture,
        max_voxel_fixture,
        per_slice_fixture,
        dataset_to_row_fixture,
        structure_to_col_fixture,
        temp_dir_fixture):

    sub_dir = temp_dir_fixture / 'baseline'
    sub_dir.mkdir()
    h5_path = mkstemp_clean(
            dir=sub_dir,
            suffix='.h5')

    with h5py.File(h5_path, 'w') as out_file:
        out_file.create_dataset(
            'datasets',
            data=json.dumps(dataset_to_row_fixture).encode('utf-8'))
        out_file.create_dataset(
            'structures',
            data=json.dumps(structure_to_col_fixture).encode('utf-8'))
        out_file.create_dataset(
            'counts',
            data=counts_fixture)
        out_file.create_dataset(
            'max_voxel',
            data=max_voxel_fixture)
        out_file.create_dataset(
            'per_slice',
            data=per_slice_fixture)

    return h5_path


def test_add_taxonomy_nodes(
        baseline_census_fixture,
        counts_fixture,
        max_voxel_fixture,
        per_slice_fixture,
        dataset_to_row_fixture,
        structure_to_col_fixture,
        temp_dir_fixture):

    h5_path = mkstemp_clean(
        dir=temp_dir_fixture,
        suffix='.h5')

    new_nodes = [
        {'tag': 'new_node_0',
         'children': ['data_9', 'data_13', 'data_2']},
        {'tag': 'new_node_1',
         'children': ['data_1', 'data_22']}]

    assert not pathlib.Path(h5_path).exists()

    add_taxonomy_nodes(
        input_census_path=baseline_census_fixture,
        output_census_path=h5_path,
        new_nodes=new_nodes)

    assert pathlib.Path(h5_path).is_file()

    # use brute force calculations to varify the contents
    # of the new census file

    n_new = counts_fixture.shape[0]+len(new_nodes)

    with h5py.File(h5_path, 'r') as test_file:
        assert test_file['counts'].shape == (n_new,
                                             counts_fixture.shape[1])
        assert test_file['max_voxel'].shape == (n_new,
                                                counts_fixture.shape[1],
                                                3)
        assert test_file['per_slice'].shape == (n_new,
                                                counts_fixture.shape[1],
                                                per_slice_fixture.shape[2])

        with h5py.File(baseline_census_fixture, 'r') as baseline_file:
            assert test_file['structures'][()] == baseline_file['structures'][()]
            test_datasets = json.loads(test_file['datasets'][()].decode('utf-8'))
            for k in dataset_to_row_fixture:
                assert test_datasets[k] == dataset_to_row_fixture[k]
            assert test_datasets['new_node_0'] == counts_fixture.shape[0]
            assert test_datasets['new_node_1'] == counts_fixture.shape[0]+1

        # check counts array
        np.testing.assert_array_equal(
            test_file['counts'][:counts_fixture.shape[0], :],
            counts_fixture)

        new0 = counts_fixture[2, :] + counts_fixture[9, :] + counts_fixture[13, :]
        np.testing.assert_allclose(
            new0,
            test_file['counts'][counts_fixture.shape[0], :],
            rtol=1.0e-6,
            atol=0.0)

        new1 = counts_fixture[1, :] + counts_fixture[22, :]
        np.testing.assert_allclose(
            new1,
            test_file['counts'][counts_fixture.shape[0]+1, :],
            rtol=1.0e-6,
            atol=0.0)

        # check per_slice array
        np.testing.assert_array_equal(
            test_file['per_slice'][:counts_fixture.shape[0], :, :],
            per_slice_fixture)

        new0 = (per_slice_fixture[2, :, :]
                + per_slice_fixture[9, :, :]
                + per_slice_fixture[13, :, :])
        np.testing.assert_allclose(
            new0,
            test_file['per_slice'][counts_fixture.shape[0], :, :],
            rtol=1.0e-6,
            atol=0.0)

        new1 = (per_slice_fixture[1, :, :]
                + per_slice_fixture[22, :, :])
        np.testing.assert_allclose(
            new1,
            test_file['per_slice'][counts_fixture.shape[0]+1, :, :],
            rtol=1.0e-6,
            atol=0.0)

        # check max_voxel array
        np.testing.assert_array_equal(
            test_file['max_voxel'][:counts_fixture.shape[0], :, :],
            max_voxel_fixture)

        # check that max_voxel was assigned the value associated
        # with the maximum element in the counts array
        n_structures = counts_fixture.shape[1]
        for i_row, children in zip((counts_fixture.shape[0],
                                    counts_fixture.shape[0]+1),
                                   ([9, 13, 2], [1, 22])):
            for i_structure in range(n_structures):
                max_v = None
                max_idx = None
                for c in children:
                    if max_v is None or counts_fixture[c, i_structure] > max_v:
                        max_v = counts_fixture[c, i_structure]
                        max_idx = c
                np.testing.assert_array_equal(
                    test_file['max_voxel'][i_row, i_structure, :],
                    max_voxel_fixture[max_idx, i_structure, :])
