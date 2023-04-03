import pytest

import h5py
import json
import numpy as np
import SimpleITK
import pathlib

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.census.utils import (
    get_mask_lookup,
    census_from_NIFTI_and_mask)

from neuroglancer_interface.census.census import (
    run_census)


@pytest.fixture
def shape_fixture():
    return (26, 57, 67)


@pytest.fixture
def temp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('run_census'))
    yield tmp_dir
    _clean_up(tmp_dir)

@pytest.fixture
def structure_mask_fixture(
        shape_fixture):

    rng = np.random.default_rng(77123)
    n_voxels = shape_fixture[0]*shape_fixture[1]*shape_fixture[2]
    results = []
    for ii in range(19):
        name = f"structure_{ii}"
        mask = np.zeros(n_voxels, dtype=int)
        chosen = rng.choice(np.arange(n_voxels),
                            rng.integers(10, n_voxels//3),
                            replace=False)
        mask[chosen] = True
        mask = mask.reshape(shape_fixture)
        this = {'structure': name, 'mask': mask}
        results.append(this)
    return results


@pytest.fixture
def structure_config_fixture(
        structure_mask_fixture,
        temp_dir_fixture):

    sub_dir = temp_dir_fixture / 'masks'
    sub_dir.mkdir()
    results = []
    for structure in structure_mask_fixture:
        path = mkstemp_clean(dir=sub_dir,
                             suffix='.nii')
        img = SimpleITK.GetImageFromArray(structure['mask'])
        SimpleITK.WriteImage(
            image=img,
            fileName=path)
        this = dict()
        this['structure'] = structure['structure']
        this['path'] = path
        results.append(this)
    return results


@pytest.fixture
def data_arr_fixture(
        shape_fixture):

    rng = np.random.default_rng(55409123)
    results = []
    for ii in range(23):
        name = f'data_{ii}'
        arr = rng.random(shape_fixture)*rng.integers(1, 5)
        this = {'tag': name, 'data': arr}
        results.append(this)
    return results


@pytest.fixture
def data_config_fixture(
        data_arr_fixture,
        temp_dir_fixture):
    sub_dir = temp_dir_fixture / 'data'
    sub_dir.mkdir()
    results = []
    for data_arr in data_arr_fixture:
        path = mkstemp_clean(
                 dir=sub_dir,
                 suffix='.nii')
        img = SimpleITK.GetImageFromArray(data_arr['data'])
        SimpleITK.WriteImage(
            image=img,
            fileName=path)
        this = {'path': path, 'tag': data_arr['tag']}
        results.append(this)
    return results


@pytest.mark.parametrize(
    "n_processors", [3,])
def test_run_census(
        temp_dir_fixture,
        data_config_fixture,
        structure_config_fixture,
        n_processors):

    h5_path = mkstemp_clean(
        dir=temp_dir_fixture,
        suffix='.h5')

    h5_path = pathlib.Path(h5_path)
    assert not h5_path.exists()

    run_census(
        mask_config_list=structure_config_fixture,
        data_config_list=data_config_fixture,
        h5_path=h5_path,
        n_processors=n_processors)

    assert h5_path.is_file()

    # verify results
    mask_lookup = get_mask_lookup(
        nifti_config_list=structure_config_fixture,
        n_processors=n_processors)

    with h5py.File(h5_path, 'r') as in_file:

        structure_to_col = json.loads(
            in_file['structures'][()].decode('utf-8'))
        data_to_row = json.loads(
            in_file['datasets'][()].decode('utf-8'))

        for data_config in data_config_fixture:
            expected_census = census_from_NIFTI_and_mask(
                mask_lookup=mask_lookup,
                nifti_path=data_config['path'])

            row = data_to_row[data_config['tag']]
            for structure in expected_census:
                this = expected_census[structure]
                col = structure_to_col[structure]

                np.testing.assert_allclose(
                    this['counts'],
                    in_file['counts'][row, col],
                    rtol=1.0e-6,
                    atol=0.0)

                np.testing.assert_array_equal(
                    this['max_voxel'],
                    in_file['max_voxel'][row, col, :])

                np.testing.assert_allclose(
                    this['per_plane'],
                    in_file['per_slice'][row, col, :],
                    rtol=1.0e-6,
                    atol=0.0)
