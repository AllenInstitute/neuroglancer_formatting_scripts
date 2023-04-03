import pytest

import numpy as np
import SimpleITK
import pathlib

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.census.utils import (
    get_mask_from_NIFTI,
    get_mask_lookup,
    census_from_NIFTI_and_mask)



@pytest.fixture
def structure_mask_fixture():
    arr = np.zeros((32, 31, 30), dtype=int)
    arr[2, 3, 4] = 1
    arr[15, 16, 17] = 1
    return arr


@pytest.fixture
def structure_nifti_fixture(
        structure_mask_fixture,
        tmp_path_factory):
    """
    Path to NIFTI file containing structure mask
    """
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('structure_mask'))
    tmp_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.nii')

    img = SimpleITK.GetImageFromArray(
        structure_mask_fixture)

    SimpleITK.WriteImage(
        image=img,
        fileName=tmp_path)

    yield tmp_path

    _clean_up(tmp_dir)


def test_get_mask_from_NIFTI(
        structure_nifti_fixture,
        structure_mask_fixture):

    actual = get_mask_from_NIFTI(
        nifti_path=structure_nifti_fixture)

    assert actual['shape'] == structure_mask_fixture.shape
    assert len(actual['pixels']) == 3
    expected = np.where(structure_mask_fixture==1)
    for ii in range(3):
        np.testing.assert_array_equal(
            expected[ii], actual['pixels'][ii])


@pytest.fixture
def shape_fixture():
    return (22, 47, 53)

@pytest.fixture
def mask_list_fixture(shape_fixture):
    """
    List of dicts:
        "structure": the name of the structure
        "mask": array that is the structure mask
    """
    rng = np.random.default_rng(553322)
    n_voxels = shape_fixture[0]*shape_fixture[1]*shape_fixture[2]
    result = []
    for ii in range(17):
        name = f"structure_{ii}"
        mask = np.zeros(n_voxels, dtype=int)
        chosen = rng.choice(len(mask), rng.integers(10, n_voxels//2))
        mask[chosen] = 1
        mask = mask.reshape(shape_fixture)
        result.append({'structure': name, 'mask': mask})
    return result


@pytest.fixture
def structure_list_nifti_fixture(
        mask_list_fixture,
        tmp_path_factory):
    """
    Returns list of configs like
        "structure": structure name
        "path": path to NIFTI file
    """
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp("list_of_nifti"))

    config_list = []
    for mask_config in mask_list_fixture:
        this_config = dict()
        this_config["structure"] = mask_config["structure"]
        nifti_path = mkstemp_clean(dir=tmp_dir, suffix='.nii')
        img = SimpleITK.GetImageFromArray(mask_config["mask"])
        SimpleITK.WriteImage(
            image=img,
            fileName=nifti_path)
        this_config["path"] = str(nifti_path)
        config_list.append(this_config)

    yield config_list

    _clean_up(tmp_dir)


@pytest.mark.parametrize(
    "n_processors", [3, 2])
def test_get_mask_lookup(
        mask_list_fixture,
        structure_list_nifti_fixture,
        n_processors):

    actual = get_mask_lookup(
        nifti_config_list=structure_list_nifti_fixture,
        n_processors=n_processors)

    assert len(actual) == len(mask_list_fixture)

    for expected_mask, expected_config in zip(mask_list_fixture,
                                              structure_list_nifti_fixture):
        expected_result = get_mask_from_NIFTI(
            nifti_path=expected_config['path'])
        this_actual = actual[expected_config['structure']]
        assert expected_config['path'] == this_actual['path']
        assert this_actual['shape'] == expected_mask['mask'].shape
        assert len(this_actual['pixels']) == 3
        for ii in range(3):
            np.testing.assert_array_equal(
                this_actual['pixels'][ii],
                expected_result['pixels'][ii])


@pytest.fixture
def data_array_fixture(
        shape_fixture):
    """
    Array of simulated data
    """
    rng = np.random.default_rng(887123)
    return rng.random(shape_fixture)


@pytest.fixture
def nifti_data_fixture(
        data_array_fixture,
        tmp_path_factory):
    """
    Path to a NIFTI file containing example data
    """
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('with_nifti_data'))
    tmp_path = mkstemp_clean(
        dir=tmp_dir,
        suffix='.nii')

    img = SimpleITK.GetImageFromArray(data_array_fixture)
    SimpleITK.WriteImage(
        image=img,
        fileName=tmp_path)

    yield pathlib.Path(tmp_path)

    _clean_up(tmp_path)


def test_census_from_nifti_and_mask(
        data_array_fixture,
        nifti_data_fixture,
        mask_list_fixture,
        structure_list_nifti_fixture):

    mask_lookup = get_mask_lookup(
        nifti_config_list=structure_list_nifti_fixture,
        n_processors=3)

    actual_census = census_from_NIFTI_and_mask(
        nifti_path=nifti_data_fixture,
        mask_lookup=mask_lookup)

    # compare actual results to brute force results
    for expected_mask, expected_config in zip(mask_list_fixture,
                                              structure_list_nifti_fixture):
        actual = actual_census[expected_config['structure']]
        mask = expected_mask['mask'].astype(bool)
        expected_counts = data_array_fixture[mask].sum()
        np.testing.assert_allclose(actual['counts'],
                                   expected_counts,
                                   rtol=1.0e-6,
                                   atol=0.0)

        max_voxel = None
        max_val = None
        pixels = mask_lookup[expected_config['structure']]['pixels']
        for (ii, jj, kk) in zip(pixels[0], pixels[1], pixels[2]):
            vv = data_array_fixture[ii, jj, kk]
            if max_val is None or vv > max_val:
                max_val = vv
                max_voxel = (ii, jj, kk)
        np.testing.assert_array_equal(max_voxel, actual['max_voxel'])

        pixels_ii = pixels[0]
        pixels_jj = pixels[1]
        pixels_kk = pixels[2]
        per_plane = np.zeros(data_array_fixture.shape[0], dtype=float)
        for unq_slice in np.unique(pixels_ii):
            valid = (pixels_ii==unq_slice)
            this = data_array_fixture[pixels_ii[valid],
                                      pixels_jj[valid],
                                      pixels_kk[valid]].sum()
            per_plane[unq_slice] = this
        np.testing.assert_allclose(
            per_plane,
            actual['per_plane'],
            rtol=1.0e-6,
            atol=0.0)
