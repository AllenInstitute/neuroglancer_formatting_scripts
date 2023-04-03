import pytest

import numpy as np
import SimpleITK
import pathlib

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.census.utils import (
    get_mask_from_NIFTI,
    get_mask_lookup)



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
def mask_list_fixture():
    """
    List of dicts:
        "structure": the name of the structure
        "mask": array that is the structure mask
    """
    rng = np.random.default_rng()
    result = []
    for ii in range(17):
        name = f"structure_{ii}"
        shape = tuple(rng.integers(10, 30, size=3))
        mask = np.zeros(shape[0]*shape[1]*shape[2], dtype=int)
        chosen = rng.choice(len(mask), rng.integers(10, 50))
        mask[chosen] = 1
        mask = mask.reshape(shape)
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
