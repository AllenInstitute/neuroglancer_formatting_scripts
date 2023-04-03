import pytest

import numpy as np
import SimpleITK
import pathlib

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.census.utils import (
    get_mask_from_NIFTI)



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
