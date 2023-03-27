import pytest

import numpy as np
import os
import pathlib
import SimpleITK
import tempfile

from neuroglancer_interface.utils.utils import (
    _clean_up)


def safe_tmp_path(dir, suffix, prefix=None):
    result = tempfile.mkstemp(
        dir=dir, suffix=suffix, prefix=prefix)
    os.close(result[0])
    return pathlib.Path(result[1])


# dimensions of test 3D volumes assuming
# all datasets are aligned
@pytest.fixture(scope='session')
def n0():
    return 37

@pytest.fixture(scope='session')
def n1():
    return 41

@pytest.fixture(scope='session')
def n2():
    return 53


@pytest.fixture(scope='session')
def pixdim_fixture():
     # in numpy array order
     return [0.03, 0.04, 0.05]

@pytest.fixture(scope='session')
def tmp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('developing_mouse'))
    tmp_dir.mkdir()
    yield tmp_dir
    _clean_up(tmp_dir)


@pytest.fixture(scope='session')
def template_arr_fixture(
        n0,
        n1,
        n2):
    """
    Returns template volume image as a numpy array
    """
    rng = np.random.default_rng(223141)
    data = rng.random(10, 225, size=(n0, n1, n2))
    return data


@pytest.fixture(scope='session')
def template_path_fixture(
        template_arr_fixture,
        pixdim_fixture,
        tmp_dir_fixture):
    """
    yields path to template image as NIFTI file
    """
    template_path = save_tmp_path(
        dir=tmp_dir_fixture,
        suffix='.nii.gz',
        prefix='template_image_')
    img = SimpleITK.GetImageFromArray(template_arr_fixture)
    img.SetMetaData('pixdim[1]', str(pixdim_fixture[2]))
    img.SetMetaData('pixdim[2]', str(pixdim_fixture[1]))
    img.SetMetaData('pixdim[3]', str(pixdim_fixture[0]))
    SimpleITK.WriteImage(
        image=img,
        fileName=template_path,
        compression='gzip',
        compressionLevle=4)

    yield template_path
