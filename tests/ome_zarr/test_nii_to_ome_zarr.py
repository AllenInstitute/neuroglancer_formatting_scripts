import pytest

import json
import numpy as np
import pathlib
import SimpleITK
from skimage.transform import resize as skimage_resize
import tempfile
import zarr

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.utils.data_utils import (
    create_root_group,
    write_array_to_group,
    write_nii_to_group)


@pytest.fixture
def temp_dir_fixture(
        tmp_path_factory):
    tmp_dir = pathlib.Path(
        tmp_path_factory.mktemp('nii_to_ome_zarr'))
    yield tmp_dir
    _clean_up(tmp_dir)

@pytest.fixture
def shape_fixture():
    return (2*2*5*7, 3*3*3*7, 5*5*5*7)

@pytest.fixture
def img_array_fixture(shape_fixture):
    rng = np.random.default_rng(556623)
    return rng.random(shape_fixture)

@pytest.fixture
def pixdim_fixture():
    return ('0.01', '0.02', '0.03')

@pytest.fixture
def img_nii_fixture(
        img_array_fixture,
        pixdim_fixture,
        temp_dir_fixture):
    sub_dir = temp_dir_fixture / 'nii'
    sub_dir.mkdir()
    tmp_path = mkstemp_clean(
            dir=sub_dir,
            suffix='.nii')

    img = SimpleITK.GetImageFromArray(
        img_array_fixture)

    img.SetMetaData('pixdim[1]', pixdim_fixture[0])
    img.SetMetaData('pixdim[2]', pixdim_fixture[1])
    img.SetMetaData('pixdim[3]', pixdim_fixture[2])

    SimpleITK.WriteImage(
        image=img,
        fileName=tmp_path)

    return tmp_path


def test_write_array_to_group(
        img_array_fixture,
        pixdim_fixture,
        temp_dir_fixture):

    sub_dir = pathlib.Path(tempfile.mkdtemp(dir=temp_dir_fixture))
    if sub_dir.exists():
        sub_dir.rmdir()

    root_group = create_root_group(output_dir=sub_dir, clobber=False)

    group_name = 'test_array'
    this_group = root_group.create_group(group_name)

    write_array_to_group(
        arr=img_array_fixture,
        group=this_group,
        x_scale=float(pixdim_fixture[2]),
        y_scale=float(pixdim_fixture[1]),
        z_scale=float(pixdim_fixture[0]),
        downscale_cutoff=10,
        default_chunk=64)

    this_zarr = sub_dir / this_group.path

    # check that base array is identical to the input
    # array
    with zarr.open(this_zarr, 'r') as test_file:
        base_arr = test_file['0'][()]
        np.testing.assert_allclose(
            base_arr,
            img_array_fixture,
            rtol=1.0e-6,
            atol=0.0)

        first_scale = skimage_resize(
             base_arr,
             (70, 63, 175))

        np.testing.assert_allclose(
            test_file['1'][()],
            first_scale,
            rtol=1.0e-6,
            atol=0.0)

    # test that scale in .zattrs is as expected
    zattrs_path = this_zarr / '.zattrs'
    zattrs = json.load(open(zattrs_path, 'rb'))

    datasets = zattrs['multiscales'][0]['datasets']
    for d in datasets:
        scale = d['coordinateTransformations'][0]['scale']
        if d['path'] == '0':
            np.testing.assert_allclose(
                scale,
                (float(pixdim_fixture[2]),
                 float(pixdim_fixture[1]),
                 float(pixdim_fixture[0])),
                rtol=1.0e-6,
                atol=0.0)
        if d['path'] == '1':
            np.testing.assert_allclose(
                scale,
                (float(pixdim_fixture[2])*2,
                 float(pixdim_fixture[1])*3,
                 float(pixdim_fixture[0])*5),
                rtol=1.0e-6,
                atol=0.0)
