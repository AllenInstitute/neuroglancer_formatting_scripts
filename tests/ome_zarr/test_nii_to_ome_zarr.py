import pytest

import json
import numpy as np
import pathlib
import SimpleITK
from skimage.transform import resize as skimage_resize
import tempfile
from unittest.mock import patch
import zarr

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.utils.rotation_utils import (
    rotate_matrix)

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
             img_array_fixture,
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


@pytest.mark.parametrize('do_transposition', [True, False])
def test_write_nii_to_group(
        img_array_fixture,
        img_nii_fixture,
        pixdim_fixture,
        temp_dir_fixture,
        do_transposition):


    sub_dir = pathlib.Path(tempfile.mkdtemp(dir=temp_dir_fixture))
    if sub_dir.exists():
        sub_dir.rmdir()

    root_group = create_root_group(output_dir=sub_dir, clobber=False)

    group_name = 'test_array'


    # because SimpleITK.WriteImage does not write out metadata
    def mock_get_metadata(self, value):
        lookup = dict()
        lookup['pixdim[1]'] = pixdim_fixture[0]
        lookup['pixdim[2]'] = pixdim_fixture[1]
        lookup['pixdim[3]'] = pixdim_fixture[2]
        return lookup[value]

    with patch('SimpleITK.Image.GetMetaData', new=mock_get_metadata):
        write_nii_to_group(
            root_group=root_group,
            group_name=group_name,
            nii_file_path=img_nii_fixture,
            downscale_cutoff=10,
            channel='red',
            do_transposition=do_transposition)

    this_zarr = sub_dir / group_name

    zattrs_path = this_zarr / '.zattrs'
    zattrs = json.load(open(zattrs_path, 'rb'))

    # check that base array is identical to the input
    # array
    with zarr.open(this_zarr, 'r') as test_file:

        # test the native resolution
        base_arr = test_file['0'][()]
        if do_transposition:
            expected_array = rotate_matrix(
                data=img_array_fixture,
                rotation_matrix = [[0, 0, -1],
                                   [0, 1, 0],
                                   [1, 0, 0]])
        else:
            expected_array = img_array_fixture

        np.testing.assert_allclose(
            base_arr,
            expected_array,
            rtol=1.0e-6,
            atol=0.0)

        # test the first downscaled array
        if do_transposition:
            first_scale = skimage_resize(
                 expected_array,
                 (175, 63, 70))
        else:
            first_scale = skimage_resize(
                 expected_array,
                 (70, 63, 175))

        np.testing.assert_allclose(
            test_file['1'][()],
            first_scale,
            rtol=1.0e-6,
            atol=0.0)

        # test the max planes
        max_idx = None
        max_val = None
        for ix in range(expected_array.shape[0]):
            v = expected_array[ix, :, :].sum()
            if max_idx is None or v > max_val:
                max_val = v
                max_idx = ix
        max_x = max_idx

        max_idx = None
        max_val = None
        for iy in range(expected_array.shape[1]):
            v = expected_array[:, iy, :].sum()
            if max_idx is None or v > max_val:
                max_val = v
                max_idx = iy
        max_y = max_idx

        max_idx = None
        max_val = None
        for iz in range(expected_array.shape[2]):
            v = expected_array[:, :, iz].sum()
            if max_idx is None or v > max_val:
                max_val = v
                max_idx = iz
        max_z = max_idx
        expected = [max_x, max_y, max_z]
        np.testing.assert_array_equal(zattrs['max_planes'], expected)

    # test that scale in .zattrs is as expected

    datasets = zattrs['multiscales'][0]['datasets']
    for d in datasets:
        scale = d['coordinateTransformations'][0]['scale']
        if d['path'] == '0':
            if do_transposition:
                expected = (float(pixdim_fixture[0]),
                 float(pixdim_fixture[1]),
                 float(pixdim_fixture[2]))
            else:
                expected = (float(pixdim_fixture[2]),
                 float(pixdim_fixture[1]),
                 float(pixdim_fixture[0]))

            np.testing.assert_allclose(
                scale,
                expected,
                rtol=1.0e-6,
                atol=0.0)
        if d['path'] == '1':

            if do_transposition:
                expected = (float(pixdim_fixture[0])*5,
                 float(pixdim_fixture[1])*3,
                 float(pixdim_fixture[2])*2)
            else:
                expected = (float(pixdim_fixture[2])*2,
                 float(pixdim_fixture[1])*3,
                 float(pixdim_fixture[0])*5)

            np.testing.assert_allclose(
                scale,
                expected,
                rtol=1.0e-6,
                atol=0.0)
