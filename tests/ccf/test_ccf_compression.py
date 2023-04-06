import pytest

import numpy as np
import pathlib
import SimpleITK
import tempfile
from unittest.mock import patch

from neuroglancer_interface.utils.utils import (
    _clean_up,
    mkstemp_clean)

from neuroglancer_interface.modules.ccf_multiscale_annotations import (
    write_out_ccf,
    get_scale_metadata,
    get_scale_metadata_with_downsampling)


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
def ccf_array_fixture(shape_fixture):
    rng = np.random.default_rng(556623)
    return rng.integers(1, 100, shape_fixture)

@pytest.fixture
def pixdim_fixture():
    return ('0.01', '0.02', '0.03')

@pytest.fixture
def ccf_nii_fixture(
        ccf_array_fixture,
        pixdim_fixture,
        temp_dir_fixture):
    sub_dir = temp_dir_fixture / 'nii'
    sub_dir.mkdir()
    tmp_path = mkstemp_clean(
            dir=sub_dir,
            suffix='.nii')

    img = SimpleITK.GetImageFromArray(
        ccf_array_fixture)

    SimpleITK.WriteImage(
        image=img,
        fileName=tmp_path)

    return tmp_path

@pytest.fixture
def itk_label_fixture(
        temp_dir_fixture):

    tmp_path = mkstemp_clean(
        dir=temp_dir_fixture,
        suffix='.txt')

    with open(tmp_path, 'w') as out_file:
        for ii in range(1, 100, 1):
            out_file.write(f'{ii} {2*ii} {3*ii} "label_{ii}"')
    return tmp_path


def test_ccf_smoketest(
        itk_label_fixture,
        ccf_nii_fixture,
        pixdim_fixture,
        temp_dir_fixture):

    sub_dir = pathlib.Path(
        tempfile.mkdtemp(dir=temp_dir_fixture))

    # because SimpleITK.WriteImage does not write out metadata
    def mock_get_metadata(self, value):
        lookup = dict()
        lookup['pixdim[1]'] = pixdim_fixture[0]
        lookup['pixdim[2]'] = pixdim_fixture[1]
        lookup['pixdim[3]'] = pixdim_fixture[2]
        return lookup[value]

    with patch('SimpleITK.Image.GetMetaData', new=mock_get_metadata):
        write_out_ccf(
            segmentation_path_list=[ccf_nii_fixture],
            label_path=itk_label_fixture,
            output_dir=sub_dir,
            use_compression=True,
            compression_blocksize=32,
            chunk_size=(32, 32, 32))
    expected = f"{int(float(pixdim_fixture[2])*1000000)}_"
    expected += f"{int(float(pixdim_fixture[1])*1000000)}_"
    expected += f"{int(float(pixdim_fixture[0])*1000000)}"
    expected = sub_dir / expected
    assert expected.is_dir()


def test_get_scale_downsampling_smoketest(
        ccf_nii_fixture,
        pixdim_fixture,
        temp_dir_fixture):

    sub_dir = tempfile.mkdtemp(dir=temp_dir_fixture)
    sub_dir = pathlib.Path(sub_dir)

    # because SimpleITK.WriteImage does not write out metadata
    def mock_get_metadata(self, value):
        lookup = dict()
        lookup['pixdim[1]'] = pixdim_fixture[0]
        lookup['pixdim[2]'] = pixdim_fixture[1]
        lookup['pixdim[3]'] = pixdim_fixture[2]
        return lookup[value]

    with patch('SimpleITK.Image.GetMetaData', new=mock_get_metadata):
        baseline_metadata = get_scale_metadata(
            segmentation_path=ccf_nii_fixture,
            chunk_size=(32, 32, 32),
            use_compression=True,
            compression_blocksize=32,
            do_transposition=False)

        downsampled_metadata = get_scale_metadata_with_downsampling(
            segmentation_path=ccf_nii_fixture,
            tmp_dir=sub_dir,
            downsample_min=0,
            chunk_size=(32, 32, 32),
            use_compression=True,
            compression_blocksize=32,
            do_transposition=False)

    assert len(downsampled_metadata) > 0
    baseline_keys = set(baseline_metadata.keys())
    unq_files = set()
    for actual in downsampled_metadata:
        actual_keys = set(actual.keys())
        assert actual_keys == baseline_keys
        fpath = pathlib.Path(actual['local_file_path'])
        assert fpath.is_file()
        unq_files.add(fpath.name)
    assert len(unq_files) == len(downsampled_metadata)
