import pytest

import numpy as np

from neuroglancer_interface.utils.ccf_utils import (
    downsample_segmentation_array)

from neuroglancer_interface.modules.ccf_multiscale_annotations import (
    _create_pyramid_of_ccf_downsamples)


def test_downsample_segmentation_array():

    baseline = np.arange((6*20*14), dtype=int).reshape((6, 20, 14))
    actual = downsample_segmentation_array(
        arr=baseline,
        downsample_by=(3, 5, 7))

    expected = np.zeros((2, 4, 2), dtype=int)

    expected[0, 0, 0] = baseline[1, 2, 3]
    expected[0, 0, 1] = baseline[1, 2, 10]
    expected[0, 1, 0] = baseline[1, 7, 3]
    expected[0, 1, 1] = baseline[1, 7, 10]
    expected[0, 2, 0] = baseline[1, 12, 3]
    expected[0, 2, 1] = baseline[1, 12, 10]
    expected[0, 3, 0] = baseline[1, 17, 3]
    expected[0, 3, 1] = baseline[1, 17, 10]
    expected[1, 0, 0] = baseline[4, 2, 3]
    expected[1, 0, 1] = baseline[4, 2, 10]
    expected[1, 1, 0] = baseline[4, 7, 3]
    expected[1, 1, 1] = baseline[4, 7, 10]
    expected[1, 2, 0] = baseline[4, 12, 3]
    expected[1, 2, 1] = baseline[4, 12, 10]
    expected[1, 3, 0] = baseline[4, 17, 3]
    expected[1, 3, 1] = baseline[4, 17, 10]

    np.testing.assert_array_equal(expected, actual)

    actual = downsample_segmentation_array(
        arr=baseline,
        downsample_by=(1, 5, 7))

    expected = np.zeros((6, 4, 2), dtype=int)
    for ix in range(6):
        expected[ix, 0, 0] = baseline[ix, 2, 3]
        expected[ix, 0, 1] = baseline[ix, 2, 10]
        expected[ix, 1, 0] = baseline[ix, 7, 3]
        expected[ix, 1, 1] = baseline[ix, 7, 10]
        expected[ix, 2, 0] = baseline[ix, 12, 3]
        expected[ix, 2, 1] = baseline[ix, 12, 10]
        expected[ix, 3, 0] = baseline[ix, 17, 3]
        expected[ix, 3, 1] = baseline[ix, 17, 10]

    np.testing.assert_array_equal(expected, actual)

    actual = downsample_segmentation_array(
        arr=baseline,
        downsample_by=(3, 1, 7))
    expected = np.zeros((2, 20, 2), dtype=int)
    for iy in range(20):
        expected[0, iy, 0] = baseline[1, iy, 3]
        expected[0, iy, 1] = baseline[1, iy, 10]
        expected[1, iy, 0] = baseline[4, iy, 3]
        expected[1, iy, 1] = baseline[4, iy, 10]


    with pytest.raises(RuntimeError, match="divisible"):
        downsample_segmentation_array(
            arr=baseline,
            downsample_by=(7, 5, 7))

    with pytest.raises(RuntimeError, match="odd"):
        downsample_segmentation_array(
            arr=baseline,
            downsample_by=(2, 5, 7))

    with pytest.raises(RuntimeError, match="positive definite"):
        downsample_segmentation_array(
            arr=baseline,
            downsample_by=(3, 0, 7))


def test_create_ccf_downsample_pyramid():

    baseline_shape = (2*2*3*5*11,
                      2*2*2*5*13*19,
                      2*2*2*2*2*2)

    actual = _create_pyramid_of_ccf_downsamples(
        baseline_shape=baseline_shape,
        downsample_cutoff=0)

    expected = [(3, 5, 1),
                (15, 65, 1),
                (15*11, 65*19, 1)]

    assert actual == expected

    # add a cutoff
    actual = _create_pyramid_of_ccf_downsamples(
        baseline_shape=baseline_shape,
        downsample_cutoff=64)

    expected = [(3, 5, 1),
                (3, 65, 1)]

    assert actual == expected

    # make second dimension the one that is only even
    baseline_shape = (2*2*3*5*11,
                      2**16,
                      2*2*2*5*13*19)

    actual = _create_pyramid_of_ccf_downsamples(
        baseline_shape=baseline_shape,
        downsample_cutoff=0)

    expected = [(3, 1, 5),
                (15, 1, 65),
                (15*11, 1, 65*19)]

    assert actual == expected
