import pytest

import numpy as np

from neuroglancer_interface.utils.ccf_utils import (
    downsample_segmentation_array)


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
