import numpy as np

from neuroglancer_interface.utils.rotation_utils import (
    rotate_matrix)


def test_rotate_matrix():
    rng = np.random.default_rng(665234)
    base_arr = rng.random((5, 6, 7))

    actual = rotate_matrix(
        data=base_arr,
        rotation_matrix = [[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0]])
    expected = base_arr.transpose(2, 1, 0)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=0.0)

    actual = rotate_matrix(
        data=base_arr,
        rotation_matrix = [[0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 0]])
    expected = base_arr.transpose(1, 2, 0)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=0.0)


    actual = rotate_matrix(
        data=base_arr,
        rotation_matrix = [[0, 1, 0],
                           [0, 0, -1],
                           [1, 0, 0]])
    expected = np.zeros((6, 7, 5), dtype=float)
    for ix in range(6):
        for iy in range(7):
            for iz in range(5):
                expected[ix, iy, iz] = base_arr[iz, ix, base_arr.shape[2]-1-iy]
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=0.0)
