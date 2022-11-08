import numpy as np

from mani_skill2.utils.common import compute_angle_between


def test_compute_angle_between():
    np.testing.assert_allclose(compute_angle_between([0, 0, 0], [0, 0, 0]), np.pi / 2)
    np.testing.assert_allclose(compute_angle_between([0, 0, 0], [0, 0, 1]), np.pi / 2)
    np.testing.assert_allclose(compute_angle_between([0, 1, 0], [0, 0, 1]), np.pi / 2)
    np.testing.assert_allclose(
        compute_angle_between([1, 0, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3), 0]),
        np.pi / 3,
    )
