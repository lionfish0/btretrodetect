from retrodetect.image_processing.normxcorr2 import normxcorr2
import numpy as np
import pytest


@pytest.mark.parametrize(
    "template, image, mode, expected_output",
    [
        (
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]]),
                "full",
                np.array([[-0.77459667, -0.81409158, -0.77459667, -0.25819889],
                          [0.06819943, 0.98994949, 0.98994949, 0.71818485],
                          [0.71818485, 0.98994949, 0.98994949, 0.06819943],
                          [-0.25819889, -0.77459667, -0.81409158, -0.77459667]]),
        ),
        (
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]]),
                "valid",
                np.array([[0.98994949, 0.98994949],
                          [0.98994949, 0.98994949]]),
        ),
        (
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]]),
                "same",
                np.array([[-0.77459667, -0.81409158, -0.77459667],
                          [0.06819943,  0.98994949,  0.98994949],
                          [0.71818485,  0.98994949,  0.98994949]]),
        ),
    ]
)

def test_normxcorr2_correctness(template, image, mode, expected_output):
    result = normxcorr2(template, image, mode)
    assert np.allclose(result, expected_output)



