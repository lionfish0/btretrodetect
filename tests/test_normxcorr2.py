from retrodetect.image_processing.normxcorr2 import normxcorr2
import numpy as np
import pytest


@pytest.fixture
def template():
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def image():
    return np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])


@pytest.mark.parametrize(
    "mode, expected_output",
    [
        ("full",
         np.array([[-0.77459667, -0.81409158, -0.77459667, -0.25819889],
                   [0.06819943, 0.98994949, 0.98994949, 0.71818485],
                   [0.71818485, 0.98994949, 0.98994949, 0.06819943],
                   [-0.25819889, -0.77459667, -0.81409158, -0.77459667]]),
         ),
        ("valid",
         np.array([[0.98994949, 0.98994949],
                   [0.98994949, 0.98994949]]),
         ),
        ("same",
         np.array([[-0.77459667, -0.81409158, -0.77459667],
                   [0.06819943, 0.98994949, 0.98994949],
                   [0.71818485, 0.98994949, 0.98994949]]),
         ),
    ]
)
def test_normxcorr2_correctness(template, image, mode, expected_output):
    result = normxcorr2(template, image, mode)
    assert np.allclose(result, expected_output)

def test_normxcorr2_invalid_mode():
    template = np.array([[1, 2], [3, 4]])
    image = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        normxcorr2(template, image, mode="invalid_mode")
