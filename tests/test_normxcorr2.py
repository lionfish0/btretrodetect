from bee_retrodetect.image_processing.normxcorr2 import normxcorr2
import numpy as np
import pytest


@pytest.mark.parametrize(
    "template, image, mode, expected_output",
    [
        (
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]]),
                "full",
                np.array([[0.55555556, 0.61548548, 0.55555556],
                          [0.61548548, 0.68041382, 0.61548548],
                          [0.55555556, 0.61548548, 0.55555556]]),
        ),
        (
                np.array([[1, 2, 3]]),
                np.array([[4, 5, 6], [7, 8, 9]]),
                "valid",
                np.array([[0.54772256]]),
        ),
        (
                np.array([[1, 2]]),
                np.array([[5, 6], [8, 9], [11, 12]]),
                "same",
                np.array([[0.55555556, 0.61548548],
                          [0.61548548, 0.68041382],
                          [0.55555556, 0.61548548]]),
        ),
    ]
)
def test_normxcorr2_correctness(template, image, mode, expected_output):
    result = normxcorr2(template, image, mode)
    assert np.allclose(result, expected_output)


def test_normxcorr2_template_larger_than_image():
    template = np.ones((5, 5))
    image = np.ones((3, 3))
    with pytest.raises(TypeError, match="TEMPLATE larger than IMG"):
        normxcorr2(template, image)


def test_normxcorr2_invalid_mode():
    template = np.ones((2, 2))
    image = np.ones((3, 3))
    with pytest.raises(ValueError, match="Invalid mode"):
        normxcorr2(template, image, mode="invalid_mode")
