from retrodetect.image_processing.image_processing import getshift
import numpy as np
import pytest
import cv2
import os


@pytest.fixture
def img_1():
    return cv2.imread(os.path.join("data", "flash1.jpg"), 0).astype(float)


@pytest.fixture
def img_2():
    return cv2.imread(os.path.join("data", "noflash1.jpg"), 0).astype(float)


def test_getshift_default(img_1, img_2):
    expected_result = ([-12, 20])
    result = getshift(img_1, img_2)
    assert np.allclose(result, expected_result)

