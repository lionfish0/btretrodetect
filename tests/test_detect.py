from retrodetect.detect import detect
import numpy as np
import pytest
import cv2
import os
from numpy import load


@pytest.fixture
def img_1():
    return cv2.imread(os.path.join(os.getcwd(), "tests", "data", "flash1.jpg"), 0).astype(float)


@pytest.fixture
def img_2():
    return cv2.imread(os.path.join(os.getcwd(), "tests", "data", "noflash1.jpg"), 0).astype(float)


def test_detect_default(img_1, img_2):
    output = detect(img_1, img_2)
    expected_output = load((os.path.join(os.getcwd(), "tests", "data", "detect_output.npz")))
    expected_output = expected_output['arr_0']
    assert np.allclose(output, expected_output)
