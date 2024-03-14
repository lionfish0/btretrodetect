from retrodetect.detect import detect, detectcontact
import numpy as np
import pytest
import cv2
import os
from numpy import load
from glob import glob
import pickle


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


@pytest.fixture
def photo_list():
    return pickle.load(open(os.path.join(os.getcwd(), "tests", "data", "trial30"), 'rb'))


@pytest.mark.parametrize(
    "n, contact, found, searchimg",
    [
        (0,
         None,
         False,
         None
         ),
        (1,
         None,
         False,
         None
         ),
        (2,
         None,
         False,
         None
         )

    ]
)
def test_detectcontact(photo_list, n, contact, found, searchimg):
    output = detectcontact(photo_list, n)
    expected_output = (contact, found, searchimg)
    assert output == expected_output
