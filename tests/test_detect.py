from retrodetect.detect import detect, detectcontact
import numpy as np
import pytest
import cv2
import os
from numpy import load
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
def test_detectcontact_fewer_than_two_photosets(photo_list, n, contact, found, searchimg):
    output = detectcontact(photo_list, n)
    expected_output = (contact, found, searchimg)
    assert output == expected_output

def test_detectcontact(photo_list):
    contact, found, searchimg = detectcontact(photo_list,3)
    assert isinstance(contact, list)
    assert all(isinstance(item, dict) for item in contact)
    assert found == False
    expected_searchimg = load((os.path.join(os.getcwd(), "tests", "data", "searchimg_3.npz")))
    expected_searchimg = expected_searchimg['arr_0']
    assert np.allclose(searchimg, expected_searchimg)


