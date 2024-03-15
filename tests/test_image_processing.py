from retrodetect.image_processing.image_processing import getshift, ensemblegetshift, getblockmaxedimage, \
    alignandsubtract, shiftimg
import numpy as np
import pytest
import cv2
import os
from numpy import load

##SC: do we want to add to add tolerance for comparison of np array
#https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
@pytest.fixture
def img_1():
    return cv2.imread(os.path.join(os.getcwd(), "tests", "data", "flash1.jpg"), 0).astype(float)


@pytest.fixture
def img_2():
    return cv2.imread(os.path.join(os.getcwd(), "tests", "data", "noflash1.jpg"), 0).astype(float)


## getshift
##SC: I suspect the invalid value of start and end largely depends on the size.shape
## of input images, which might affect the normxcorr2 (which in theory should only allow
## template smaller than input image. So perhaps the test should derive from there
def test_getshift_default(img_1, img_2):
    expected_output = ([-12, 20])
    output = getshift(img_1, img_2)
    assert np.allclose(output, expected_output)


@pytest.mark.parametrize(
    "start, end, searchbox, step, expected_output",
    [
        (
                ([200, 200]),
                ([2000, 2000]),
                100,
                8,
                ([-20, 20])
        ),
        (
                None,
                None,
                200,
                8,
                ([-16, 24])
        ),
        (
                None,
                None,
                20,
                2,
                ([-12, 20])
        )
    ]
)
def test_getshift_different_start_end_searchbox_step(img_1, img_2, start, end, searchbox, step, expected_output):
    output = getshift(img_1, img_2, start, end, searchbox, step)
    assert np.allclose(output, expected_output)


def test_ensemblegetshift_default(img_1, img_2):
    assert ensemblegetshift(img_1, img_2) == [-20, 12]


@pytest.mark.parametrize(
    "searchblocksize, ensemblesizesqrt, step, expected_output",
    [
        (100,
         4,
         8,
         [-16, 20]
         ),
        (200,
         3,
         8,
         [-12, 20]
         ),
        (250,
         5,
         8,
         [-12, 20]
         ),
        (50,
         3,
         2,
         [-16, 18]
         ),
    ]
)
def test_ensemblegetshift_different_searchblocksize_ensemblesizesqrt(img_1, img_2, searchblocksize, ensemblesizesqrt,
                                                                     step,
                                                                     expected_output):
    output = ensemblegetshift(img_1, img_2, 100, step, searchblocksize, ensemblesizesqrt)
    assert output == expected_output


def test_getblockmaxedimage_default(img_2):
    output = getblockmaxedimage(img_2, 2, 3)
    expected_output = load((os.path.join(os.getcwd(), "tests", "data", "getblockmaxedimage_output.npz")))
    expected_output = expected_output['arr_0']
    assert np.allclose(output, expected_output)


def test_alignandsubtract_default(img_1):
    noflash = load((os.path.join(os.getcwd(), "tests", "data", "getblockmaxedimage_output.npz")))
    noflash = noflash['arr_0']
    flash = img_1
    shift = [-16, 18]
    margin = 100
    expected_output = load((os.path.join(os.getcwd(), "tests", "data", "alignandsubtract_output.npz")))
    expected_output = expected_output['arr_0']
    output = alignandsubtract(noflash, shift, flash, None, None, margin)
    assert np.allclose(output, expected_output)


@pytest.fixture
def test_image():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def cval():
    return 255  # SC: can be any integer


@pytest.mark.parametrize(
    "shift, expected",
    [
        # Positive x-shift, positive y-shift
        ((1, 1), np.array([[255, 255, 255], [255, 1, 2]])),
        # Positive x-shift, negative y-shift
        ((1, -1), np.array([[255, 255, 255], [2, 3, 255]])),
        # Positive x-shift, no y-shift
        ((1, 0), np.array([[255, 255, 255], [1, 2, 3]])),
        # Negative x-shift, positive y-shift
        ((-1, 1), np.array([[255, 4, 5], [255, 255, 255]])),
        # Negative x-shift, negative y-shift
        ((-1, -1), np.array([[5, 6, 255], [255, 255, 255]])),
        # Negative x-shift, no y-shift
        ((-1, 0), np.array([[4, 5, 6], [255, 255, 255]])),
        # No x-shift, positive y-shift
        ((0, 1), np.array([[255, 1, 2], [255, 4, 5]])),
        # No x-shift, negative y-shift
        ((0, -1), np.array([[2, 3, 255], [5, 6, 255]])),
        # No shift
        ((0, 0), np.array([[1, 2, 3], [4, 5, 6]])),
    ],
)
def test_shiftimg(test_image, shift, cval, expected):
    result = shiftimg(test_image, shift, cval)
    assert np.allclose(result, expected)
