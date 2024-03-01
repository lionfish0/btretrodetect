from retrodetect.image_processing.image_processing import getshift, ensemblegetshift, getblockmaxedimage, \
    alignandsubtract
import numpy as np
import pytest
import cv2
import os
from numpy import load


@pytest.fixture
def img_1():
    return cv2.imread(os.path.join("data", "flash1.jpg"), 0).astype(float)


@pytest.fixture
def img_2():
    return cv2.imread(os.path.join("data", "noflash1.jpg"), 0).astype(float)


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
    expected_output = load((os.path.join("data", "getblockmaxedimage_output.npy")))
    assert np.allclose(output, expected_output)


def test_alignandsubtract_default(img_1):
    noflash = load((os.path.join("data", "getblockmaxedimage_output.npy")))
    flash = img_1
    shift = [-16, 18]
    margin = 100
    expected_output = load((os.path.join("data", "alignandsubtract_output.npy")))
    output = alignandsubtract(noflash, shift, flash, None, None, margin)
    assert np.allclose(output, expected_output)
