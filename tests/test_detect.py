from retrodetect.detect import detect, detectcontact
import numpy as np
import pytest
import cv2
import os
from numpy import load
import pickle
import tempfile
import zipfile
from glob import glob
import itertools

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

#temporary dir fixture https://docs.pytest.org/en/6.2.x/tmpdir.html#the-tmpdir-fixture
@pytest.fixture(scope="session")
def photo_list(tmpdir_factory):
    p = tmpdir_factory.mktemp("photo_object")
    with zipfile.ZipFile('tests/data/demo.zip') as myzip:
        myzip.extractall(p)
    file_list = []
    for imfilename in sorted(glob(os.path.join(p, 'photo_object*.np'))):
        photoitem = pickle.load(open(imfilename, 'rb'))
        file_list.append(photoitem)

    return file_list

@pytest.fixture
def contact_expected_output():
    with open('tests/output_contact_n3', 'rb') as file:
        output = pickle.load(file)
    file.close()
    return output



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

def test_detectcontact(photo_list,contact_expected_output):
    contact, found, searchimg = detectcontact(photo_list,3)
    assert isinstance(contact, list)
    assert all(isinstance(item, dict) for item in contact)
    assert len(contact) == 20
    assert found is False
    expected_searchimg = load((os.path.join(os.getcwd(), "tests", "data", "searchimg_3.npz")))
    expected_searchimg = expected_searchimg['arr_0']
    assert np.allclose(searchimg, expected_searchimg)

    ## testing items in contact list
    for item_tested, item_expected in itertools.zip_longest(contact, contact_expected_output):
        assert np.allclose(item_tested.pop('patch'), item_expected.pop('patch'))
        assert np.allclose(item_tested.pop('searchpatch'), item_expected.pop('searchpatch'))
        assert item_tested == item_expected



