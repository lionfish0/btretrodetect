"""
This module contains functions used to process the images that are taken for detection of the reflective tags.
"""
import numpy as np
from .normxcorr2 import normxcorr2

def shiftimg(
        test: np.array,
        shift: tuple, #SC: however, the function (ensemblegetshift) which output for the 'shift' of this function actually returns a list. so not sure how it can work
        cval: int
) -> np.array:
    """
    Returns an image in NumPy array resulting from the shift given.
    :param test: A NumPy array representing the input image.
    :param shift: A tuple of (x, y) representing the amount to shift the image.
    :param cval: The value to fill the empty space created by the shift
    :return: A new NumPy array representing the shifted image.
    """
    new = np.full_like(test, cval)
    if shift[0] > 0:
        if shift[1] > 0:
            new[shift[0]:, shift[1]:] = test[:-shift[0], :-shift[1]]
        if shift[1] < 0:
            new[shift[0]:, :shift[1]] = test[:-shift[0], -shift[1]:]
        if shift[1] == 0:
            new[shift[0]:, :] = test[:-shift[0], :]
    if shift[0] < 0:
        if shift[1] > 0:
            new[:shift[0], shift[1]:] = test[-shift[0]:, :-shift[1]]
        if shift[1] < 0:
            new[:shift[0], :shift[1]] = test[-shift[0]:, -shift[1]:]
        if shift[1] == 0:
            new[:shift[0], :] = test[-shift[0]:, :]
    if shift[0] == 0:
        if shift[1] > 0:
            new[:, shift[1]:] = test[:, :-shift[1]]
        if shift[1] < 0:
            new[:, :shift[1]] = test[:, -shift[1]:]
        if shift[1] == 0:
            new[:, :] = test[:, :]
    return new


def getshift(
        img_1: np.array,
        img_2: np.array,
        start: np.array = None,
        end: np.array = None,
        searchbox: int = 100,
        step: int = 8
) -> np.array:
    """
    Returns the optimal shift to align a cropped region of img_1 with img_2.
    :param img_1: The input image from which a sub-region will be aligned.
    :param img_2: The reference image to which the sub-region of img_1 will be aligned.
    :param start: The starting indices (x, y) for the sub-region within img_1. Defaults to [searchbox, searchbox].
    :param end: The ending indices (exclusive) for the sub-region within img_1. Defaults to the sub-region ending searchbox pixels from the edge of img_1.
    :param searchbox: The maximum distance to search for the optimal shift in each direction. Defaults to 100 pixels.
    :param step: The step size used when searching for the optimal shift. Defaults to 8 pixels.
    :return: The optimal shift amount (x, y) as a NumPy array. The shift represents the pixel offset needed to align the center of the cropped region in `imgA` with the corresponding region in `imgB`.

    """

    # SC: Do we want to do this?  # * **ValueError:** If `start` or `end` are not compatible shapes with `imgA.shape`.
    if start is None:
        start = np.array([searchbox, searchbox])
    if end is None:
        end = np.array(img_1.shape) - searchbox

    img_2 = img_2[start[0]:end[0], start[1]:end[1]]

    # SC this seems redundant, as it is just adding back to what is cropped if Start is none.
    # SC it seems to make sure the ImgApart is larger than the crops img2
    imgApart = img_1[(start[0] - searchbox):(end[0] + searchbox),
               (start[1] - searchbox):(end[1] + searchbox)]
    temp = normxcorr2(img_2[::step, ::step],
                      imgApart[::step, ::step], mode='valid')
    shift = step * np.array(np.unravel_index(temp.argmax(), temp.shape))
    shift = shift - searchbox
    return shift


def ensemblegetshift(
        img_1: np.array,
        img_2: np.array,
        searchbox: int = 100,
        step: int = 8,
        searchblocksize: int = 50, ensemblesizesqrt=3
) -> list:
    """
    searchblock: how big each search image pair should be.
    ensemblesizesqrt: number of items for ensemble for one dimension.

    """
    starts = []
    for x in np.linspace(0, img_1.shape[0], ensemblesizesqrt + 2)[1:-1].astype(int):
        for y in np.linspace(0, img_1.shape[1], ensemblesizesqrt + 2)[1:-1].astype(int):
            starts.append([x, y])

    shifts = np.zeros([len(starts), 2])
    for i, start in enumerate(starts):
        shifts[i] = getshift(img_1, img_2, step=step, searchbox=searchbox,
                             start=start, end=start + np.array([searchblocksize, searchblocksize]))
    medianshift = np.median(shifts, 0)
    medianshift = [int(medianshift[0]), int(medianshift[1])]
    return medianshift


def getblockmaxedimage(img, blocksize, offset):
    """
    Effectively replaces each pixel with approximately the maximum of all the
    pixels within offset*blocksize of the pixel (in a square).

    Get a new image of the same size, but filtered such that each square patch
    of blocksize has its maximum calculated, then a search box of size
    (1+offset*2)*blocksize centred on each pixel is applied which finds the
    maximum of these patches.

    img = image to apply the filter to
    blocksize = size of the squares
    offset = how far from the pixel to look for maximum
    """
    # import time
    # times = []
    # times.append(time.time())
    k = int(img.shape[0] / blocksize)
    l = int(img.shape[1] / blocksize)
    if blocksize == 1:
        maxes = img
    else:
        # from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
        maxes = img[:k * blocksize, :l *
                                     blocksize].reshape(k, blocksize, l, blocksize).max(axis=(-1, -3))
    # times.append(time.time())
    templist = []
    xm, ym = maxes.shape
    i = 0
    # (if offset=1, for xoff in [0]) (if offset=2, for xoff in [-1,0,1])...
    for xoff in range(-offset + 1, offset, 1):
        for yoff in range(-offset + 1, offset, 1):
            if i == 0:
                max_img = maxes[xoff + offset:xoff + xm -
                                              offset, yoff + offset:yoff + ym - offset]
            else:
                max_img = np.maximum(
                    max_img, maxes[xoff + offset:xoff + xm - offset, yoff + offset:yoff + ym - offset])
            i += 1
            # templist.append(maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
    # times.append(time.time())
    # max_img = templist[0]
    # for im in templist[1:]:
    #    max_img = np.maximum(max_img,im)
    # times.append(time.time())
    out_img = np.full_like(img, 255)
    # times.append(time.time())
    inner_img = max_img.repeat(blocksize, axis=0).repeat(blocksize, axis=1)
    # times.append(time.time())
    # s = (out_img.shape-new_inner_img.shape)/2
    out_img[blocksize * offset:(blocksize * offset + inner_img.shape[0]),
    blocksize * offset:(blocksize * offset + inner_img.shape[1])] = inner_img
    # times.append(time.time())
    # print("----------")
    # print(np.diff(times))
    #    out_img[:blocksize*offset,:] = 255
    #    out_img[-(blocksize*offset):,:] = 255
    #    out_img[:,:blocksize*offset] = 255
    #    out_img[:,-(blocksize*offset):] = 255
    return out_img


def alignandsubtract(subimg, shift, foreimg, start=None, end=None, margin=100):
    """
    Subtract subimg (after shifting) from foreimg, 
    removing a box defined by start and end (or
    margin (default=100) around edge if not specified by start and end)"""
    if start is None:
        start = np.array([margin, margin])
    if end is None:
        end = np.array(subimg.shape) - np.array([margin, margin])

    subimgshifted = shiftimg(
        subimg[start[0]:end[0], start[1]:end[1]], shift, cval=255)
    temp = foreimg.copy()[start[0]:end[0], start[1]:end[1]]
    temp -= subimgshifted
    return temp
