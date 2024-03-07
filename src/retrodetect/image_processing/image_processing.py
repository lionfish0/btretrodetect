"""
This module contains functions used to process the images that are taken for detection of the reflective tags.
"""
import numpy as np
from .normxcorr2 import normxcorr2


def shiftimg(
        test: np.array,
        shift: tuple,
        # SC: however, the function (ensemblegetshift) which output for the 'shift' of this function actually returns a list. so not sure how it can work
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
        searchblocksize: int = 50,
        ensemblesizesqrt=3
) -> list:  # but the output of this function needs to be a tuple for use in another function alignandsubtract
    """#SC is this correct?
    Calculates the median shift to align img_1 with img_2 using an ensemble approach.

    searchblock: how big each search image pair should be.
    ensemblesizesqrt: number of items for ensemble for one dimension.

    :param img_1: The input image from which sub-regions will be extracted and aligned.
    :param img_2: The reference image to which different sub-regions of img_1 will be aligned.
    :param searchbox: The maximum distance to search for the optimal shift in each direction within each sub-region. Defaults to 100 pixels.
    :param step: The step size used when searching for the optimal shift within each sub-region. Defaults to 8 pixels.
    :param searchblocksize: The size of the square sub-regions extracted from img_1 for alignment. Defaults to 50 pixels.
    :param ensemblesizesqrt: The square root of the number of sub-regions to be extracted from each dimension of img_1 for the ensemble. Defaults to 3, resulting in a total of 9 sub-regions.
    :return: A list containing the median shift (x, y) as integers.

    **Notes:**

* The function creates an ensemble of sub-regions by extracting them from `imgA` in a grid-like fashion based on `ensemblesizesqrt` and `searchblocksize`.
* For each sub-region, the `getshift` function is used to find the optimal shift to align it with `imgB`.
* The median shift across all the sub-regions in the ensemble is then calculated and returned.
* This approach aims to provide a more robust estimation of the overall shift by considering alignments at multiple locations within `imgA`.

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


def getblockmaxedimage(
        img: np.array,
        blocksize: int,
        offset: int
) -> np.array:
    """
    Applies a filter to an image that replaces each pixel with the approximate maximum within a local neighborhood (offset*blocksize).

    :param img: The input image.
    :param blocksize: The size of the square patches used for finding local maxima.
    :param offset: The extent of the neighborhood around each pixel to search for the maximum, expressed as a multiple of the `blocksize`.
    :return: A new image with the same shape as the input image, where each pixel is replaced by the approximate maximum within its local neighborhood.
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


def alignandsubtract(
        subimg: np.array,
        shift: tuple,
        foreimg: np.array,
        start: np.array = None,
        end: np.array = None,
        margin: int = 100
) -> np.array:
    """
    Subtracts a shifted sub-image from a foreground image, removing a region defined by start and end or
    by margin (default=100) around edge if not specified by start and end
    :param subimg: The sub-image to be subtracted.
    :param shift: The amount to shift the subimg before subtraction.
    :param foreimg: The foreground image from which the shifted subimg will be subtracted.
    :param start: The starting indices (x, y) for the region to be removed from the `foreimg`. Defaults to a margin of `margin` pixels from the edges.
    :param end: The ending indices (exclusive) for the region to be removed from the `foreimg`. Defaults to a margin of `margin` pixels from the edges, calculated based on the `subimg` shape.
    :param margin: The margin (in pixels) to use around the edges of the image if `start` and `end` are not specified. Defaults to 100 pixels.
    :return: A new image of the same shape as the specified region in `foreimg` after subtracting the shifted `subimg`.


#SC: do we want to do this part
**Raises:**

* **ValueError:** If `start` or `end` are not compatible shapes with `subimg.shape`.

**Notes:**

* If `start` and `end` are not provided, the function removes a rectangular region of size equal to the `subimg` with a margin of `margin` pixels from the edges of the `foreimg`.
* The `shiftimg` function is used to shift the `subimg` by the specified `shift` amount before subtraction.
* The specified region in the `foreimg` is replaced by the element-wise difference between the original and shifted `subimg`.

"""
    if start is None:
        start = np.array([margin, margin])
    if end is None:
        end = np.array(subimg.shape) - np.array([margin, margin])

    subimgshifted = shiftimg(
        subimg[start[0]:end[0], start[1]:end[1]], shift, cval=255)
    temp = foreimg.copy()[start[0]:end[0], start[1]:end[1]]
    temp -= subimgshifted
    return temp
