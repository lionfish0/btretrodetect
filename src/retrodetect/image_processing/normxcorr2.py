"""This module contains Octave/Matlab normalized cross-correlation implementation in python 3.5.

Returns:
    np.ndarray[np.float32, Any]: correlation coefficients of the template and image
"""
# Author: Ujash Joshi, University of Toronto, 2017
# Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>
# https://github.com/Sabrewarrior/normxcorr2-python
# http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html
# Addition of docstring and modification is made for clarification of code

import numpy as np
from scipy.signal import fftconvolve


def normxcorr2(
        template: np.array,
        image: np.array,
        mode: str = "full"
) -> np.array:
    """Calculates how similar the template appears within different locations of the input image by computing the
    normalized cross-correlation of the template and image. It returns a matrix containing the correlation coefficients.
    :param template: The N-dimensional template used for the correlation. Each dimension of the template must be smaller than the corresponding dimension of the image.
    :param image: The N-dimensional input image.
    :param mode: The mode for the `fftconvolve` function, controlling the output size. Defaults to `"full"`.
    * "full": The output includes all elements of the convolution, resulting in an output size that is the sum of the image and template sizes minus 1 in each dimension.
    * "valid": The output only contains elements that do not rely on zero-padding, resulting in an output size smaller than both the image and template sizes.
    * "same": The output has the same size as the image, centered with respect to the `"full"` output.
    :return: An N-dimensional array of the same dimensionality as the `image` but with a size that depends on the chosen `mode`.
    """
    # SC: need to do try break
    # If this happens, it is probably a mistake
    #    if np.ndim(template) > np.ndim(image) or \
    #            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
    #        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / \
            (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out
