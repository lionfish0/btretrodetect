#Author: Ujash Joshi, University of Toronto, 2017                    
#Based on Octave implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de> 
#https://github.com/Sabrewarrior/normxcorr2-python
#http://lordsabre.blogspot.ca/2017/09/matlab-normxcorr2-implemented-in-python.html
#Addition of docstring is made for clarification of code   

    """This module contains Octave/Matlab normalized cross-correlation implementation in python 3.5.

    Returns:
        _type_: _description_
    """
from typing import Any  # for type hinting for ndarray
import numpy as np
from scipy.signal import fftconvolve


def normxcorr2(template: np.ndarray[np.float32, Any], image: np.ndarray[np.float32, Any], mode: str = "full") -> np.ndarray[np.float32, Any]:
    """Calculates how similar the template appears within different locations of the input image by computing the normalized cross-correlation of the template and image. The resulting matrix C contains the correlation coefficients.
    
    Args:
        template (np.ndarray[np.float32, Any]): N-D array of template or filter used for cross-correlation. Length of each dimension must be less than length of image.
        image (np.ndarray[np.float32, Any]): input image
        mode (str, optional): _description_. Defaults to "full".

    Returns:
        np.ndarray[np.float32, Any]: _description_
    """    """

    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs. 
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """

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
