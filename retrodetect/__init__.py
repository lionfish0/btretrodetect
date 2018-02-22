from scipy.ndimage import interpolation
import numpy as np
from retrodetect.normxcorr2 import normxcorr2

def getshift(imgA,imgB,start=None,end=None,searchbox=100,step=8):
    """
    Line up part of imgA (specified by start and end) with imgB
    If start/end None, we clip 100 pixels from the edge.
    
    - Search just within shifts of a distance up to
                   the searchbox (default=100px)
    - Search in steps of step pixels (default = 4px)
    Returns shifted version of imgA.
    """
    if start is None:
        start = np.array([100,100])
    if end is None:
        end = np.array(imgA.shape)-100
    
    imgB = imgB[start[0]:end[0],start[1]:end[1]]
    imgApart = imgA[start[0]-searchbox:end[0]+searchbox,start[1]-searchbox:end[1]+searchbox]
    temp = normxcorr2(imgApart[::step,::step],imgB[::step,::step],mode='valid')
    shift = step*np.array(np.unravel_index(temp.argmax(), temp.shape))
    shift = shift - searchbox
    return shift

def getblockmaxedimage(img,blocksize=70,offset=2):
    """
    Effectively replaces each pixel with approximately the maximum of all the
    pixels within offset*blocksize of the pixel.
    
    Get a new image of the same size, but filtered such that each square patch
    of blocksize has its maximum calculated, then a search box of size
    (1+offset*2)*blocksize centred on each pixel is applied which finds the
    maximum of these patches.
    
    img = image to apply the filter to
    blocksize = size of the squares
    offset = how far from the pixel to look for maximum
    """
    blockcountx = 1+int(img.shape[0]/blocksize)
    blockcounty = 1+int(img.shape[1]/blocksize)

    maxes = np.empty([blockcountx,blockcounty])
    for x,blockx in enumerate(range(0,img.shape[0],blocksize)):
        for y,blocky in enumerate(range(0,img.shape[1],blocksize)):
            maxes[x,y] = np.max(img[blockx:blockx+blocksize,blocky:blocky+blocksize])

    templist = []
    xm,ym = maxes.shape
    for xoff in range(-offset,offset,1):
      for yoff in range(-offset,offset,1):
        templist.append(maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
    max_img = templist[0]
    for im in templist[1:]:
        max_img = np.maximum(max_img,im)

    out_img = np.ones_like(img)*255
    for x,blockx in enumerate(range(0,img.shape[0]-2*blocksize*offset,blocksize)):
        for y,blocky in enumerate(range(0,img.shape[1]-2*blocksize*offset,blocksize)):
            out_img[blockx+(blocksize*offset):blockx+blocksize+(blocksize*offset),blocky+(blocksize*offset):blocky+blocksize+(blocksize*offset)] = max_img[x,y]
    return out_img

def alignandsubtract(subimg,shift,foreimg,start=None,end=None):
    """
    Subtract subimg (after shifting) from foreimg, 
    removing a box defined by start and end (or
    100 around edge if not specified by start and end)"""
    if start is None:
        start = np.array([100,100])
    if end is None:
        end = np.array(subimg.shape)-100
        
    #TODO: REPLACE THIS WITH SOMETHING FASTER AND APPROXIMATE!
    subimgshifted = interpolation.shift(subimg[start[0]:end[0],start[1]:end[1]],shift[::-1],cval=255,prefilter=False,order=0)
    temp = foreimg.copy()[start[0]:end[0],start[1]:end[1]]
    temp-=subimgshifted
    return temp
    
def detect(flash,noflash):
    """
    Using defaults, run the algorithm on the two images
    """
    shift = getshift(flash,noflash)
    out_img = getblockmaxedimage(noflash)
    done = alignandsubtract(out_img,shift,flash)
    return done
