import numpy as np
from retrodetect.normxcorr2 import normxcorr2

def shiftimg(test,shift,cval):
    new = np.full_like(test,cval)
    if shift[0]>0:
        if shift[1]>0:
            new[shift[0]:,shift[1]:] = test[:-shift[0],:-shift[1]]
        if shift[1]<0:
            new[shift[0]:,:shift[1]] = test[:-shift[0],-shift[1]:]
        if shift[1]==0:
            new[shift[0]:,:] = test[:-shift[0],:]
    if shift[0]<0:
        if shift[1]>0:
            new[:shift[0],shift[1]:] = test[-shift[0]:,:-shift[1]]
        if shift[1]<0:
            new[:shift[0],:shift[1]] = test[-shift[0]:,-shift[1]:]
        if shift[1]==0:
            new[:shift[0],:] = test[-shift[0]:,:]
    if shift[0]==0:
        if shift[1]>0:
            new[:,shift[1]:] = test[:,:-shift[1]]
        if shift[1]<0:
            new[:,:shift[1]] = test[:,-shift[1]:]
        if shift[1]==0:
            new[:,:] = test[:,:]
    return new
    
def getshift(imgA,imgB,start=None,end=None,searchbox=100,step=8):
    """
    Line up part of imgA (specified by start and end) with imgB
    If start/end None, we clip 100 pixels from the edge.
    
    - Search just within shifts of a distance up to
                   the searchbox (default=100px)
    - Search in steps of step pixels (default = 4px)
    Returns amount imgA is to be shifted
    """
    if start is None:
        start = np.array([searchbox,searchbox])
    if end is None:
        end = np.array(imgA.shape)-searchbox
    
    imgB = imgB[start[0]:end[0],start[1]:end[1]]
    imgApart = imgA[start[0]-searchbox:end[0]+searchbox,start[1]-searchbox:end[1]+searchbox]
    temp = normxcorr2(imgB[::step,::step],imgApart[::step,::step],mode='valid')
    shift = step*np.array(np.unravel_index(temp.argmax(), temp.shape))
    shift = shift - searchbox
    return shift
    
def ensemblegetshift(imgA,imgB,searchbox=100,step=8,searchblocksize=50,ensemblesizesqrt=3):
    """
    searchblock: how big each search image pair should be.
    ensemblesizesqrt: number of items for ensemble for one dimension.
    
    """
    starts = []
    for x in np.linspace(0,imgA.shape[0],ensemblesizesqrt+2)[1:-1].astype(int):
        for y in np.linspace(0,imgA.shape[1],ensemblesizesqrt+2)[1:-1].astype(int):
            starts.append([x,y])
            
    shifts = np.zeros([len(starts),2])
    for i,start in enumerate(starts):
        shifts[i] = getshift(imgA,imgB,step=step,searchbox=searchbox,start=start,end=start+np.array([searchblocksize,searchblocksize]))
    return np.median(shifts,0)    

def old_getblockmaxedimage(img,blocksize=70,offset=2):
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
    for xoff in range(-offset+1,offset,1): #(if offset=1, for xoff in [0]) (if offset=2, for xoff in [-1,0,1])...
      for yoff in range(-offset+1,offset,1):
        templist.append(maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
    max_img = templist[0]
    for im in templist[1:]:
        max_img = np.maximum(max_img,im)

    out_img = np.ones_like(img)*255
    for x,blockx in enumerate(range(0,img.shape[0]-2*blocksize*offset,blocksize)):
        for y,blocky in enumerate(range(0,img.shape[1]-2*blocksize*offset,blocksize)):
            out_img[blockx+(blocksize*offset):blockx+blocksize+(blocksize*offset),blocky+(blocksize*offset):blocky+blocksize+(blocksize*offset)] = max_img[x,y]
    return out_img
    
def getblockmaxedimage(img,blocksize, offset):
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
    k = int(arr.shape[0] / blocksize)
    l = int(arr.shape[1] / blocksize)
    maxes = arr.reshape(k,blocksize,l,blocksize).max(axis=(-1,-3)) #from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
    
    templist = []
    xm,ym = maxes.shape
    for xoff in range(-offset+1,offset,1): #(if offset=1, for xoff in [0]) (if offset=2, for xoff in [-1,0,1])...
      for yoff in range(-offset+1,offset,1):
        templist.append(maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
    max_img = templist[0]
    for im in templist[1:]:
        max_img = np.maximum(max_img,im)
    
    out_img = np.empty_like(img)
    new_inner_img = max_img.repeat(blocksize,axis=0).repeat(blocksize,axis=1)
    out_img[blocksize*offset:-(blocksize*offset),blocksize*offset:-(blocksize*offset)] = new_inner_img
    out_img[:blocksize*offset,:] = 255
    out_img[-(blocksize*offset):,:] = 255
    out_img[:,:blocksize*offset] = 255
    out_img[:,-(blocksize*offset):] = 255
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
        
    subimgshifted = shiftimg(subimg[start[0]:end[0],start[1]:end[1]],shift,cval=255)
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
