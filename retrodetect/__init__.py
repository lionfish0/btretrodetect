import numpy as np
from retrodetect.normxcorr2 import normxcorr2
import QueueBuffer as QB


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
    medianshift = np.median(shifts,0)
    medianshift = [int(medianshift[0]),int(medianshift[1])]
    return medianshift

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
    #import time
    #times = []
    #times.append(time.time())
    k = int(img.shape[0] / blocksize)
    l = int(img.shape[1] / blocksize)
    if blocksize==1:
        maxes = img
    else:
        maxes = img[:k*blocksize,:l*blocksize].reshape(k,blocksize,l,blocksize).max(axis=(-1,-3)) #from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
    #times.append(time.time())
    templist = []
    xm,ym = maxes.shape
    i = 0
    for xoff in range(-offset+1,offset,1): #(if offset=1, for xoff in [0]) (if offset=2, for xoff in [-1,0,1])...
      for yoff in range(-offset+1,offset,1):
        if i==0:
          max_img = maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset]
        else:
          max_img = np.maximum(max_img,maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
        i+=1
        #templist.append(maxes[xoff+offset:xoff+xm-offset,yoff+offset:yoff+ym-offset])
    #times.append(time.time())
    #max_img = templist[0]
    #for im in templist[1:]:
    #    max_img = np.maximum(max_img,im)
    #times.append(time.time())
    out_img = np.full_like(img,255)
    #times.append(time.time())
    inner_img = max_img.repeat(blocksize,axis=0).repeat(blocksize,axis=1)
    #times.append(time.time())
    #s = (out_img.shape-new_inner_img.shape)/2
    out_img[blocksize*offset:(blocksize*offset+inner_img.shape[0]),blocksize*offset:(blocksize*offset+inner_img.shape[1])] = inner_img
    #times.append(time.time())
    #print("----------")
    #print(np.diff(times))
#    out_img[:blocksize*offset,:] = 255
#    out_img[-(blocksize*offset):,:] = 255
#    out_img[:,:blocksize*offset] = 255
#    out_img[:,-(blocksize*offset):] = 255
    return out_img

def alignandsubtract(subimg,shift,foreimg,start=None,end=None,margin=100):
    """
    Subtract subimg (after shifting) from foreimg, 
    removing a box defined by start and end (or
    margin (default=100) around edge if not specified by start and end)"""
    if start is None:
        start = np.array([margin,margin])
    if end is None:
        end = np.array(subimg.shape)-np.array([margin,margin])
        
    subimgshifted = shiftimg(subimg[start[0]:end[0],start[1]:end[1]],shift,cval=255)
    temp = foreimg.copy()[start[0]:end[0],start[1]:end[1]]
    temp-=subimgshifted
    return temp
    
def detect(flash,noflash,blocksize=2,offset=3,searchbox=20,step=2,searchblocksize=50,ensemblesizesqrt=3,dilate=True,margin=10):
    """
    Using defaults, run the algorithm on the two images
    """
    shift = ensemblegetshift(flash,noflash,searchbox,step,searchblocksize,ensemblesizesqrt)
    #print(shift)
    if dilate: noflash = getblockmaxedimage(noflash,blocksize,offset)
    done = alignandsubtract(noflash,shift,flash,margin=margin)
    return done
    
def detectcontact(q,n,savesize = 20,delsize=15,thresholds = [10,0.75],historysize = 20,blocksize = 10):
    unsortedsets = []
    for i in range(n-historysize,n+1):
        photoitem = q.read(i)
        if photoitem is None: continue
        if photoitem[1] is None: continue
        photoitem[1] = photoitem[1].astype(np.float)
        tt = photoitem[2]['triggertime']
        chosenset = None
        for s in unsortedsets:
            if np.abs(tt-np.mean([photoi[2]['triggertime'] for photoi in s]))<0.5:
                chosenset = s
        if chosenset is None: 
            unsortedsets.append([photoitem])
        else:
            chosenset.append(photoitem)

    sets = []
    for s in unsortedsets:
        if len(s)<2: #if the set only has one photo in, skip.
            continue
        newset = {'flash':[],'noflash':[]}
        setmean = np.mean([np.mean(photoimg) for i,photoimg,data in s if photoimg is not None])
        for photoitem in s:
            if photoitem[1] is not None:
                if np.mean(photoitem[1])>setmean+0.1:
                    newset['flash'].append(photoitem)
                else:
                    newset['noflash'].append(photoitem)
        if len(newset['flash'])==0: continue #no point including sets without a flash
        sets.append(newset)

    last_diff = None
    this_diff = None
    for i,s in enumerate(sets):
        this_set = i==len(sets)-1 #whether the set is the one that we're looking for the bee in.
        for s_nf in s['noflash']:
            if this_set: 
                diff = detect(s['flash'][0][1],s_nf[1],blocksize=blocksize) #for the current search image we dilate
                if this_diff is None:
                    this_diff = diff
                else:
                    this_diff = np.minimum(diff,this_diff) 
            else: 
                diff = detect(s['flash'][0][1],s_nf[1],dilate=None) #for the past ones we don't
                if last_diff is None:
                    last_diff = diff
                else:
                    last_diff = np.maximum(diff,last_diff) #TODO: Need to align to other sets

    #we just align to the first of the old sets.
    imgcorrection = 20
    shift = ensemblegetshift(sets[-1]['noflash'][0][1],sets[0]['noflash'][0][1],searchbox=imgcorrection,step=2,searchblocksize=50,ensemblesizesqrt=3)
    res = alignandsubtract(last_diff,shift,this_diff,margin=10)

    #get simple image difference to save as patch.
    img = sets[-1]['flash'][0][1]-sets[-1]['noflash'][0][1]
    searchimg = res.copy()
    contact = None
    for i in range(10):
        y,x = np.unravel_index(searchimg.argmax(), searchimg.shape)
        searchmax = searchimg[y,x]

        #if (x<savesize) or (y<savesize) or (x>searchimg.shape[1]-savesize-1) or (y>searchimg.shape[0]-savesize-1): continue
        #target = 1*(((y-truey+alignmentcorrection)**2 + (x-truex+alignmentcorrection)**2)<10**2)
        #print(x,truex,y,truey)
        patch = img[y-savesize+imgcorrection:y+savesize+imgcorrection,x-savesize+imgcorrection:x+savesize+imgcorrection].astype(np.float32)
        searchimg[y-delsize:y+delsize,x-delsize:x+delsize]=0
        #patches.append({'patch':patch,'max':searchmax,'x':x,'y':y})
        patch
        searchmax

        patimg = patch.copy()
        centreimg = patimg[17:24,17:24].copy()
        patimg[37:44,37:44]=0

        centremax=np.max(centreimg.flatten())
        #centremean=np.mean(centreimg.flatten())
        mean=np.mean(patimg.flatten())
        #median=np.median(patimg.flatten())
        #maxp=np.max(patimg.flatten())
        #minp=np.min(patimg.flatten())
        ##print(searchmax,mean,centremax)
        #Possible contact
        if (searchmax>thresholds[0]) & (mean<thresholds[1]) & (centremax>7):
            contact = {'x':x+imgcorrection,'y':y+imgcorrection,'patch':patch,'mean':mean,'searchmax':searchmax}
    return contact
