import numpy as np

def getblockmaxedimage(img,blocksize, offset):
    """
    This is a fast approximate dilation method (could probably replace with
    a true dilation now the pi5 is being used).
    
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

    k = int(img.shape[0] / blocksize)
    l = int(img.shape[1] / blocksize)
    if blocksize==1:
        maxes = img
    else:
        maxes = img[:k*blocksize,:l*blocksize].reshape(k,blocksize,l,blocksize).max(axis=(-1,-3)) #from https://stackoverflow.com/questions/18645013/windowed-maximum-in-numpy
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

    out_img = np.full_like(img,0) 
    inner_img = max_img.repeat(blocksize,axis=0).repeat(blocksize,axis=1)
    out_img[blocksize*offset:(blocksize*offset+inner_img.shape[0]),blocksize*offset:(blocksize*offset+inner_img.shape[1])] = inner_img
    return out_img
    
    
class Retrodetect:
    def __init__(self,Nmaxdilations=5):
        self.max_dilated_img = None
        self.idx = 0
        self.Nmaxdilations = Nmaxdilations
        self.runningavg = 0 #used to remove no-flash photos!
        
    def find_tags(self,diff,img):
        results = []
        savesize = 20
        delsize = 15


        np.set_printoptions(precision=2,suppress=True)
        for Npatches in range(25):
            y,x = np.unravel_index(diff.argmax(), diff.shape)
            
                
            diff_max = diff[y,x]
            img_patch = img[y-savesize:y+savesize,x-savesize:x+savesize].astype(np.float32).copy()
            temp = img_patch.copy()
            diff_patch = diff[y-savesize:y+savesize,x-savesize:x+savesize].copy().astype(np.float32)        
            diff[max(0,y-delsize):min(diff.shape[0],y+delsize),max(0,x-delsize):min(diff.shape[1],x+delsize)]=-5000

            if (x<=20) or (x>=diff.shape[1]-20) or (y<=20) or (y>=diff.shape[0]-20):
                res=np.array([-100,x,y,0,0,0,0,0,0,0,diff_patch,temp])
                results.append(res)
                continue
                
            centreimg = img_patch[17:24,17:24].copy()
            #img_patch[37:44,37:44]=0
            centre_max=np.max(centreimg.flatten())
            bg_mean=np.mean(img_patch.flatten())
            outersurround_mean = np.sum(img_patch[[16,20,24,20,16,16,24,24],[20,16,20,24,16,24,16,24]])
            outersurround_max = np.max(img_patch[[16,20,24,20,16,16,24,24],[20,16,20,24,16,24,16,24]])
            innersurround_max = np.max(img_patch[[18,20,22,20,18,18,22,22],[20,18,20,22,18,22,18,22]])
            
            #not used?
            centre_sum = np.sum(img_patch[[20,20,20,19,21],[20,21,19,20,20]])
            pred = -10
            #pred-= 250/(1+diff_max) #value at max in difference image
            pred+= centre_sum/4
            pred+= diff_max/2 #?
            if diff_max<10:
                pred+=diff_max-10
            #pred-= 250/(1+centre_max) #maximum in centre of tag zone in normal image
            pred-= 30*innersurround_max/centre_max #ratio of the max value in the innersurrounding pixels and the maximum
            #pred-= innersurround_max/5
            pred-= 400*outersurround_max/centre_max
            pred-= 10*outersurround_mean
            #pred-=outersurround_max #how bright the outer surrounding pixels are

            res=np.array([pred,x,y,diff_max,centre_max,bg_mean,outersurround_max,innersurround_max,centre_sum,outersurround_mean,diff_patch,temp])
            results.append(res)
        return results
        
    def process_photo(self,photo_item,idx=None,skipNoFlash=True):
        """
        Processes a photo object.
         - normalise
         - compute the dilated image
         - compute diff image from the max dilated image of last self.Nmaxdilations
        
        """
        if idx is not None: self.idx = idx
        if self.max_dilated_img is None:
            try:
                imgshape = list(photo_item['img'].shape)
            except AttributeError:
                return []
            imgshape.append(self.Nmaxdilations)
            self.max_dilated_img = np.full(imgshape,5000)
            
        #convert to float
        try:
            img = photo_item['img'].astype(np.float32)
        except AttributeError:
            return []
            
        #remove no-flash photos...
        if skipNoFlash:
            avg = np.mean(img)
            if avg<self.runningavg*0.1: #if we're less than 10% of the average... probably a no-flash photo
                print("|NF",end="")
                return []
            self.runningavg = self.runningavg*0.7 + avg*0.3 #rolling average (handles gradual changes in daylight?)
        
        #normalise photo
        #compute the difference photo...
        diff_img = img-np.max(self.max_dilated_img,2)
        
        dilated_img = getblockmaxedimage(img,3,3)
        self.max_dilated_img[:,:,self.idx % self.Nmaxdilations] = dilated_img
        self.idx+=1 #this just is an index to acess the max_dilate_img array
        
        return self.find_tags(diff_img,img)
        
