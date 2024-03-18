import numpy as np

from retrodetect.image_processing.image_processing import ensemblegetshift, getblockmaxedimage, alignandsubtract
import numbers
import os


def detect(
        flash: np.array,
        noflash: np.array,
        blocksize: int = 2,
        offset: int = 3,
        searchbox: int = 20,
        step: int = 2,
        searchblocksize: int = 50,
        ensemblesizesqrt: int = 3,
        dilate: bool = True,
        margin: int = 10
) -> np.array:
    """
    Returns the differences detected between two images, flash and noflash.
    :param flash: The first image, assumed to be taken with flash.
    :param noflash: The second image, assumed to be taken without flash.
    :param blocksize: The size of blocks to use for processing the images for the getblockmaxedimage function.Defaults to 2.
    :param offset: The offset to apply when comparing blocks for the getblockmaxed function. Defaults to 3.
    :param searchbox: The maximum distance to search for the optimal shift in each direction within each sub-region for ensemblegetshift function. Defaults to 20.
    :param step: The step size used when searching for the optimal shift within each sub-region for ensemblegetshift function. Defaults to 2.
    :param searchblocksize: The size of the square sub-regions extracted from flash for alignment for ensemblegetshift function. Defaults to 50.
    :param ensemblesizesqrt: The square root of the number of sub-regions to be extracted from each dimension of flash for the ensemblegetshift function. Defaults to 3.
    :param dilate: Whether to perform dilation on the non-flash image before processing. Defaults to True.
    :param margin: The margin to use when aligning the images for the alignandsubtract function. Defaults to 10.
    :return: The difference image, highlighting areas of difference between the two input images.
    """
    shift = ensemblegetshift(flash, noflash, searchbox,
                             step, searchblocksize, ensemblesizesqrt)
    # print(shift)
    if dilate:
        noflash = getblockmaxedimage(noflash, blocksize, offset)
    done = alignandsubtract(noflash, shift, flash, margin=margin)
    return done


def detectcontact(
        photolist: list,
        n: int,
        savesize: int = 20,
        delsize: int = 15,
        thresholds: list = [9, 0.75, 6],
        historysize: int = 10,
        blocksize: int = 10,
        Npatches: int = 20
) -> tuple:
    """
    This function analyzes a sequence of photos captured by a tracking system to detect potential retroreflectors.
    :param photolist: A list of dictionaries representing captured photos. Each dictionary (`photoitem`) should have the following keys:
        - `record`: (dictionary) Contains metadata about the photo, including whether the photo is taken with flash (`flash`),
        end of a set (`endofset`), `flashselection`, `direction`, `index`, and information about the trigger time (`triggertime`, `triggertimestring`).
        - `img`: (numpy.ndarray) The image data as a 2D array.
        - `mean` (float, optional): The pre-computed mean value of the image (used for efficiency). If not provided, it will be calculated.

    :param n: The index within `photolist` to use as the current (flash) image for analysis.
    :param savesize: Controls the size (in pixels) of the square patch extracted around a candidate peak in the search image
     and to be saved into the output `patch` in the output 'contact'. Defaults to 20.
    :param delsize: Controls the size (in pixels) of the square patch that is deleted from the search image around the maximum
     Defaults to 15.
    :param thresholds: A list of three values used as thresholds for 'non-ML' decision boundary classifying a candidate
     peak as a potential retroreflector. Defaults to [9, 0.75, 6]. The elements correspond to:
        - `thresholds[0]`: Maximum intensity threshold in the search image.
        - `thresholds[1]`: Mean intensity threshold of the patch around the peak.
        - `thresholds[2]`: Maximum intensity threshold within the central region of the patch. Defaults to [9, 0.75, 6].
    :param historysize: The number of photos (including the current one) to consider from `photolist` during the analysis.
    Defaults to 10 photos back in time.
    :param blocksize: The amount of dilation applied to the current no-flash image before computing the difference with
    the flash image. Defaults to 10.
    :param Npatches: The number of patches to consider around potential peaks/maximum in the search image.
    Defaults to 20.
    :return: A tuple containing three elements:

        contact (list): A list of dictionaries, each representing a potential retroreflector candidate. Each dictionary (`contact_item`) has the following keys:
            - `x`: (int) The x-coordinate (pixel position) of the candidate peak in the search image.
            - `y`: (int) The y-coordinate (pixel position) of the candidate peak in the search image.
            - `patch`: (numpy.ndarray) The patch extracted from the current flash image around the candidate peak by minusing the current no-flash photo.
            - `searchpatch`: (numpy.ndarray) The patch extracted from the search image around the candidate peak, i.e.,
            the difference between current pairs and previous pairs of photos (variously dilated)
                               which is searched for its maximum values.
            - `mean`: (int) The mean intensity value of the patch from the current flash image.
            - `centre`: (int) The maximum intensity value within the central region of the patch from the current flash image.
            - `innersurround`: (int) The maximum intensity value in a ring-shaped region around the central area of the patch from the current flash image.
            - `outersurround`: (int) The maximum intensity value in the outer region of the patch from the current flash image.
            - `searchmax`: (int) The maximum intensity value in the searchpatch.
            - `centremax`: (int) The maximum intensity value within the central region of the searchpatch.
            - `confident`: (bool) A flag indicating whether the analysis is confident about the candidate being a retroreflector.
            - `prediction`: (float) A score reflecting the confidence level (closer to negative values indicates higher confidence).

        found (bool): A boolean value indicating whether a candidate with high confidence (`confident=True`) was found.

        searchimg (numpy.ndarray): The search image used for peak detection after processing and thresholding, useful for debugging.


    TODO Fix Bug: The code relies on the savesize = 20, as that places the peak at 20,20 in the patch.


    """
    from time import time
    unsortedsets = []
    startn = n - historysize
    if startn < 0:
        startn = 0
    for i in range(startn, n + 1):
        # photoitem = q.read(i)
        photoitem = photolist[i]
        if photoitem is None:
            continue
        if photoitem['record'] is None:
            continue
        if photoitem['img'] is None:
            continue
        # assert not isinstance(
        #    photoitem['img'][0, 0], numbers.Integral), "Need image array to be float not integers." #SC:check one of the number of the img array # I think we need to force it
        photoitem['img'] = photoitem['img'].astype(float)
        if 'mean' not in photoitem:
            photoitem['mean'] = np.mean(photoitem['img'][::5, ::5])
        # photoitem['img'] = photoitem['img'].astype(np.float) #already done
        tt = photoitem['record']['triggertime']
        chosenset = None  # chosenset is always NONE after the loop?
        for s in unsortedsets:  # SC: it is empty in each loop so I don't think it loops through
            if np.abs(tt - np.mean([photoi['record']['triggertime'] for photoi in s])) < 0.5:
                chosenset = s
        if chosenset is None:
            unsortedsets.append([photoitem])
        else:
            chosenset.append(photoitem)

    starttime = time()
    sets = []
    for s in unsortedsets:
        if len(s) < 2:  # if the set only has one photo in, skip.
            continue
        newset = {'flash': [], 'noflash': []}
        setmean = np.mean([photoitem['mean']
                           for photoitem in s if photoitem['img'] is not None])
        for photoitem in s:
            if photoitem['img'] is not None:
                if photoitem['mean'] > setmean + 0.1:
                    newset['flash'].append(photoitem)
                else:
                    newset['noflash'].append(photoitem)
        if len(newset['flash']) == 0:
            continue  # no point including sets without a flash
        sets.append(newset)

    starttime = time()
    last_diff = None
    this_diff = None
    if len(sets) < 2:
        print("Fewer than two photo sets available")  # warning msg
        return None, False, None  # we can't do this if we only have one photo set
    for i, s in enumerate(sets):
        # whether the set is the one that we're looking for the bee in.
        this_set = i == len(sets) - 1
        for s_nf in s['noflash']:
            if this_set:
                intertime = time()
                # for the current search image we dilate
                diff = detect(s['flash'][0]['img'],
                              s_nf['img'], blocksize=blocksize)
                if this_diff is None:
                    this_diff = diff
                else:
                    this_diff = np.minimum(diff, this_diff)
            else:
                intertime = time()
                if 'nodilationdiff' in s_nf:
                    diff = s_nf['nodilationdiff']
                else:
                    # for the past ones we don't
                    diff = detect(s['flash'][0]['img'],
                                  s_nf['img'], dilate=None)  # SC: dilate should be boolean. FALSE
                    if diff is not None:
                        s_nf['nodilationdiff'] = diff
                if last_diff is None:
                    last_diff = diff
                else:
                    # TODO: Need to align to other sets
                    last_diff = np.maximum(diff, last_diff)

    if (last_diff is None) or (this_diff is None):
        print("Insufficient data")
        return None, False, None

    starttime = time()
    # if there are large changes in the image the chances are the camera's moved... remove those sets before then
    keepafter = 0
    for i in range(len(sets) - 1):
        if np.mean(np.abs(sets[i]['noflash'][0]['img'][::5, ::5] - sets[-1]['noflash'][0]['img'][::5, ::5])) > 3:
            keepafter = i
    sets = sets[keepafter:]
    #    #we just align to the first of the old sets.
    imgcorrection = 20
    #    shift = ensemblegetshift(sets[-1]['noflash'][0]['img'],sets[0]['noflash'][0]['img'],searchbox=imgcorrection,step=2,searchblocksize=50,ensemblesizesqrt=3)
    #    #res = alignandsubtract(last_diff,shift,this_diff,margin=10)

    res = detect(this_diff, last_diff, blocksize=10,
                 offset=3, searchbox=imgcorrection)

    # get simple image difference to save as patch.
    img = sets[-1]['flash'][0]['img'] - sets[-1]['noflash'][0]['img']
    searchimg = res.copy()
    contact = []
    found = False
    print("---------------------------")
    for i in range(Npatches):
        y, x = np.unravel_index(searchimg.argmax(), searchimg.shape)
        searchmax = searchimg[y, x]

        # if (x<savesize) or (y<savesize) or (x>searchimg.shape[1]-savesize-1) or (y>searchimg.shape[0]-savesize-1): continue
        # target = 1*(((y-truey+alignmentcorrection)**2 + (x-truex+alignmentcorrection)**2)<10**2)
        # print(x,truex,y,truey)
        patch = img[y - savesize + imgcorrection:y + savesize + imgcorrection, x -
                                                                               savesize + imgcorrection:x + savesize + imgcorrection].astype(
            np.float32)
        searchpatch = searchimg[y - savesize:y + savesize,
                      x - savesize:x + savesize].astype(np.float32)
        searchimg[max(0, y - delsize):min(searchimg.shape[0], y + delsize),
        max(0, x - delsize):min(searchimg.shape[1], x + delsize)] = 0

        patimg = patch.copy()
        centreimg = patimg[17:24, 17:24].copy()
        patimg[37:44, 37:44] = 0

        centremax = np.max(centreimg.flatten())
        mean = np.mean(patimg.flatten())
        # Possible contact
        if (searchmax > thresholds[0]) & (mean < thresholds[1]) & (centremax > thresholds[2]):
            confident = True
        else:
            confident = False
        if confident:
            found = True

        outersurround = max(patch[16, 20], patch[20, 16], patch[24, 20], patch[20, 24],
                            patch[16, 16], patch[16, 24], patch[24, 16], patch[24, 24])
        innersurround = max(patch[18, 20], patch[20, 18], patch[22, 20], patch[20, 22],
                            patch[18, 18], patch[18, 22], patch[22, 18], patch[22, 22])
        centre = np.sum([patch[20, 20], patch[20, 21],
                         patch[20, 19], patch[19, 20], patch[21, 20]])
        res = np.array(
            [[searchmax, centremax, mean, outersurround, innersurround, centre]])
        pred = -4  # 250 - centremax
        # pred -= (centremax-200)/160
        # pred -= (searchmax-200)/60
        pred += 50 / (1 + searchmax)  # 50->+1
        pred += 50 / (1 + centremax)
        # if centremax>250: pred-=2
        # if searchmax>70: pred-=2
        # pred -= min((centremax/innersurround)/30,4)
        # pred -= (centremax/outersurround)/60
        pred += 20 * innersurround / centremax
        pred += 20 * outersurround / centremax
        pred += mean / 10  # not that helpful
        pred += outersurround / 20
        pred = [[pred]]
        print(pred, [x, y], [searchmax, centremax, mean,
                             outersurround, innersurround, centre])

        contact.append({'x': x + imgcorrection, 'y': y + imgcorrection, 'patch': patch, 'searchpatch': searchpatch,
                        'mean': int(mean), 'centre': int(centre), 'innersurround': int(
                innersurround), 'outersurround': int(outersurround), 'searchmax': int(searchmax),
                        'centremax': int(centremax), 'confident': confident, 'prediction': pred[0][0]})
    return contact, found, searchimg
