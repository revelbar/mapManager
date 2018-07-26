import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist



def get_NDVI(RED, NIR):
    NDVI = (NIR-RED)
    NDVI2 = NIR+RED
    ndvi = np.where(NDVI2 == 0, 0, NDVI/NDVI2)
    return ndvi


def stretch(img, ratio):
    """
    Stretches the image ratio times its initial size
    
    Parameters
    ----------
    img: numpy array
        has a shape (pixel height, pixel width, number of bands)
    ratio: int
        ratio between the sizes of the target image and the img
        
    Returns
    -------
    numpy array
        has a shape of (img.shape[0]*ratio, img.shape[1]*ratio[, img.shape[2]])
    """
    stretched = np.repeat(img, ratio, axis=1)
    stretched = np.repeat(stretched, ratio, axis=0)
    return stretched
def trimBG(img):
    edge = canny(rgb2gray(img))
    mask = ndi.binary_fill_holes(edge)

    return clip(img, mask, 1)

def clip(img, mask, ratio): #can be really slow?
    """
    Clips the image based on a mask
        
    Parameters
    ----------
    mask: numpy array
        2D array with all values are false/true. Will default to the set mask in the class instance if
        parameter is not indicated.
    ratio: int
        ratio btwn the resolution of the mask over the resolution of the img
        
    Returns
    -------
    numpy array
        float or int. The array of the cropped map
    numpy array
        boolean. A mask of the resulting map to cut out any unnecessary data/white background.
        Can be discarded if image formed by the clipping is perfectly orthogonal/rectangular
        with no white spaces
    """

    minx = None
    maxx = None
    miny = None
    maxy = None
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j]:
                miny = (i,j)
                break
        if (miny != None):
            break
    for j in range(mask.shape[1]):
        for i in range(mask.shape[0]):
            if mask[i][j]:
                minx = (i,j)
                break
        if (minx != None):
            break
    for i in reversed(range(mask.shape[0])):
        for j in reversed(range(mask.shape[1])):
            if mask[i][j]:
                maxy = (i,j)
                break
        if (maxy != None):
            break
    for j in reversed(range(mask.shape[1])):
        for i in reversed(range(mask.shape[0])):
            if mask[i][j]:
                maxx = (i,j)
                break
        if (maxx != None):
            break
    croppedMask = mask[miny[0]:maxy[0]+1,minx[1]:maxx[1]+1]
    cropped = img[int(ratio*miny[0]):int(ratio*(maxy[0]+1)),int(ratio*minx[1]):int(ratio*(maxx[1]+1))]
    #print("{}:{},{}:{}".format(miny[0],maxy[0]+1,minx[1],maxx[1]+1))
    croppedMask = stretch(croppedMask,ratio)
    if len(img.shape) > 2:
        croppedMask = np.repeat(croppedMask[:,:, np.newaxis], img.shape[2], axis=2)
        bandLen = img.shape[2]
    else:
        bandLen = 1
    if croppedMask.shape[0] != cropped.shape[0]:
        croppedMask = croppedMask[:cropped.shape[0], :]
    if croppedMask.shape[1] != cropped.shape[1]:
        croppedMask = croppedMask[:, :cropped.shape[1]]
    
    cropped = np.where(croppedMask, cropped, [255 for i in range(bandLen)])
    return (cropped, croppedMask)

def agg_pixels(img, mask = None):
    if len(img.shape) == 2:
        pixels = img.flatten()
    else:
        pixels = img.reshape(-1, img.shape[-1])
    if (type(mask) !=type(None)):
        if len(mask.shape) == 2:
            mask=mask.flatten()
        else:
            mask = mask[:,:,0].flatten()
        pixels = pixels[mask]
    return pixels

def inc_contrast(img,mask = None, plot =True):
    if type(mask) == np.ndarray:
        pixels = agg_pixels(img, mask)
        pixels_contrast = equalize_hist(pixels)
        img_contrast = array
        itr = iter(pixels_contrast)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i][j]:
                    img_contrast[i][j] = next(itr)
    else:
        img_contrast = equalize_hist(img)

    if plot:
        plt.figure(figsize=(20,20))
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(img_contrast)
    return img_contrast