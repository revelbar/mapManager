import matplotlib.pyplot as plt
from skimage.external import tifffile
from .imageData import *
from .segmentation import *
from .file_manager import *
import numpy as np
from skimage import img_as_ubyte
from matplotlib.pyplot import cm
from skimage.exposure import equalize_hist
import warnings
from copy import deepcopy
import os
import pickle


class Image(object):
    """
    A class that streamlines the reading and accessing of image files from your working directory.

    Attributes
    ----------
    data: numpy ndarray
        the image array to be checked

    Methods
    -------
    info: (N/A)
        gets the image dimensions, max, min, and datatype of the image array.
    skg_changedtype: (skimage function)
        change the datatype of the array using skimage's functions (highly recommended)
    np_changedtype: (str)
        change the datatype of the array using numpy's astype functions (don't do this as your first choice)
    display: (int)
        plots the image using matplotlib.pyplot
    make_map: (str, float)
        makes a Grayscale/Multispectral class instance of the data
    """
    def __init__(self, file_path):
        """
        Constructor of Image

        Parameters
        ----------
        file_path: str
            indicates the file name or the path to the file you want to open
        """
        self.data = rdr.imread(file_path)
    def info(self):
        """
        Gets the image dimensions, max, min, and datatype of the image array. Use this to check if the datatype of the array matches its range of values

        Ex. if your max and min is 0 to 255 and your datatype is set to be uint16, unless you really know it's supposed to be uint16, you should probably
        change the datatype to 'uint8' to make calculations accurate.
        """
        print("Image size of {} by {}, with {} bands".format(*self.data.shape)) # (x,y,z)
        print("Maximum value: {} \nMinimum value: {}".format(self.data.max(), self.data.min()))
        print("Default dtype: {}".format(self.data.dtype))
    def np_changedtype(self, dtype):
        """
        Changes the datatype of the array using numpy's astype functions (don't do this as your first choice)

        Parameters
        ----------
        dtype: str
            the datatype you want your image to be. Refer to documentation of np.astype function to see all the options

        Returns
        -------
        None
        """
        self.data = self.data.astype(dtype)
    def skg_changedtype(self, func):
        """
        Changes the datatype of the array using skimage's functions (highly recommended)

        Parameters
        ----------
        func: function
            name of the skimage function you want to apply your image with. Check the documentation of skimage to see all the options

        Returns
        -------
        None
        """
        self.data = func(self.data)
    def display(self, *args):
        """
        Plots the image using matplotlib.pyplot

        Parameters
        ----------
        *args: int
            list of bands you want to plug into the RGB channel
            if none is given, function will plot all channels

        Returns
        -------
        None
        """
        if len(args)==3 or len(args) ==1:
            plt.imshow(self.data[:,:,[*args]])
        elif len(args) == 0:
            plt.imshow(self.data)
        else:
            raise IndexError("Function only accepts 3 or 1 bands!")
    def make_map(self, *args):
        if len(self.data.shape) > 2:
            return Multispectral(self.data,*args)
        else:
            return Grayscale(self.data,*args)

class Super_Map(object):
    """
    Base class of all the maps the module will handle
        
    Attributes
    ----------
    _img: numpy ndarray
        array that represents the pixel value of an image
    name: str
        designated name of this map (for graphs, summaries, and saves)
    resolution: float
        meters per pixel
    _segments: dict
        contains the class instances of Segments for the map to call on
    _mask: numpy ndarray
        2D array that determines which pixels are to be excluded from calculations
        NOTE: this must have the same dimensions of pixels as _img
    
    Methods
    -------
    array (property): numpy ndarray
        gives a copy of the image array
    segments (property): dict
        gives a copy of the segments
    array_calc (property): numpy ndarray
        gives a flattened array of pixel values, mask-tagged pixels excluded. Used for calculations
    copy: Grayscale/Multispectral 
        gives a copy of the object
    set_mask
    set_segments
    _get_segments
    _clip_segment
    clip_mask
    save
    fit_segments
    calc_bySegment
    
    
    """

    def __init__(self,img, name, resolution = None):
        """
        Constructor
        
        Parameters
        ----------
        img: numpy ndarray
            has a shape (pixel height, pixel width, number of bands)
        name: str 
            designated name of this map (for graphs, summaries, and saves)
        resolution: float (optional)
            indicated how many meters per pixel. If there is no indicated, it defaults to None
            
        Returns
        -------
        Class instance of Super_Map
        """
        
        self._img = img.copy()
        self.name = name
        self.resolution = resolution
        self._segments = {}
        self._mask = None

    
    @property
    def array(self):
        """
        Returns
        -------
        numpy array 
            stored in the class
        """
        return self._img.copy()

    @property
    def segments(self):
        """
        Gives a list of segmentations registered under the class instance

        Returns
        -------
        dict.key()
            list of registered segmentations
        """
        return self._segments.copy()

    @property
    def array_calc(self):
        return agg_pixels(self._img, mask = self._mask)

    def copy(self):
        """
        Creates a deep copy of the class instance

        Returns
        -------
        Super_Map, Grayscale, Multispectral
            Class instance of Super_Map or any of its subclasses
        """

        map_copy = deepcopy(self)
        return map_copy
    def set_mask(self, mask):
        """
        Manually sets the default mask of the map. Tags the background or unnecessary pixels in the map.
        
        Parameters
        ----------
        mask: numpy array
            2D array with all values are false/true. Indicates which pixels will be included in histograms, 
            statistics, etc. If you want to remove the mask, use : .setMask(None)
        
        Returns
        -------
        None
        """
        self._mask = mask
    def set_segments(self, segments, name):
        """
        Manually sets up a segmentation array for the class instance to access.

        Parameters
        ----------
        segments: Segment
            an instance of Segment. Has to have the same size as the image.
        name: str
            Name of the segment

        Returns
        -------
        None
        """
        if segments.__class__.__name__ != Segments.__name__:
            raise ValueError("Invalid data type! Segmentation registered must be a class instance of Segments")
        self._segments[name] = segments   
    def _get_segments(self, segments):
        """
        Private function for the class to help in retrieving and choosing segmentations for class functions
        
        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used

        Returns
        -------
        Segment
            user-selected segmentation
        """
        if type(segments) == str or type(segments) == list:
            segmentation = self._segments[segments]
        elif segments.__class__.__name__ == Segments.__name__:
            segmentation =segments
        else:
            raise ValueError('Invalid data type! Please input a class instance of Segments or a str indicating a set segmentation in the class instance')
        return segmentation
    def _clip_segment(self, segments,seg_id):
        """
        A private function that clips the image based on a segmentation and makes a new instance of a subclass of the Super_Map depending on the self's current class.
        If no segmentation is specified, it will try to default to a pre-existing segmentation stored in the instance.
        
        Parameters
        ----------
        segments: Segment, str
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        seg_id: int, str, or list
            key that will access the segments being called
        
        Returns
        -------
        Grayscale or Multispectral
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        segmentation = self.fit_segments(segments)
        mask = segmentation[seg_id]
        newMap = self.clip_mask(int(segmentation.resolution/self.resolution), mask)
        return newMap
     
    def clip_mask(self, ratio, mask = None): #can be really slow?
        """
        Clips the image based on a mask and makes a new instance of a subclass of the Super_Map depending on the self's current class
        
        Parameters
        ----------
        ratio: int
            ratio btwn the resolution of the mask over the resolution of the img
        mask: numpy array (optional)
            2D array with all values are false/true. Will default to the set mask in the class instance if
            parameter is not indicated.
        
        
        Returns
        -------
        Grayscale or Multispectral
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        if (type(mask) != np.ndarray) and (type(self._mask) != np.ndarray):
            raise ValueError('No mask available to trim image. Please add a mask to this function\'s argument or use .set_mask() function')
        elif (type(mask) == np.ndarray):
            pass
        else:
            mask = self._mask
        result= clip(self._img, mask, ratio)#imd.clip(self._img, mask, ratio)

        if self.__class__.__name__ == Multispectral.__name__:
            newMap = Multispectral(result[0], self.name + "_clipped", resolution = self.resolution) 
        elif self.__class__.__name__ == Grayscale.__name__:
            newMap = Grayscale(result[0], self.name + "_clipped", resolution = self.resolution)
        newMap.set_mask(result[1])
        return newMap
    def save(self, project_name=None, overwrite=False):
        """
        Saves the data stored in a class instance for future use

        Data is placed in a subdirectory inside the folder whose name is in var save_folder. The folder name for the map is the same as the name of the class instance

        Directory:

        \working_directory
            \saved_data
                \project_name (if applicable)
                    \map
                        metadata.pkl
                        _img.pkl
                        _mask.pkl (if applicable)
                        \segments (if applicable)
                            \segment1
                                - data
                            \segment2
                                - data
                \map
                    metadata.pkl
                    _img.pkl
                    _mask.pkl (if applicable)
                    \segments (if applicable)
                        \segment1
                            - data
                        \segment2
                            - data
             -script1.py

        Parameters
        ----------
        project_name: str (optional)
            name of the folder inside the save_folder you want to save the map in
        overwrite: boolean (default = False)
            if False, will not overwrite a folder whose name is the same as itself
            if True, will overwrite an same-named folder

        Returns
        -------
        str
            name of the folder holding the data

        """
        if type(project_name) == str:
            new_save = os.path.join(save_folder, project_name)
        else:
            new_save = save_folder
        if overwrite == False and os.path.exists(os.path.join(new_save, self.name)):
            i = 1
            new_path = self.name + "({})".format(i)
            while os.path.exists(os.path.join(new_save,new_path)):
                i+=1
                new_path = self.name + "({})".format(i)
            path = new_path
        else:
            path = self.name
        check_folder(os.path.join(new_save,path))
        save(self._img, "_img", folder = path, overwrite = True)
        if type(self._mask) != type(None):
            save(self._mask, "_mask", folder = path, overwrite = True)
        metadata = {'name': self.name, 'resolution': self.resolution}
        save(metadata, "metadata",folder = path, overwrite = True)

        if len(self._segments) > 0:
            for i in self._segments:
                seg_path = os.path.join(path, "segments")
                self._segments[i].save(folder=seg_path, series_name = i, overwrite=True)
        return path
    def apply_index(self, func, *args):
        """
        Calculates a value for each pixel in the image from the values of the available bands. 

        If your function will take the whole array, don't put in any arguments.
        
        Parameters
        ----------
        func: declared function
            must return an array
        *args: int
            band numbers to be used by the function
            
        Returns
        -------
        numpy ndarray
            the result of the function's calculation
        """
        if len(args) >0:
            new_param = [self[i].astype(float) for i in args]
            return func(*new_param)
        else:
            return func(self.array)
    def fit_segments(self, segments):
        """
        Will stretch the segment's array until it's the same size as that of the image

        Constraints:
        1. assumes that segments' resolution is bigger than that of the image
        2. can only accept ratio intergers

        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used

        Returns
        -------
        Segments
            a new segment whose size is the same as the
        """
        segmentation = self._get_segments(segments)
        try:
            ratio = int(segmentation.resolution/self.resolution)
        except:
            ratio = 1
        if ratio != 1:
            seg_array = stretch(segmentation.array, ratio)[:self._img.shape[0],:self._img.shape[1]]
            results = segmentation.copy()
            results._img = seg_array
            results.resolution = segmentation.resolution/ratio
            return results
        else:
            return segmentation

    def calc_bySegment(self, segments, func, *args, seg_id=None, isimg=True):
        """
        Will label each segment by the function's result.

        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
        func: function 
            should return an integer or boolean
        *args: int (optional)
            band number you are going to use for your calculate
        seg_id: int, list, str (optional)
            identifier of the segment in the segmentation
        isimg: boolean (default =True)
            if True, will return an image array with each segment colored with their resulting value
            if False, will return a table of values
        Returns
        -------
        numpy array, dict
            result of your function for each segment
        """

        
            
        seg = self.fit_segments(segments)
        if isimg:
            def _calc_pixels(mask, array):
                result = calc_pixels(self.array, func, *args, mask = mask)
                return np.where(mask, result,array)
            array = np.zeros_like(seg.array)
            if type(seg_id) == type(None):
                for i in seg:
                    array = _calc_pixels(i, array)
            elif type(seg_id) == int:
                array = _calc_pixels(seg[i], array)
            else:
                if type(seg_id) == list:
                    keys = seg_id
                elif type(seg_id) == str:
                    keys = seg.group(seg_id)
                for i in keys:
                    array = _calc_pixels(seg[i], array)
            return array
        else:
            dict_values= {}
            if type(seg_id) == type(None):
                keys = iter(seg.id)
                for i in seg:
                    dict_values[next(keys)] = calc_pixels(self.array, func, *args, mask = i)
            elif type(seg_id) == int:
                dict_values[seg_id] = calc_pixels(self.array, func, *args, mask = seg[i])
            else:
                if type(seg_id) == list:
                    keys = seg_id
                elif type(seg_id) == str:
                    keys = seg.group(seg_id)
                for i in keys:
                    dict_values[i] = calc_pixels(self.array, func, *args, mask = seg[i])
            return dict_values

    ####################### MAGIC FUNCTIONS ###########################
    def __repr__(self):
        return "{} (numpy array of {}, resolution: {})".format(type(self), self._img.shape, self.resolution)
    def __str__(self):
        if type(self._mask) != type(None):
            return "Map Name: {} \nShape (h px, w px[, bands]): {} \nResolution: {} m/px \nDataType: {} \nMask of {} set.".format(self.name, self._img.shape, self.resolution, self._img.dtype, self._mask.shape)
        else:
            return "Map Name: {} \nShape (h px, w px[, bands]): {} \nResolution: {} m/px \nDataType: {}".format(self.name, self._img.shape, self.resolution, self._img.dtype)

class Multispectral(Super_Map):
    """
    A class for maps with more than 1 band (ideally at least 3)
        
    A subclass of Super_Map that accepts only 3D arrays. Used as a container for multispectral 
    maps. The first 3 bands will, by default, be labelled and treated as the RGB bands.
    
    Attributes
    ----------
    _img: numpy ndarray
        array that represents the pixel value of an image
    name: str
        designated name of this map (for graphs, summaries, and saves)
    resolution: float
        meters per pixel
    _segments: dict
        contains the class instances of Segments for the map to call on
    _mask: numpy ndarray
        2D array that determines which pixels are to be excluded from calculations
        NOTE: this must have the same dimensions of pixels as _img
    band_names: dict
        contains the name of each band, with the key being the band number and the value being the band name itself
        
    Methods
    -------
    TBC
        
    
    """
    def __init__(self, img, name, resolution = None):
        """
        Constructor
        
        Parameters
        ----------
        img: numpy array
            has a shape (pixel height, pixel width, number of bands)
        name: str 
            designated name of this map (for graphs and summaries)
        resolution: float (optional)
            indicated how many meters per pixel. If there is no indicated, it defaults to None

        Returns
        -------
        Class instance of Multispectral
        
        """
    
        if len(img.shape) < 3:
            raise ValueError("Image only has one band!")
        super().__init__(img, name, resolution)
        self.band_names = {}
        self.band_names[1] = 'Red'
        self.band_names[2] = "Green"
        if img.shape[2] > 2:
            self.band_names[3] = "Blue"
    def _call_channels(self, *args):
        """
        Calls the bands and assigns them to a channel.

        This is a private function to reduce redundancies in the class functions.

        Parameters
        ----------
        *args: int (optional)
            indicates which band you want to plug in in the red, green, and blue channels.
            if only one number is given, only 1 band is displayed.
            if no number is given, it will automatically show bands 1, 2, and 3.

        Returns
        -------
        list
            band numbers to be plugged into the function, listed in the order of the RGB channel.
        list
            name of each band
        """
        if len(args) == 1:
            array = self[args[0]]
            try:
                band_names = [self.band_names[args[0]]]
            except:
                band_names = ["Band {}".format(args[0])]
            
        else:
            if len(args) == 0:
                bands = [1,2,3]
            elif len(args) == 3:
                bands=[*args]
            else:
                raise IndexError('Function only accepts 1 or 3 bands for displaying!')
            array = self._img[:,:,[i-1 for i in bands]]
            band_names = []
            for i in bands:
                try:
                    band_names.append(self.band_names[i])
                except:
                    band_names.append("Band {}".format(i))
        return array, band_names
    def display(self, *args):
        """
        Shows a plot of the map.
        
        Parameters
        ----------
        *args: int (optional)
            indicates which band you want to plug in in the red, green, and blue channels.
            if only one number is given, only 1 band is displayed.
            if no number is given, it will automatically show bands 1, 2, and 3.

        Returns
        -------
        None
        """
        array, band_names = self._call_channels(*args)
        if len(args) == 1:
            plt.title("{}: {}".format(self.name, *band_names))
            plt.imshow(array, cmap = 'gray')
        else:
            plt.title("{}: {}, {}, and {}".format(self.name, *band_names))
            plt.imshow(array)
    def name_band(self, bandN, band_name):
        """
        Names an available band in the map by a name given by the user.
        
        Parameters
        ----------
        bandN: int
            band number of the band you want to name
            NOTE: when dealing with the array, bandN-1 (ex. band1 in the array is the 0th slice)
        band_name: str 
            name of the band number
            
        Returns
        -------
        None
        """
        if bandN > self._img.shape[2] or bandN<0:
            raise ValueError("Invalid band number!")
        self.band_names[bandN] = band_name
        print("Named Band {} of Map {} as {}".format(bandN, self.name, band_name))
    def add_band(self, band, band_name = None):
        """
        Joins a new band to your map.
        
        Parameters
        ----------
        band: numpy array
            the new band to be joined. Should have the same 2D shape (or maps should be of 
            the same pixel size). Ideally, the datatype of the band should conform to that of
            the map.
        band_name: str (optional)
            name of the band to be joined. Defaults to None if not indicated.
            
        Returns
        -------
        None
        """
        if band.dtype != self._img.dtype:
            warnings.warn(
                "New band has a different dtype from the map (New: {}, Old: {})! Datatype of the map might be modified".format(band.dtype, self._img.dtype))
        bandN = self._img.shape[2]+1
        if type(band_name) != type(None):
            self.band_names[bandN] = band_name
        img = self._img.copy()
        self._img = np.dstack((img, band))
        print("Added Band {} to Map {}".format(bandN, self.name))
    def remove_band(self, bandN):
        """
        Removes the band indicated from the map.
        
        In the event of a removal, band numbers above the removed band number will be decreased by 1.
        
        Parameters
        ----------
        bandN: int
            Designated number of the band you want to remove from the map
        
        Returns
        -------
        None
        """
        dict_hold = {}
        keyList = np.array(list(self.band_names.keys()))
        for i in keyList[keyList< bandN]:
            dict_hold[i] = self.band_names[i]
        for i in keyList[keyList> bandN]:
            dict_hold[i-1] = self.band_names[i]
        self.band_names = dict_hold
        slice1 = self._img[:,:,:bandN]
        slice2 = self._img[:,:,bandN+1:]
        self._img = np.dstack((slice1, slice2))    
    def histogram(self, plot=True, list_of_bands = None):    
        """
        Shows you the distribution of the color values in the map
        
        Calculates the histogram of each color band. Provided you set plot = True, it will 
        plot the result using matplotlib.pyplot:
            - x axis: pixel value
            - y axis: frequency
        
        Parameters
        ----------
        plot: boolean (optional)
            True: will plot the histo or do you just want the values?
        list_of_bands: list or int (optional)
            int, band number/s you want to include in the histogram
        
        Returns
        -------
        dict
            contains each bands' histogram results (2 numpy arrays)
                1. frequency of each pixel value
                2. bins of the histogram
        """
        if type(list_of_bands) != type(None):
            if type(list_of_bands) == list:
                img = self._img[:,:,[i-1 for i in list_of_bands]]
            else:
                img = self._img[:,:,int(list_of_bands)]
        else:
            img = self._img
            list_of_bands = [i for i in range(1,self._img.shape[2]+1)]
        band_names = {}
        for i in list_of_bands:
            try:
                band_names[i] = self.band_names[i]
            except:
                band_names[i] = "Band " + str(i)
        color=iter(cm.rainbow(np.linspace(0,1,len(list_of_bands))))
        bands_histo = {}
        minim = int(img.min())
        maxim = int(img.max())
        for i in list_of_bands:
            pixels = agg_pixels(self[i], mask = self._mask)#imd.agg_pixels(self[i], mask = self._mask)
            bands_histo[i] = np.histogram(pixels, bins =np.arange(minim-1, maxim+1,1))
        if plot:
            plt.figure(figsize=(20,7))
            plt.title("{} Histogram".format(self.name))
            for i in bands_histo:
                c=next(color)
                band_in = bands_histo[i]
                plt.plot(band_in[1][:len(band_in[1])-1], band_in[0], label = band_names[i], color = c)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
            plt.show()
        return bands_histo 
    def statistics(self):
        """
        Calculates basic statistic description for the map
        
        Returns
        -------
        list
            contains dicts of statistical calculations from each band
                band: band_name
                mean: average value
                min: smallest value in the map
                max: largest value in the map
        """
        stats = []
        itr = iter(self)
        for i in range(self._img.shape[2]):
            pixels = agg_pixels(next(itr), mask = self._mask)#imd.agg_pixels(next(itr), mask = self._mask)
            try:
                band_name = self.band_names[i+1]
            except:
                band_name = "Band " + str(i+1)
            stats.append({'band': band_name,
                          'mean': pixels.mean(),
                          'median': np.median(pixels),
                          'min': pixels.min(),
                          'max': pixels.max()})
        return stats
    def inc_contrast(self,*args ,mask = None, plot = True):
        """
        Increases the contrast in the image by equalizing the histogram of each color band. If a mask is given, then the function will discard the pixels marked as excluded
        in its equalization

        Parameters
        ----------
        *args: int (optional)
            indicates which band you want to plug in in the red, green, and blue channels.
            if only one number is given, only 1 band is displayed.
            if no number is given, it will automatically show bands 1, 2, and 3.
        mask: numpy ndarray (optional)
            a 2D array that marks which pixels will be removed during equalization.
            If none is given, it will check if the class instance has a default mask and use that instead.
            If no mask is available, the function will proceed without removing pixels.
        plot: boolean (default=True)
            if True, will show a plot of the resulting image and the old image
            if False, will not plot

        Returns
        -------
        numpy ndarray
            image array of the contrasted image
        """

        array, band_names = self._call_channels(*args)
        if type(mask) == np.ndarray:
            mask = mask
        elif type(self._mask) == np.ndarray:
            mask = self._mask

        img_contrast = inc_contrast(array, mask = mask, plot=plot)#imd.inc_contrast(self._img[:,:,bands], mask = mask, plot=plot)
        return img_contrast
    def segmentMap_fz(self,*args, scale=100.0, sigma=0.95, min_size=50, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Felzenszwalb method. 
        
        Calls the function segmentMap_fz from the segmentation library.
        
        Parameters
        ----------
        *args: int (optional)
            indicates which band you want to plug in in the red, green, and blue channels.
            if only one number is given, only 1 band is displayed.
            if no number is given, it will automatically show bands 1, 2, and 3.
        scale: float
        sigma: float
        min_size: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean (optional)
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean (optional)
            True - segmentation is set as the default segmentation_fz of the map
            False - self.segment is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if contrast:
            img = self.inc_contrast(*args,plot=False)  
        else:
            img, band_names = self._call_channels(*args)
        if len(img.shape) == 2:
            img = np.repeat(img[:,:, np.newaxis], 3, axis=2)        
        segments_fz = segmentMap_fz(img, name=self.name, mask=self._mask,scale=scale, sigma=sigma, min_size=min_size, plot = plot)
        segments_fz.resolution = self.resolution
        if set_default:
            self._segments['Felzenszwalb'] = segments_fz
        return segments_fz    
    def segmentMap_quick(self,*args, kernel_size=3, max_dist=6, ratio=0.5, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Quickshift method. 
        
        Creates an array (with the same shape as the map) of cluster assignments of each pixel in 
        the map. Also shows the plot of the map with the boundares of each segment.See skimage 
        documentation for a more in-depth discussion of this segmentation method.
        
        Parameters
        ----------
        *args: int (optional)
            indicates which band you want to plug in in the red, green, and blue channels.
            if only one number is given, only 1 band is displayed.
            if no number is given, it will automatically show bands 1, 2, and 3.
        kernel_size: float
        max_dist: float
        ratio: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean
            True - segmentation is set as the default segmentation_fz of the map
            False - self._segments is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if contrast:
            img = self.inc_contrast(*args,plot=False)
        else:
            img, band_names = self._call_channels(*args)
        if len(img.shape) == 2:
                img = np.repeat(img[:,:, np.newaxis], 3, axis=2)              
        segments_q = segmentMap_quick(img, name=self.name, mask=self._mask,kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, plot = plot)
        segments_q.resolution = self.resolution
        if set_default:
            self._segments['Quickshift'] = segments_q
        return segments_q

    def view_segments(self, segments, seg_id = None, *args):
        """
        Plots the image with the boundaries between user's selected segmentation

        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        seg_id: int, str, or list (optional)
            key that will access the segments being called
        *args: int (optional)
            indicates which band you want to plug in in the red, green, and blue channels.
            if only one number is given, only 1 band is displayed.
            if no number is given, it will automatically show bands 1, 2, and 3.
        
        Returns
        -------
        None
        """
        segments = self.fit_segments(segments)

        slf_img, band_names = self._call_channels(*args)
        print(slf_img.shape)
        if type(seg_id) != type(None):
            mask =  segments[seg_id]
            
            if len(band_names)>2:
                mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2)
#            print(mask == True)
            print(mask.shape)
            img = np.where(mask, 0, slf_img)
        else:
            img = slf_img
        if len(band_names) == 3:
            plt.title("{}: {}, {}, and {}".format(self.name, *band_names))
        else:
            plt.title("{}: {}".format(self.name, *band_names))
        plt.imshow(mark_boundaries(img,segments.array))

    def clip_segment(self, segments,seg_id, plot = True):
        """
        Clips the image based on a segmentation and makes a new instance of a subclass of the Super_Map depending on the self's current class.
        If no segmentation is specified, it will try to default to a pre-existing segmentation stored in the instance.
        
        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
        seg_id: int, str, or list
            key that will access the segments being called
        plot: boolean (optional)
            indicates whether user wants to plot the result or not
            
        
        Returns
        -------
        Multispectral
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        newMap = super()._clip_segment(segments,seg_id)
        if plot:
            newMap.display(1,2,3)

        return newMap
    def save(self,project_name = None, overwrite=False):
        folder = super().save(project_name = project_name, overwrite=overwrite)
        if len(self.band_names) > 0:
            save(self.band_names, "band_names", folder=folder, overwrite=True)

    ######################## MAGIC FUNCTIONS ############################
    def __iter__(self):
        self.index = -1
        return self
    def __next__(self):
        self.index +=1
        if self.index >=  len(self):
            raise StopIteration 
        try:
            print("{} Band called".format(self.band_names[self.index+1]))
        except:
            print("Band {} called".format(self.index+1))     
        return self[self.index]
    def __len__(self):
        return self._img.shape[2]
    def __getitem__(self, key):
        return self._img.copy()[:,:,key-1]
    def __str__(self):
        superMeta = super().__str__()
        #"Map Name: {} \nShape (h px, w px, bands): {} \nResolution: {} m/px \nDataType: {}".format(self.name, self._img.shape, self.resolution, self._img.dtype)
        if len(self.band_names) >0:
            bandMeta = "\n\nNamed bands: {}".format(self.band_names)
        else:
            bandMeta = "\n\nNo bands have been named yet"
        return(superMeta+bandMeta)
    
class Grayscale(Super_Map):
    """
    A class for maps with only 1 band.
        
    A subclass of Super_Map that accepts only 2D arrays. Used either as a container of 
    maps with only 1 band or arrays from calculating the maps'index (ex. NDVI, EBBI, etc) 
    values.
    
    Attributes
    ----------
    _img: numpy ndarray
        array that represents the pixel value of an image
    name: str
        designated name of this map (for graphs, summaries, and saves)
    resolution: float
        meters per pixel
    _segments: dict
        contains the class instances of Segments for the map to call on
    _mask: numpy ndarray
        2D array that determines which pixels are to be excluded from calculations
        NOTE: this must have the same dimensions of pixels as _img
    List of unique functions: display, histogram, statistics
        
    
    """
    def __init__(self, img, name, resolution = None):
        """
        Constructor
        
        Parameters
        ----------
        img: numpy array
            has a shape (pixel height, pixel width, number of bands)
        name: str 
            designated name of this map (for graphs and summaries)
        resolution: float (optional)
            indicated how many meters per pixel. If there is no indicated, it defaults to None

        Returns
        -------
        Class instance of Grayscale
        
        """
        if len(img.shape) > 2:
            raise ValueError("Image has more than one band! Try using Multispectral instead")
        super().__init__(img, name, resolution)
    
    def display(self, cmap = 'gray'):
        """
        Shows a plot of the map.
        
        Parameters
        ----------
        figsize: tuple
            format is (x size,y size)
        cmap: str (optional)
            color scheme of the map. Check matplotlib.pyplot documentation for the complete 
            list of available options.
        
        Returns
        -------
        None
        """
        plt.title(self.name)
        plt.imshow(self._img, cmap=cmap)
    def histogram(self, plot=True):
        """
        Shows you the distribution of the color value in the map
        
        Calculates the histogram of your map. Provided you set plot = True, it will 
        plot the result using matplotlib.pyplot:
            - x axis: pixel value
            - y axis: frequency
        
        Parameters
        ----------
        plot: boolean (optional)
            True: will plot the histo or do you just want the values?
        
        Returns
        -------
        numpy array
            frequency of each pixel value
        numpy array
            bins of the histogram
        """
        pixels = agg_pixels(self._img, mask = self._mask)#imd.agg_pixels(self._img, mask = self._mask)
        histo = np.histogram(pixels, bins =np.linspace(pixels.min(), pixels.max()+1, 100))
        if plot:
            plt.figure(figsize=(20,7))
            plt.title("{} Histogram".format(self.name))
            plt.plot(histo[1][:len(histo[1])-1], histo[0])
            plt.show()
        return histo   
    def statistics(self):
        """
        Calculates basic statistic description for the map
        
        Returns
        -------
        dict
            mean: average value
            median: median
            min: smallest value in the map
            max: largest value in the map
        """
        pixels = agg_pixels(self._img, mask = self._mask)#imd.agg_pixels(self._img, mask = self._mask)
        stats={
                'mean': pixels.mean(),
                'median': np.median(pixels),
                'min': pixels.min(),
                'max': pixels.max()}
        return stats
    def inc_contrast(self,mask = None, plot = True):

        if type(mask) == np.ndarray:
            mask = mask
        elif type(self._mask) == np.ndarray:
            mask = self._mask

        img_contrast = inc_contrast(self._img, mask = mask, plot=plot)#imd.inc_contrast(self._img, mask = mask, plot=plot)
        return img_contrast
    def segmentMap_fz(self,scale=100.0, sigma=0.95, min_size=50, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Felzenszwalb method. 
        
        Calls the function segmentMap_fz from the segmentation library.
        
        Parameters
        ----------
        
        scale: float
        sigma: float
        min_size: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean (optional)
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean (optional)
            True - segmentation is set as the default segmentation_fz of the map
            False - self.segment is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if contrast:
            img = self.inc_contrast(plot=False)
            img = np.repeat(img[:,:, np.newaxis], 3, axis=2)    
        else:
            img = np.repeat(self.array[:,:, np.newaxis], 3, axis=2)  
        segments_fz = segmentMap_fz(img, name=self.name,mask=self._mask,scale=scale, sigma=sigma, min_size=min_size, plot = plot)
        segments_fz.resolution = self.resolution
        if set_default:
            self._segments['Felzenszwalb'] = segments_fz
        return segments_fz

    
    def segmentMap_quick(self,kernel_size=3, max_dist=6, ratio=0.5, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Quickshift method. 
        
        Creates an array (with the same shape as the map) of cluster assignments of each pixel in 
        the map. Also shows the plot of the map with the boundares of each segment.See skimage 
        documentation for a more in-depth discussion of this segmentation method.
        
        Parameters
        ----------
        kernel_size: float
        max_dist: float
        ratio: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean (optional)
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean (optional)
            True - segmentation is set as the default segmentation_fz of the map
            False - self.segment is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if contrast:
            img = self.inc_contrast(plot=False)
            img = np.repeat(img[:,:, np.newaxis], 3, axis=2)
        else:
            img = np.repeat(self.array[:,:, np.newaxis], 3, axis=2) 

        segments_q = segmentMap_quick(img, name=self.name, mask=self._mask,kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, plot = plot)
        segments_q.resolution = self.resolution
        if set_default:
            self._segments['Quickshift'] = segments_q

        return segments_q
    def clip_segment(self, segments,seg_id, plot=True):
        """
        Clips the image based on a segmentation and makes a new instance of a subclass of the Super_Map depending on the self's current class.
        If no segmentation is specified, it will try to default to a pre-existing segmentation stored in the instance.
        
        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
        seg_id: int, str, or list
            key that will access the segments being called
        plot: boolean (optional)
            indicates whether user wants to plot the result or not
            
        
        Returns
        -------
        Grayscale
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        newMap = super()._clip_segment(segments, seg_id)
        if plot:
            newMap.display()

        return newMap

    def view_segments(self, segments, seg_id = None):
        """
        Plots the image with the boundaries between user's selected segmentation

        Parameters
        ----------
        segments: Segment, str
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        seg_id: int, str, or list (optional)
            key that will access the segments being called
        Returns
        -------
        None
        """
        segments = self.fit_segments(segments)
        
        plt.title(self.name)
        if type(seg_id) != type(None):
            img = np.where(segments[seg_id], 200, self._img)
        else:
            img = self._img
        plt.imshow(mark_boundaries(img,segments.array))
    ######################## MAGIC FUNCTIONS ############################
    def __len__(self):
        return 1

def load_map(name):
    """
    Retrieves Super_Map data from a previous session and plugs it into a new instance of Segments
    
    Files of a map have to be in a folder (whose name is str name). File tree below:
        \project_folder
            \saved_data
                \name_of_map
                    metadata.pkl
                    _img.pkl
                    _mask.pkl
                    \segments
                        etc etc
                \other_map
            script1.py
            map.tif
            
    NOTE: folder "saved_data" is the default folder for placing saved files and loading. You can
    change this by going to constants.py and changing the value of the variable save_folder
    
    Parameters
    ----------
    name: str
        name of the map
        name of the folder to be accessed
        
    Returns
    -------
    Grayscale or Multispectral
        class instance of Segments with data from a previous session
    """
    folder_path = name
    img = load(folder=folder_path, name="_img")
    metadata = load(folder=folder_path, name="metadata")
    if len(img.shape) > 2:
        loaded_map = Multispectral(img, metadata['name'], metadata['resolution'])
    else:
        loaded_map = Grayscale(img, metadata['name'], metadata['resolution'])
    try:
        mask = load(folder=folder_path, name="_mask")
    except:
        mask = None
    loaded_map.set_mask(mask)
    if os.path.exists(os.path.join(save_folder, folder_path,"segments")):
        seg_dir = os.path.join(folder_path, "segments")
        for i in os.listdir(os.path.join(save_folder, folder_path, "segments")):
            print(i)
            loaded_seg = load_segments(i, folder = seg_dir)
            loaded_map.set_segments(loaded_seg,i)
    return loaded_map

        -------
        numpy array 
            stored in the class
        """
        return self._img.copy()

    @property
    def segments(self):
        """
        Gives a list of segmentations registered under the class instance

        Returns
        -------
        dict.key()
            list of registered segmentations
        """
        return self._segments.copy()

    @property
    def array_calc(self):
        return agg_pixels(self._img, mask = self._mask)

    def copy(self):
        """
        Creates a deep copy of the class instance

        Returns
        -------
        Super_Map, Grayscale, Multispectral
            Class instance of Super_Map or any of its subclasses
        """

        map_copy = deepcopy(self)
        return map_copy
    def set_mask(self, mask):
        """
        Manually sets the default mask of the map. Tags the background or unnecessary pixels in the map.
        
        Parameters
        ----------
        mask: numpy array
            2D array with all values are false/true. Indicates which pixels will be included in histograms, 
            statistics, etc. If you want to remove the mask, use : .setMask(None)
        
        Returns
        -------
        None
        """
        self._mask = mask
    def set_segments(self, segments, name):
        """
        Manually sets up a segmentation array for the class instance to access.

        Parameters
        ----------
        segments: Segment
            an instance of Segment. Has to have the same size as the image.
        name: str
            Name of the segment

        Returns
        -------
        None
        """
        if segments.__class__.__name__ != Segments.__name__:
            raise ValueError("Invalid data type! Segmentation registered must be a class instance of Segments")
        self._segments[name] = segments   
    def _get_segments(self, segments):
        """
        Private function for the class to help in retrieving and choosing segmentations for class functions
        
        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance

        Returns
        -------
        Segment
            user-selected segmentation
        """
        if type(segments) == None:
            if len(self._segments.keys()) < 0:
                raise ValueError('No segmentation available to trim image. Please add a segmentation to this function\'s argument or use .set_segments() function')
            else:
                segmentation = self._segments[self._segments.keys()[0]]
                mask = segmentation[seg_id]
                print("Retrieved segmentation name " + self._segments.keys()[0])
        elif type(segments) == str or type(segments) == list:
            segmentation = self._segments[segments]
        elif segments.__class__.__name__ == Segments.__name__:
            segmentation =segments
        else:
            raise ValueError('Invalid data type! Please input a class instance of Segments or a str indicating a set segmentation in the class instance')
        return segmentation
    def _clip_segment(self, seg_id, segments = None):
        """
        A private function that clips the image based on a segmentation and makes a new instance of a subclass of the Super_Map depending on the self's current class.
        If no segmentation is specified, it will try to default to a pre-existing segmentation stored in the instance.
        
        Parameters
        ----------
        seg_id: int, str, or list
            key that will access the segments being called
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
            
        
        Returns
        -------
        Grayscale or Multispectral
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        segmentation = self.fit_segments(segments)
        mask = segmentation[seg_id]
        newMap = self.clip_mask(int(segmentation.resolution/self.resolution), mask)
        return newMap
     
    def clip_mask(self, ratio, mask = None): #can be really slow?
        """
        Clips the image based on a mask and makes a new instance of a subclass of the Super_Map depending on the self's current class
        
        Parameters
        ----------
        ratio: int
            ratio btwn the resolution of the mask over the resolution of the img
        mask: numpy array (optional)
            2D array with all values are false/true. Will default to the set mask in the class instance if
            parameter is not indicated.
        
        
        Returns
        -------
        Grayscale or Multispectral
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        if (type(mask) != np.ndarray) and (type(self._mask) != np.ndarray):
            raise ValueError('No mask available to trim image. Please add a mask to this function\'s argument or use .set_mask() function')
        elif (type(mask) == np.ndarray):
            pass
        else:
            mask = self._mask
        result= clip(self._img, mask, ratio)#imd.clip(self._img, mask, ratio)

        if self.__class__.__name__ == Multispectral.__name__:
            newMap = Multispectral(result[0], self.name + "_clipped", resolution = self.resolution) 
        elif self.__class__.__name__ == Grayscale.__name__:
            newMap = Grayscale(result[0], self.name + "_clipped", resolution = self.resolution)
        newMap.set_mask(result[1])
        return newMap
    def save(self, overwrite=False):
        """
        Saves the data stored in a class instance for future use

        Data is placed in a subdirectory inside the folder whose name is in var save_folder. The folder name for the map is the same as the name of the class instance

        Directory:

        \working_directory
            \saved_data
                \map
                    metadata.pkl
                    _img.pkl
                    _mask.pkl (if applicable)
                    \segments (if applicable)
                        \segment1
                            - data
                        \segment2
                            - data
             -script1.py

        Parameters
        ----------
        overwrite: boolean (optional)
            if False, will not overwrite a folder whose name is the same as itself
            if True, will overwrite an same-named folder

        Returns
        -------
        str
            name of the folder holding the data

        """
        if overwrite == False and os.path.exists(os.path.join(save_folder, self.name)):
            i = 1
            new_path = self.name + "({})".format(i)
            while os.path.exists(os.path.join(save_folder,new_path)):
                i+=1
                new_path = self.name + "({})".format(i)
            path = new_path
        else:
            path = self.name
        check_folder(os.path.join(save_folder,path))
        save(self._img, "_img", folder = path, overwrite = True)
        if type(self._mask) != type(None):
            save(self._mask, "_mask", folder = path, overwrite = True)
        metadata = {'name': self.name, 'resolution': self.resolution}
        save(metadata, "metadata",folder = path, overwrite = True)

        if len(self._segments) > 0:
            for i in self._segments:
                seg_path = os.path.join(path, "segments")
                self._segments[i].save(folder=seg_path, series_name = i, overwrite=True)
        return path
    def apply_index(self, func, *args):
        """

        """
        if len(args) >0:
            new_param = [self[i].astype(float) for i in args]
            return func(*new_param)
        else:
            return func(self.array)
    def fit_segments(self, segments):
        """
        Will stretch the segment's array until it's the same size as that of the image

        Constraints:
        1. assumes that segments' resolution is bigger than that of the image
        2. can only accept ratio intergers

        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance

        Returns
        -------
        Segments
            a new segment whose size is the same as the
        """
        segmentation = self._get_segments(segments)
        try:
            ratio = int(segmentation.resolution/self.resolution)
        except:
            ratio = 1
        if ratio != 1:
            seg_array = stretch(segmentation.array, ratio)[:self._img.shape[0],:self._img.shape[1]]
            return Segments(seg_array, name = segmentation.name, resolution = segmentation.resolution/ratio)
        else:
            return segmentation

    def calc_bySegment(self, segments, func, *args, seg_id=None):
        """
        Will label each segment by the function's result.

        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        func: function 
            should return an integer or boolean
        *args: int (optional)
            band number you are going to use for your calculate
        seg_id: int, list, str (optional)
            identifier of the segment in the segmentation
        Returns
        -------
        numpy array
            result of your function for each segment
        """
        def calc_pixels(mask, array):
            if len(array.shape) > len(seg.array.shape):
                pixels = agg_pixels(self.array, mask = np.repeat(mask[:,:,np.newaxis],3, axis=2))
            else:
                pixels = agg_pixels(self.array, mask =mask)
            if len(args) > 0:
                new_param = [pixels[j-1].astype(float) for j in args]
                result = func(*new_param)
            else:
                result = func(pixels)
            return np.where(mask, result,array)
            
        seg = self.fit_segments(segments)
        array = np.zeros_like(seg.array)
        if type(seg_id) == type(None):
            for i in seg:
                array = calc_pixels(i, array)
        elif type(seg_id) == int:
            array = calc_pixels(seg[i], array)
        else:
            if type(seg_id) == list:
                keys = seg_id
            elif type(seg_id) == str:
                keys = seg.group(seg_id)
            for i in keys:
                array = calc_pixels(seg[i], array)
        return array

    ####################### MAGIC FUNCTIONS ###########################
    def __repr__(self):
        return "{} (numpy array of {}, resolution: {})".format(type(self), self._img.shape, self.resolution)
    def __str__(self):
        if type(self._mask) != type(None):
            return "Map Name: {} \nShape (h px, w px[, bands]): {} \nResolution: {} m/px \nDataType: {} \nMask of {} set.".format(self.name, self._img.shape, self.resolution, self._img.dtype, self._mask.shape)
        else:
            return "Map Name: {} \nShape (h px, w px[, bands]): {} \nResolution: {} m/px \nDataType: {}".format(self.name, self._img.shape, self.resolution, self._img.dtype)

class Multispectral(Super_Map):
    """
    A class for maps with more than 1 band (ideally at least 3)
        
    A subclass of Super_Map that accepts only 3D arrays. Used as a container for multispectral 
    maps. The first 3 bands will, by default, be treated as the RGB bands.
        
    List of functions: display, name_band, add_band, remove_band
        
    Parameters
    ----------
    img: numpy array
        has a shape (pixel height, pixel width, number of bands)
    name: str 
        designated name of this map (for graphs and summaries)
    resolution: float (optional)
        indicated how many meters per pixel. If there is no indicated, it defaults to None
            
    Returns
    -------
    Class instance of Multispectral
    """
    def __init__(self, img, name, resolution = None):
    
        if len(img.shape) < 3:
            raise ValueError("Image only has one band!")
        super().__init__(img, name, resolution)
        self.band_names = {}
        self.band_names[1] = 'Red'
        self.band_names[2] = "Green"
        if img.shape[2] > 2:
            self.band_names[3] = "Blue"
    
    def display(self, band1 = 1, band2 = None, band3=None):
        """
        Shows a plot of the map.
        
        Parameters
        ----------
        band1, band2 (optional), band3 (optional): int 
            indicates which band you want to plug in in the red, green, and blue channels. By default, band2 and band3 are set as None. 
            If either of the two of them are None when the function is called, only the first band indicated will be plotted.

        Returns
        -------
        None
        """
        if type(band2) != int or type(band3) != int:
            try:
                band_names = self.band_names[band1]
            except:
                band_names = "Band {}".format(band1)
            plt.title("{}: {}".format(self.name, band_names))
            plt.imshow(self[band1], cmap = 'gray')
        else:
            bands=[band1,band2,band3]
            band_names = []
            for i in range(3):
                try:
                    band_names.append(self.band_names[bands[i]])
                except:
                    band_names.append("Band {}".format(bands[i]))
                
            tifffile.imshow(self._img[:,:,[(i-1) for i in bands]], title = "{}: {}, {}, and {}"
                            .format(self.name, band_names[0], band_names[1], band_names[2]))
    def name_band(self, bandN, band_name):
        """
        Names an available band in the map by a name given by the user.
        
        Parameters
        ----------
        bandN: int
            band number of the band you want to name
            NOTE: when dealing with the array, bandN-1 (ex. band1 in the array is the 0th slice)
        band_name: str 
            name of the band number
            
        Returns
        -------
        None
        """
        if bandN > self._img.shape[2] or bandN<0:
            raise ValueError("Invalid band number!")
        self.band_names[bandN] = band_name
        print("Named Band {} of Map {} as {}".format(bandN, self.name, band_name))
    def add_band(self, band, band_name = None):
        """
        Joins a new band to your map.
        
        Parameters
        ----------
        band: numpy array
            the new band to be joined. Should have the same 2D shape (or maps should be of 
            the same pixel size). Ideally, the datatype of the band should conform to that of
            the map.
        band_name: str (optional)
            name of the band to be joined. Defaults to None if not indicated.
            
        Returns
        -------
        None
        """
        if band.dtype != self._img.dtype:
            warnings.warn(
                "New band has a different dtype from the map (New: {}, Old: {})! Datatype of the map might be modified".format(band.dtype, self._img.dtype))
        bandN = self._img.shape[2]+1
        if type(band_name) != type(None):
            self.band_names[bandN] = band_name
        img = self._img.copy()
        self._img = np.dstack((img, band))
        print("Added Band {} to Map {}".format(bandN, self.name))
    def remove_band(self, bandN):
        """
        Removes the band indicated from the map.
        
        In the event of a removal, band numbers above the removed band number will be decreased by 1.
        
        Parameters
        ----------
        bandN: int
            Designated number of the band you want to remove from the map
        
        Returns
        -------
        None
        """
        dict_hold = {}
        keyList = np.array(list(self.band_names.keys()))
        for i in keyList[keyList< bandN]:
            dict_hold[i] = self.band_names[i]
        for i in keyList[keyList> bandN]:
            dict_hold[i-1] = self.band_names[i]
        self.band_names = dict_hold
        slice1 = self._img[:,:,:bandN]
        slice2 = self._img[:,:,bandN+1:]
        self._img = np.dstack((slice1, slice2))    
    def histogram(self, plot=True, list_of_bands = None):    
        """
        Shows you the distribution of the color values in the map
        
        Calculates the histogram of each color band. Provided you set plot = True, it will 
        plot the result using matplotlib.pyplot:
            - x axis: pixel value
            - y axis: frequency
        
        Parameters
        ----------
        plot: boolean (optional)
            True: will plot the histo or do you just want the values?
        list_of_bands: list or int (optional)
            int, band number/s you want to include in the histogram
        
        Returns
        -------
        dict
            contains each bands' histogram results (2 numpy arrays)
                1. frequency of each pixel value
                2. bins of the histogram
        """
        if type(list_of_bands) != type(None):
            if type(list_of_bands) == list:
                img = self._img[:,:,[i-1 for i in list_of_bands]]
            else:
                img = self._img[:,:,int(list_of_bands)]
        else:
            img = self._img
            list_of_bands = [i for i in range(1,self._img.shape[2]+1)]
        band_names = {}
        for i in list_of_bands:
            try:
                band_names[i] = self.band_names[i]
            except:
                band_names[i] = "Band " + str(i)
        color=iter(cm.rainbow(np.linspace(0,1,len(list_of_bands))))
        bands_histo = {}
        minim = int(img.min())
        maxim = int(img.max())
        for i in list_of_bands:
            pixels = agg_pixels(self[i], mask = self._mask)#imd.agg_pixels(self[i], mask = self._mask)
            bands_histo[i] = np.histogram(pixels, bins =np.arange(minim-1, maxim+1,1))
        if plot:
            plt.figure(figsize=(20,7))
            plt.title("{} Histogram".format(self.name))
            for i in bands_histo:
                c=next(color)
                band_in = bands_histo[i]
                plt.plot(band_in[1][:len(band_in[1])-1], band_in[0], label = band_names[i], color = c)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
            plt.show()
        return bands_histo 
    def statistics(self):
        """
        Calculates basic statistic description for the map
        
        Returns
        -------
        list
            contains dicts of statistical calculations from each band
                band: band_name
                mean: average value
                min: smallest value in the map
                max: largest value in the map
        """
        stats = []
        itr = iter(self)
        for i in range(self._img.shape[2]):
            pixels = agg_pixels(next(itr), mask = self._mask)#imd.agg_pixels(next(itr), mask = self._mask)
            try:
                band_name = self.band_names[i+1]
            except:
                band_name = "Band " + str(i+1)
            stats.append({'band': band_name,
                          'mean': pixels.mean(),
                          'median': np.median(pixels),
                          'min': pixels.min(),
                          'max': pixels.max()})
        return stats
    def inc_contrast(self,band1=1, band2=None, band3=None ,mask = None, plot = True):
        if type(band2) != int or type(band3) != int:
            bands = band1-1
        else:
            bands=[band1-1,band2-1,band3-1]

        if type(mask) == np.ndarray:
            mask = mask
        elif type(self._mask) == np.ndarray:
            mask = self._mask

        img_contrast = inc_contrast(self._img[:,:,bands], mask = mask, plot=plot)#imd.inc_contrast(self._img[:,:,bands], mask = mask, plot=plot)
        return img_contrast
    def segmentMap_fz(self,band1 = 1, band2 = None, band3 = None, scale=100.0, sigma=0.95, min_size=50, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Felzenszwalb method. 
        
        Calls the function segmentMap_fz from the segmentation library.
        
        Parameters
        ----------
        band1, band2 (optional), band3 (optional): int
            indicates which bands will be plugged into the RGB channels for segmentation. Will default to monochrome if two bands at most of the arguments are given.
        scale: float
        sigma: float
        min_size: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean (optional)
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean (optional)
            True - segmentation is set as the default segmentation_fz of the map
            False - self.segment is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if type(band2) != int or type(band3) != int:
            bands = [band1-1, band1-1, band1-1]
            try:
                band_names = self.band_names[band1-1]
            except:
                band_names = "Band {}".format(band1-1)
            
        else:
            bands=[band1-1,band2-1,band3-1]
            band_names = []
            for i in range(3):
                try:
                    band_names.append(self.band_names[bands[i]])
                except:
                    band_names.append("Band {}".format(bands[i]))
        if contrast:
            img = self.inc_contrast(band1 = band1, band2 = band2, band3 = band3,plot=False)
            if len(img.shape) == 2:
                img = np.repeat(img[:,:, np.newaxis], 3, axis=2)    
        else:
            img = self.array[:,:,bands]        
        segments_fz = segmentMap_fz(img, name=self.name,scale=scale, sigma=sigma, min_size=min_size, plot = plot)
        segments_fz.resolution = self.resolution
        if set_default:
            self._segments['Felzenszwalb'] = segments_fz
        return segments_fz    
    def segmentMap_quick(self,band1 = 1, band2 = None, band3 = None, kernel_size=3, max_dist=6, ratio=0.5, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Quickshift method. 
        
        Creates an array (with the same shape as the map) of cluster assignments of each pixel in 
        the map. Also shows the plot of the map with the boundares of each segment.See skimage 
        documentation for a more in-depth discussion of this segmentation method.
        
        Parameters
        ----------
        band1, band2 (optional), band3 (optional): int
            indicates which bands will be plugged into the RGB channels for segmentation. Will default to monochrome if two bands at most of the arguments are given.
        kernel_size: float
        max_dist: float
        ratio: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean
            True - segmentation is set as the default segmentation_fz of the map
            False - self._segments is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if type(band2) != int or type(band3) != int:
            bands = [band1-1, band1-1, band1-1]
            try:
                band_names = self.band_names[band1]
            except:
                band_names = "Band {}".format(band1)
            
        else:
            bands=[band1-1,band2-1,band3-1]
            band_names = []
            for i in range(3):
                try:
                    band_names.append(self.band_names[bands[i]])
                except:
                    band_names.append("Band {}".format(bands[i]))
        if contrast:
            img = self.inc_contrast(band1 = band1, band2 = band2, band3 = band3,plot=False)
            if len(img.shape) == 2:
                img = np.repeat(img[:,:, np.newaxis], 3, axis=2)    
        else:
            img = self.array[:,:,bands]             
        segments_q = segmentMap_quick(img, name=self.name,kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, plot = plot)
        segments_q.resolution = self.resolution
        if set_default:
            self._segments['Quickshift'] = segments_q
        return segments_q

    def view_segments(self, segments, seg_id = None, band1 = 1, band2 = None, band3 = None):
        """
        Plots the image with the boundaries between user's selected segmentation

        Parameters
        ----------
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        seg_id: int, str, or list (optional)
            key that will access the segments being called
        band1, band2 (optional), band3 (optional): int 
            indicates which band you want to plug in in the red, green, and blue channels. By default, band2 and band3 are set as None. 
            If either of the two of them are None when the function is called, only the first band indicated will be plotted.
        
        Returns
        -------
        None
        """
        segments = self.fit_segments(segments)

        if type(band2) != int or type(band3) != int:
            bands = band1-1
            try:
                band_names = self.band_names[band1]
            except:
                band_names = "Band {}".format(band1)
            
        else:
            bands=[band1-1,band2-1,band3-1]
            band_names = []
            for i in range(3):
                try:
                    band_names.append(self.band_names[bands[i]+1])
                except:
                    band_names.append("Band {}".format(bands[i]+1))

                print("accessed array {}".format(bands[i]))
        slf_img = self._img[:,:,bands]
        print(slf_img.shape)
        if type(seg_id) != type(None):
            mask =  segments[seg_id]
            
            if type(bands) == list:
                mask = np.repeat(mask[:,:, np.newaxis], 3, axis=2)
#            print(mask == True)
            print(mask.shape)
            img = np.where(mask, 0, slf_img)
        else:
            img = slf_img
        plt.title("{}: {}, {}, and {}".format(self.name, band_names[0], band_names[1], band_names[2]))
        plt.imshow(mark_boundaries(img,segments.array))

    def clip_segment(self,seg_id, segments = None, plot = True):
        """
        Clips the image based on a segmentation and makes a new instance of a subclass of the Super_Map depending on the self's current class.
        If no segmentation is specified, it will try to default to a pre-existing segmentation stored in the instance.
        
        Parameters
        ----------
        seg_id: int, str, or list
            key that will access the segments being called
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        plot: boolean (optional)
            indicates whether user wants to plot the result or not
            
        
        Returns
        -------
        Multispectral
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        newMap = super()._clip_segment(seg_id, segments = segments)
        if plot:
            newMap.display(1,2,3)

        return newMap
    def save(self,overwrite=False):
        folder = super().save(overwrite=overwrite)
        if len(self.band_names) > 0:
            save(self.band_names, "band_names", folder=folder, overwrite=True)

    ######################## MAGIC FUNCTIONS ############################
    def __iter__(self):
        self.index = -1
        return self
    def __next__(self):
        self.index +=1
        if self.index >=  len(self):
            raise StopIteration 
        try:
            print("{} Band called".format(self.band_names[self.index+1]))
        except:
            print("Band {} called".format(self.index+1))     
        return self[self.index]
    def __len__(self):
        return self._img.shape[2]
    def __getitem__(self, key):
        return self._img.copy()[:,:,key-1]
    def __str__(self):
        superMeta = super().__str__()
        #"Map Name: {} \nShape (h px, w px, bands): {} \nResolution: {} m/px \nDataType: {}".format(self.name, self._img.shape, self.resolution, self._img.dtype)
        if len(self.band_names) >0:
            bandMeta = "\n\nNamed bands: {}".format(self.band_names)
        else:
            bandMeta = "\n\nNo bands have been named yet"
        return(superMeta+bandMeta)
    
class Grayscale(Super_Map):
    """
    A class for maps with only 1 band.
        
    A subclass of Super_Map that accepts only 2D arrays. Used either as a container of 
    maps with only 1 band or arrays from calculating the maps'index (ex. NDVI, EBBI, etc) 
    values.
        
    List of unique functions: display, histogram, statistics
        
    Parameters
    ----------
    img: numpy array
        has a shape (pixel height, pixel width, number of bands)
    name: str 
        designated name of this map (for graphs and summaries)
    resolution: float (optional)
        indicated how many meters per pixel. If there is no indicated, it defaults to None
            
    Returns
    -------
    Class instance of Grayscale
    """
    def __init__(self, img, name, resolution = None):
        
        if len(img.shape) > 2:
            raise ValueError("Image has more than one band! Try using Multispectral instead")
        super().__init__(img, name, resolution)
    
    def display(self, cmap = 'gray'):
        """
        Shows a plot of the map.
        
        Parameters
        ----------
        figsize: tuple
            format is (x size,y size)
        cmap: str (optional)
            color scheme of the map. Check matplotlib.pyplot documentation for the complete 
            list of available options.
        
        Returns
        -------
        None
        """
        plt.title(self.name)
        plt.imshow(self._img, cmap=cmap)
    def histogram(self, plot=True):
        """
        Shows you the distribution of the color value in the map
        
        Calculates the histogram of your map. Provided you set plot = True, it will 
        plot the result using matplotlib.pyplot:
            - x axis: pixel value
            - y axis: frequency
        
        Parameters
        ----------
        plot: boolean (optional)
            True: will plot the histo or do you just want the values?
        
        Returns
        -------
        numpy array
            frequency of each pixel value
        numpy array
            bins of the histogram
        """
        pixels = agg_pixels(self._img, mask = self._mask)#imd.agg_pixels(self._img, mask = self._mask)
        histo = np.histogram(pixels, bins =np.linspace(pixels.min(), pixels.max()+1, 100))
        if plot:
            plt.figure(figsize=(20,7))
            plt.title("{} Histogram".format(self.name))
            plt.plot(histo[1][:len(histo[1])-1], histo[0])
            plt.show()
        return histo   
    def statistics(self):
        """
        Calculates basic statistic description for the map
        
        Returns
        -------
        dict
            mean: average value
            median: median
            min: smallest value in the map
            max: largest value in the map
        """
        pixels = agg_pixels(self._img, mask = self._mask)#imd.agg_pixels(self._img, mask = self._mask)
        stats={
                'mean': pixels.mean(),
                'median': np.median(pixels),
                'min': pixels.min(),
                'max': pixels.max()}
        return stats
    def inc_contrast(self,mask = None, plot = True):

        if type(mask) == np.ndarray:
            mask = mask
        elif type(self._mask) == np.ndarray:
            mask = self._mask

        img_contrast = inc_contrast(self._img, mask = mask, plot=plot)#imd.inc_contrast(self._img, mask = mask, plot=plot)
        return img_contrast
    def segmentMap_fz(self,scale=100.0, sigma=0.95, min_size=50, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Felzenszwalb method. 
        
        Calls the function segmentMap_fz from the segmentation library.
        
        Parameters
        ----------
        
        scale: float
        sigma: float
        min_size: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean (optional)
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean (optional)
            True - segmentation is set as the default segmentation_fz of the map
            False - self.segment is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if contrast:
            img = self.inc_contrast(plot=False)
            img = np.repeat(img[:,:, np.newaxis], 3, axis=2)    
        else:
            img = np.repeat(self.array[:,:, np.newaxis], 3, axis=2)  
        segments_fz = segmentMap_fz(img, name=self.name,scale=scale, sigma=sigma, min_size=min_size, plot = plot)
        segments_fz.resolution = self.resolution
        if set_default:
            self._segments['Felzenszwalb'] = segments_fz
        return segments_fz

    
    def segmentMap_quick(self,kernel_size=3, max_dist=6, ratio=0.5, plot = True, contrast = False, set_default = False): #will slow/crash your laptop
        """
        Segments the map based on the Quickshift method. 
        
        Creates an array (with the same shape as the map) of cluster assignments of each pixel in 
        the map. Also shows the plot of the map with the boundares of each segment.See skimage 
        documentation for a more in-depth discussion of this segmentation method.
        
        Parameters
        ----------
        kernel_size: float
        max_dist: float
        ratio: float
        plot: boolean
            if True, function will plot the segmentation results
        contrast: boolean (optional)
            if True, function will heighten the contrast of the image with function equalize_hist
        set_default: boolean (optional)
            True - segmentation is set as the default segmentation_fz of the map
            False - self.segment is not touched
        
        Returns
        -------
        Segments
            An instance of class Segment whose assigned array contains the cluster assignment of 
            each pixel in the map. Size of the array is the same as the img. 
        """
        if contrast:
            img = self.inc_contrast(plot=False)
            img = np.repeat(img[:,:, np.newaxis], 3, axis=2)
        else:
            img = np.repeat(self.array[:,:, np.newaxis], 3, axis=2) 

        segments_q = segmentMap_quick(img, name=self.name,kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, plot = plot)
        segments_q.resolution = self.resolution
        if set_default:
            self._segments['Quickshift'] = segments_q

        return segments_q
    def clip_segment(self, seg_id, segments = None, plot=True):
        """
        Clips the image based on a segmentation and makes a new instance of a subclass of the Super_Map depending on the self's current class.
        If no segmentation is specified, it will try to default to a pre-existing segmentation stored in the instance.
        
        Parameters
        ----------
        seg_id: int, str, or list
            key that will access the segments being called
        segments: Segment, str (optional)
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        plot: boolean (optional)
            indicates whether user wants to plot the result or not
            
        
        Returns
        -------
        Grayscale
            A new instance of a subclass of Super_Map whose map is the clipped image produced by the function. Mask from the clipping
            is set as default.

        """
        newMap = super()._clip_segment(seg_id, segments = segments)
        if plot:
            newMap.display()

        return newMap

    def view_segments(self, segments, seg_id = None):
        """
        Plots the image with the boundaries between user's selected segmentation

        Parameters
        ----------
        segments: Segment, str
            if Segment, contains a numpy array with segmentation assigments of the pixels in the map.
            if str, indicates what segmentation (already set in the class instance) to be used
            if None, function will default to the first-set segmentation registered in the class instance
        seg_id: int, str, or list (optional)
            key that will access the segments being called
        Returns
        -------
        None
        """
        segments = self.fit_segments(segments)
        
        plt.title(self.name)
        if type(seg_id) != type(None):
            img = np.where(segments[seg_id], 200, self._img)
        else:
            img = self._img
        plt.imshow(mark_boundaries(img,seg_array))
    ######################## MAGIC FUNCTIONS ############################
    def __len__(self):
        return 1

def load_map(name):
    """
    Retrieves Super_Map data from a previous session and plugs it into a new instance of Segments

    Parameters
    ----------
    name: str
        name of the map
        
    Returns
    -------
    Grayscale or Multispectral
        class instance of Segments with data from a previous session
    """
    folder_path = name
    img = load(folder=folder_path, name="_img")
    metadata = load(folder=folder_path, name="metadata")
    if len(img.shape) > 2:
        loaded_map = Multispectral(img, metadata['name'], metadata['resolution'])
    else:
        loaded_map = Grayscale(img, metadata['name'], metadata['resolution'])
    try:
        mask = load(folder=folder_path, name="_mask")
    except:
        mask = None
    loaded_map.set_mask(mask)
    if os.path.exists(os.path.join(save_folder, folder_path,"segments")):
        seg_dir = os.path.join(folder_path, "segments")
        for i in os.listdir(os.path.join(save_folder, folder_path, "segments")):
            print(i)
            loaded_seg = load_segments(i, folder = seg_dir)
            loaded_map.set_segments(loaded_seg,i)
    return loaded_map
