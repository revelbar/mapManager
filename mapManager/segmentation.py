import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, quickshift, mark_boundaries
from .constants import *
from .file_manager import *
import os

def load_segments(name, folder = None):
    """
    Retrieves Segment data from a previous session and plugs it into a new instance of Segments

    Parameters
    ----------
    name: str
        name of the segment
    folder: str (optional)
        name where the folder for the segment should be. Leave blank if it's just at the 1st level of the save_folder directory
    
    Returns
    -------
    Segments
        class instance of Segments with data from a previous session
    """
    if type(folder) == type(None):
        folder_path = name
    else:
        folder_path = os.path.join(folder, name)
    img = load(folder=folder_path, name="_img")
    metadata = load(folder=folder_path, name="metadata")

    loaded_seg = Segments(img, metadata['name'], metadata['resolution'])
    try:
        segment_names = load(folder=folder_path, name="_segment_names")
        
    except:
        segment_names = {}
    loaded_seg._segment_names = segment_names
    return loaded_seg
    
    


def segmentMap_fz(img, name="",scale=100.0, sigma=0.95, min_size=50, plot = True): #will slow/crash your laptop
    """
    Segments the map based on the Felzenszwalb method. 
        
    Creates an array (with the same shape as the map) of cluster assignments of each pixel in 
    the map. Also shows the plot of the map with the boundares of each segment.See skimage 
    documentation for a more in-depth discussion of this segmentation method.
        
    Parameters
    ----------
    img: numpy array 
        array representation of the image to be segmented. Has to have the shame (h px, w px, 3)
    name: str
        name of the instance of the segmentation
    scale: float
    sigma: float
    min_size: float
    plot: boolean (optional)
        if True, will display a plot of the segmentation
        
    Returns
    -------
    Segments
        An instance of class Segment whose assigned array contains the cluster assignment of 
        each pixel in the map. Size of the array is the same as the img. 
    """        
    segments_fz = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
    if plot:
        plt.figure(figsize=(10,10))
        plt.title("Felzenszwalbs's method")
        plt.imshow(mark_boundaries(img, segments_fz))
    class_segments = Segments(segments_fz, name = name + "_fz {},{},{}".format(scale,sigma,min_size))
    return class_segments

def segmentMap_quick(img, name ="", kernel_size=3, max_dist=6, ratio=0.5, plot = True): #will slow/crash your laptop
    """
    Segments the map based on the Quickshift method. 
        
    Creates an array (with the same shape as the map) of cluster assignments of each pixel in 
    the map. Also shows the plot of the map with the boundares of each segment.See skimage 
    documentation for a more in-depth discussion of this segmentation method.
        
    Parameters
    ----------
    img: numpy array 
        array representation of the image to be segmented. Has to have the shame (h px, w px, 3)
    name: str
        name of the instance of the segmentation
    kernel_size: float
    max_dist: float
    ratio: float
    plot: boolean (optional)
        if True, will display a plot of the segmentation
    Returns
    -------
    Segments
        An instance of class Segment whose assigned array contains the cluster assignment of 
        each pixel in the map. Size of the array is the same as the img. 
    """
    segments_q = quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    print('Quickshift number of segments: {}'.format(len(np.unique(segments_q))))
    if plot:
        plt.figure(figsize=(10,10))
        plt.title("Quickshift method")
        plt.imshow(mark_boundaries(img, segments_q))
    class_segments = Segments(segments_q, name = name + "_fz {},{},{}".format(kernel_size, max_dist, ratio))

    return class_segments

class Segments(object):
    """
    A class that will handle results of the segmentation from the other map classes. Aims to
    streamline the process of calling segments and creating masks
        
    Parameters
    ----------
    segments: numpy array
        2D array of ints with each unique int a segment label
    name: str (optional) 
        name of the segment 
    resolution: int (optional)
        m/px of the map
    """
    def __init__(self, segments, name = None, resolution = 1.0):

        self._img = segments
        self.name = name
        self._segment_names = {} # self.segment_names['name'] = [cluster 1, cluster 2, cluster 3, etc]
        self.resolution = resolution ######
    @property
    def array(self):
        """
        Returns
        -------
        numpy array
            copy of the img's array
        """
        return self._img.copy()
    @property
    def keys(self):
        """
        Gives the list of segment names dictated by you

        Returns
        -------
        dict.keys
            user-defined segment names
        """
        return self._segment_names.keys()
    @property
    def group(self):
        return self._segment_names.copy()
    @property
    def id (self):
        """
        Gives an array of all the available seg id's

        Returns
        -------
        numpy array
            all the unique ids available in the segmentation
        """
        return np.unique(self._img)    
    def display(self, seg_id =None, figsize=(10,10)):
        """
        Shows a plot of the segments, or highlights the segments queried.
        
        Parameters
        ----------
        seg_id: list, int, str (optional)
            list of segment_ids they want to highlight
            int. seg_id they want to highlight
            str. will refer to the dict self._segment_names to get the list of seg_id to be queried
        figsize: tuplet of numbers
            size of figure
        Returns
        -------
        None
        """
        plt.figure(figsize=figsize)
        if type(self.name) != type(None):
            plt.title(self.name)
        if type(seg_id) != type(None):
            if (type(seg_id) == int or type(seg_id) == str) or type(seg_id) == list:
                highlights= np.where(self[seg_id], 255, 0).astype('uint8')
#            elif :
 #               mask = np.zeros_like(self._img)
  #              for i in seg_id:
   #                 mask = np.where((mask == 1) | (self._img == i), 1, 0)
    #            highlights = np.where(mask > 0, 255, 0).astype('uint8')
            else:
                raise ValueError("Invalid data type in the called index! Only strings and ints are allowed")
            plt.imshow(mark_boundaries(highlights, self._img))
        else:
            plt.imshow(mark_boundaries(self._img, self._img))
    def name_segment(self, seg_id, name):
        """
        Names a segment in the map
        
        Adds an entry in the dictionary self.segment_names, with the name as a key and seg_id 
        as an entry in the list bound to the key
        
        Parameters
        ----------
        seg_id: int
            the int label of the segment in the map
        name: str
            the name you want to assign to the segment in question
        
        Returns
        -------
        None
        """
        if seg_id in np.unique(self._img):
            pass
        else:
            raise KeyError("No such segment in the map!")
        try:
            self._segment_names[name].append(seg_id)
        except:
            self._segment_names[name] = []
            self._segment_names[name].append(seg_id)
    def remove_name (self, name):
        del self._segment_names[name]
        print("Removed segment grouping {}".format(name))
    def merge_segments(self,list_of_segid):
        """
        Changes the seg_ids of all the segments indicated to the same id so they will 
        all be considered as one segment. The first element of the list will be considered
        as the id of the new segment
        
        NOTE: USING THIS FUNCTION CAN RENDER YOUR SEGMENT NAMES INVALID, ESP WHEN SOME SEGMENTS
        BELONG TO MORE THAN ONE GROUP. ONLY USE THIS BEFORE YOU START NAMING SEGMENTS.
        BEGINNING WHEN YOU ARE 
        
        Parameters
        ----------
        list_of_segid: list, str
            ints. Contains the seg_ids of the segments that will merge
            str. Name of the previously grouped cluster.
        
        Returns
        -------
        None
        """
        if type(list_of_segid) == list:
            mask = np.zeros_like(self._img)
            for i in list_of_segid:
                mask = np.where((mask == 1) | (self._img == i), 1, 0)
            
            result = np.where(mask > 0, new_id, self._img)
            
            print("Segments {} have been renamed as {}".format(list_of_segid, new_id))
        elif type(list_of_segid) == str:
            mask = self[list_of_segid]
            new_id = self._segment_names[list_of_segid][0]
            result = np.where(mask > 0, new_id, self._img)
            print("Segments of group {} have been renamed as {}".format(list_of_segid, new_id))
        else:
            raise KeyError("Argument's datatype is invalid.")
        self._img = result
    def save(self, folder = None, series_name = None, overwrite=False):
        sup_folder = None
        if type(folder) == type(None) and type(series_name) == type(None):
            folder = self.name
        elif type(folder) == type(None) and type(series_name) != type(None):
            folder = series_name
        elif type(folder) != type(None) and type(series_name) == type(None):
            sup_folder = folder
            folder = self.name
        else:
            sup_folder = folder
            folder = series_name

        if type(sup_folder) != type(None):
            path = os.path.join(sup_folder, folder)
        else:
            path = folder

        if overwrite == False and os.path.exists(os.path.join(save_folder, path)):
            i = 1
            new_folder = folder + "({})".format(i)
            if type(sup_folder) != type(None):
                new_path = os.path.join(sup_folder, new_folder)
            else:
                new_path = new_folder
            while os.path.exists(os.path.join(save_folder,new_path)):
                i+=1
                if type(sup_folder) != type(None):
                    new_path = os.path.join(sup_folder, new_folder)
                else:
                    new_path = new_folder
                    new_path = self.name + "({})".format(i)
            path = new_path
            
        check_folder(os.path.join(save_folder,path))
        save(self._img, "_img", folder = path, overwrite = overwrite)
        if len(self._segment_names) > 0:
            save(self._segment_names, "_segment_names", folder =path, overwrite=overwrite)
        metadata = {'name': self.name, 'resolution': self.resolution}
        save(metadata, "metadata", folder=path, overwrite=overwrite)
        return path
    def export(self, overwrite = False):
        if len(export_folder) > 0: 
            check_folder(export_folder)
        path = os.path.join(export_folder, self.name+"_export.txt")
        if overwrite != True and os.path.exists(path):
            path = overwrite_name(self.name + "_export", "txt",export_folder)
        np.savetxt(path, self._img, fmt='%d')
        print ("Exported segments as " +path)

################### MAGIC COMMANDS ###########################
    def __len__(self):
        return len(np.unique(self._img))
    def __getitem__(self, key):
        """
        Makes the segments of the array callable via indexing syntax
        
        Parameters
        ----------
        key: str, int, list
            if string, it will refer to the dictionary of segment_names to get the segments who are
            labeled by str key
            if int, it will refer to the int id of the segments given by the array
            if list, it will highlight all of the pixels whose int id matches with any of the ints in the list
        Returns
        -------
            numpy array
        
        """
        if type(key) == str:
            segment_ids = self._segment_names[key] #list
            result = np.zeros_like(self._img)
            for i in segment_ids:
                result = np.where((result == 1) | (self._img == i), 1, 0)
        
        elif type(key) == list:
            result = np.zeros_like(self._img)
            for i in key:
                result = np.where((self._img == i), 1, result)
        else:
            result = np.where((self._img == key), 1, 0)
        
        return result > 0
    def __iter__(self):
        self.index = -1
        return self
    def __next__(self):
        self.index +=1
        if self.index >= len(self):
            raise StopIteration
        ind = self.id[self.index]
        return self[ind]
    def __str__(self):
        name = ""
        if type(self.name) != type(None):
            name = " " + self.name
        return "Map{} has a shape of {}\n{} unique segments\n\nNamed segments:\n{}".format(name, self._img.shape, len(self), self._segment_names)
    