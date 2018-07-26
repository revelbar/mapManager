MapManager.py Tutorial
======================

## Prerequisite Knowledge 

I have tried to keep the usage of this package as user-friendly as possible; however, some skills in the following packages will significantly help you:

1. Numpy (Python Library)
2. GDAL (just look up how to use gdalinfo)

## CODING 

The best Python 3 libraries to import (besides MapManager) before starting any project are numpy (for handling arrays), tifffile from skimage.external (my go-to function for handling tiff files), and matplotlib.pyplot for plotting heatmaps and images among others.

	# Useful External Libraries
	import matplotlib.pyplot as plt
	from skimage.external import tifffile

	# Importing the package

	import mapManager as mm

To open an image and convert it to an array of numbers, you can use any image library you are familiar with, but my personal go-to is tifffile:

	img = tifffile.imread('map.tif') # this will output an array with the shape (# of pixels height, # of pixels width, # of bands (if map has more than 1 band)). See PIL or OpenCV for more options

From there, if your file only has 1 band:

	map_gray = mm.Grayscale(img, "name", resolution = 0.25) #To get the resolution of a map, simply run 'gdalinfo path_to_img' in your command line.

If your file has more than 1 band:

	map_multi = mm.Multispectral(img, "name", resolution = 0.25) #To get the resolution of a map, simply run 'gdalinfo path_to_img' in your command line.

You can check any documentation for any function by doing the following:

	print(mm.CLASS.function_name.__doc__) #for any function in a class
	print (mm.function.__doc__) #for any function outside a class
	print(mm.CLASS.__doc__) #for any info on any class

### BASIC MANIPULATION OF MAPS

To plot the image:
	
	map_gray.display() #for only 1 band
	map_multi.display(red_channel, green_channel, blue_channel)

To get a basic statistical description of the map:

	var = map_gray.statistics()
	var = map_multi.statistics()

To show a histogram of the color band/s:

	var = map_gray.histogram()
	var = map_multi.histogram()

There's also a built-in function for increasing the contrast of an image:

	var = map_gray.inc_contrast()
	var = map_multi.inc_contrast()

#### ARRAY MANIPULATION OF MAPS

If you need to get the array stored in the class to do custom calculations:
	
	var = map_gray.array
	var = map_multi.array

Alternatively, the Multispectral class comes with its own magic functions in handling its multi-band data. If you wish to call only one band in its array:

	band = map_multi[band_number]

You can also loop through the bands in the Multispectral class instance:
	
	for i in map_multi:
		do_something()

You can check how many bands you have:
	len(map_multi)

To add/remove a band in a multispectral map:
	
	map_multi.add_band(img_array, "name")
	map_multi.remove_band(band_num)

You can also assign a name to a band:

	map_multi.name_band(4, 'NIR')

To add markers for pixels to be removed from any calculation:

	map_multi.set_mask(mask_array)


## SEGMENTATION 


The module comes with its own way of handling segmenting maps via the class Segments:

	segments = mm.Segments(array, "name") # array is 2D with each value an id for each pixel

You can attach labels to certain id #:
	
	segments.name_segment(10, 'name') #we label segment id 10 as 'name'

Removing a name would be:

	segments.remove_name('name')

To get the array back:

	var = segments.array

Alternatively, you can use built-in magic functions to call areas that belong to certain segments.

	var = segments[1] # calls segment id 1
	var = segments['name'] # calls segment ids under the grouping 'name'
	var = segments[[1,2,3,4]] # calls segment ids 1,2,3, and 4

These calls will create a mask of the segmentation that will label the segments called as True. You can graph these masks by the function display:

	segments.display() # will not highlight anything, just show the segmentation of the image
	segments.display(1) # will highlight segment id 1
	segments.display('name') # will highlight segment ids under the grouping 'name'
	segments.display([1,2,3,4]) # will highlight segment ids 1,2,3, and 4

The package comes with two default segmentation techniques: Felnzenswalb's and the Quickshift methods. To use either:

	var = mm.segmentMap_fz(img) # see documentation for variables you can toggle with this algorithm to get different results
	var = mm.segmentMap_quick(img) # see documentation for variables you can toggle with this algorithm to get different results

Or, if you have already loaded your image data in a class instance of Grayscale or Multispectral:

	seg_q1 = map_multi.segmentMap_quick(1,2,3) # setting bands 1,2,3 in channels 1,2,3, we apply the quickshift algorithm on the image

	seg_q1 = map_multi.segmentMap_fz(1,2,3,contrast=True) # setting bands 1,2,3 in channels 1,2,3 and increasing the contrast in the image, we apply Felzenswalb's algorithm on the image

	seg_q2 = map_gray.segmentMap_fz(1,2,3,contrast=True) # setting bands 1,2,3 in channels 1,2,3 and increasing the contrast in the image, we apply Felzenswalb's algorithm on the image

You can automatically add the segmentation into the class instance by adding set_default = True in the function's argument (ex. map_gray.segmentMap_fz(1,2,3,contrast=True, set_default = True)). If you have any other segmentation you wish to add:

	map_multi.set_segments(seg_q1, 'name') # this will only accept a class instance of Segments

You can call the available segmentations of a certain map by:

	seg = map_multi.segments
	print(seg.keys()) # will show you the names of the available segments

	seg['seg_name'].segmentation_function # do what you want with any segmentation in the function

You can clip any part of the map tagged as part of a segment by:

	newMap = map_multi.clip_segment(200, 'name') # clipped map part under seg id 200 in the added segmentation named 'name'
	newMap = map_multi.clip_segment(200, seg_q2) # clipped map part under seg id 200 in segmentation seg_q2
