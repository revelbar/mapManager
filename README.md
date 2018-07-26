# mapManager

A python package on image processing specifically for geographical maps.

MapManager is a module wrapped around scikit-image for the purposes of image processing and data visualization of GeoTiff files.

Example:

	import MapManager as mm
	from skimage.external import tifffile

	img = tifffile.imread('map.tif')

	map = mm.Multispectral(img, resolution = 0.25) #To get the resolution of a map, simply run 'gdalinfo path' in your command line.

	#To display the file
	map.display()

	#Get histogram
	map.histogram()

	#Get statistics
	map.statistics()

	#Get the description of the map
	print(map)

Features
--------

- Handle multispectral bands
- Plot or display maps
- Apply masks on maps to crop or clip them
- Output basic statistical description and histograms of each band in a mapp
- Apply Felzenszwalb's and the Quickshift methods of region-based segmentation
- Streamline handling and production of multiple segmentations and masks for a map
- Calculate indices from available maps

## SETTING UP

Before opening Python and running your codes, make sure to keep the data and the package in the same folder as your code:

project_folder

├── script.py

├── map.tif

├── mapManager

|   ├── __init__.py

|   ├── segmentation.py

|   ├── cls_maps.py

|   └── imageData.py


Table of Contents of Documentation 
--------

1. Readme
2. Tutorial Guide
3. Examples
