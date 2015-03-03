"""
author: Alexander Neumann
date: 03.03.2015 10:33:21
"""

# imports
import numpy as np
import pylab
import mahotas as mh
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.filter import canny

# helper functions
# array loop
def closing_element(element_size):
    if element_size % 2 == 0:
        print 'Please give odd element size next time!'
        element_size -= 1
    temp_element = np.zeros((element_size / 2 + 1, element_size / 2 + 1))
    for i in range(len(temp_element)):
        for j in range(len(temp_element)):
            if i + j < element_size / 2:
                temp_element[i][j] = 0
            else:
                temp_element[i][j] = 1
    temp90 = np.rot90(temp_element)
    temp_half = np.concatenate((temp_element, temp90))
    temp180 = np.rot90(temp_half, 2)
    temp_full = np.concatenate((temp_half, temp180), axis=1)
    t1 = np.delete(temp_full, element_size / 2, 0)
    element = np.delete(t1, element_size / 2, 1)
    return element

# hough
def fuck_hough(image):
    edges = canny(img_closed)

    hough_radius = np.arange(160, 361, 15)
    hough_res = hough_circle(edges, hough_radius)
    centers = []
    accums = []
    radii = []
    for radius, h in zip(hough_radius, hough_res):
        num_peaks = 5
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        print peaks
    return edges

# load img
working_dir = '/Users/alexander_neumann/Desktop/bioinf/steady_state/131015_zoom1.7/'
img_dir = working_dir + 'Series006_z0_ch00.tif'
img = mh.imread(img_dir)

# median filter
filter_size = np.ones((3,) * len(img.shape), img.dtype)
img_filtered = mh.median_filter(img, Bc=filter_size)

# get binary
img_cut = img_filtered > 10

# distance transform
img_distance = mh.distance(img_cut)

# morphological closing
closing_size = closing_element(9)
img_closed = mh.close(img_distance > 5, Bc=closing_size)

img_hough = fuck_hough(img_closed)

# show original vs processed
img_processed = img_hough
pylab.gray()
plt.subplot(1, 2, 1)
plt.imshow(img_closed)
plt.subplot(1, 2, 2)
plt.imshow(img_processed)
pylab.show()
