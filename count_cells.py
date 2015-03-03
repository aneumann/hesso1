"""
author: Alexander Neumann
date: 03.03.2015 10:33:21
"""

# imports
import numpy as np
import pylab
import mahotas as mh
import matplotlib.pyplot as plt


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

# load img
working_dir = "C:/project_image_processing/steady_state/131015_zoom1.7/"
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

# skeleton transform
img_dist = mh.distance(img_closed)
img_dist = mh.gaussian_filter(img_dist, 8)
img_dist = img_dist.astype(int)

labels_dist, nr_objects = mh.label(img_dist)

centers = mh.center_of_mass(img_dist, labels_dist)

print type(centers)
# find max values, seeds
rmax = mh.regmax(img_dist)

# show original vs processed
img_processed = img_closed

pylab.gray()
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(img_processed)
plt.subplot(2, 2, 3)
plt.imshow(img_dist)
plt.subplot(2, 2, 4)
#pylab.imshow(mh.overlay(img_dist, labels_dist))
pylab.imshow(centers)
pylab.show()
