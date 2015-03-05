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
np.set_printoptions(threshold='nan')
#print labels_dist

sizes =  mh.labeled.labeled_size(labels_dist)
too_small_nuclei = np.where(sizes <900)
labels_dist_new = mh.labeled.remove_regions(labels_dist, too_small_nuclei)

# voronoi segmentation
whole = mh.segmentation.gvoronoi(labels_dist)

# fnd background
background = mh.median_filter(img, Bc=np.ones((5,)*len(img.shape), img.dtype))
background = img < 3
background = mh.median_filter(background, Bc=np.ones((5,)*len(img.shape), img.dtype))
background = mh.median_filter(background, Bc=np.ones((5,)*len(img.shape), img.dtype))

background_label, nr_objects_two = mh.label(background)
sizes_background =  mh.labeled.labeled_size(background_label)
background_big = np.where(sizes_background < 2000)
background_label_new = mh.labeled.remove_regions(background_label, background_big)
background_label[background_label>255] = 250
background_label_new[background_label_new>255] = 250

background_label_new  = mh.close_holes(background_label_new, Bc=np.ones((5,)*len(img.shape), img.dtype))
background_watershed = 

print nr_objects_two
#print background_label

#background = mh.close(background,Bc=np.ones((3,)*len(img.shape), img.dtype))
#background = mh.distance(background)
#background = np.floor(background)
#print background
#background = mh.dilate(background,Bc=np.ones((13,)*len(img.shape), img.dtype))

######################################################################

#not needed anymore
# count nuclei with a certain area threshold of 900
#hist, bin_edges = np.histogram(labels_dist, bins=256)
#count_nuclei = hist[hist!=0]
#count_nuclei = np.delete(count_nuclei,0)
# area threshold
#too_small_nuclei = np.where(count_nuclei<900)
#count_nuclei = count_nuclei[(count_nuclei>900)]

# clean labels_dist image from too_small_nuclei
#masked = np.ma.masked_less(count_nuclei, 900)
#print masked
#print labels_dist
#labels_dist_old = np.copy(labels_dist)

#for idx in range(len(too_small_nuclei[0])):
#    #labels_dist_new =
#   #np.where(labels_dist == too_small_nuclei[0][idx])
#
#    labels_dist[labels_dist == too_small_nuclei[0][idx]] = 0

###########################################################################



# centers of mass, find the middle of the cell areas
centers=mh.center_of_mass(img_dist, labels_dist)

#print type(centers)
#print centers
# find max values, seeds
rmax = mh.regmax(img_dist)

# show original vs processed
img_processed = img_closed

pylab.gray()
plt.subplot(2, 2, 1)
plt.imshow(background_label_new)
plt.subplot(2, 2, 2)
pylab.imshow(labels_dist)
plt.subplot(2, 2, 3)
pylab.imshow(background_label)
plt.subplot(2, 2, 4)
pylab.imshow(mh.overlay(background_label_new,labels_dist))
#pylab.imshow(background)
pylab.show()
