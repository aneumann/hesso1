"""
author: Alexander Neumann
date: 04.03.2015 10:00:01
"""

# imports
from scipy import ndimage
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
import pylab

# load img
working_dir = '/Users/alexander_neumann/Desktop/bioinf/steady_state/131015_zoom1.7/'
img_dir = working_dir + 'Series006_z0_ch00.tif'
img = mh.imread(img_dir)

# what first? opening or thresholding
# thresholding to remove dark fragments
img_binary = img > 30

# morphological operation to reduce small fragments
img_opened = ndimage.grey_erosion(img_binary, size=(1, 1))
print type(img_opened[0][0])

# count connected structures

# show
img_processed = img_opened
pylab.gray()
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(img_processed)
pylab.show()
