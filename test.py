# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:52:43 2015

@author: Alexander Neumann
"""

# imports
import numpy as np
import scipy.fftpack
import mahotas
import pylab


def week_two():
    # Do a basic implementation of JPEG.
    #     Divide the image into non-overlapping 8x8 blocks.
    #     Compute the DCT (discrete cosine transform) of each block. This is implemented in popular packages such as
    #       Matlab. Quantize each block. You can do this using the tables in the video or simply divide each coefficient
    #       by N, round the result to the nearest integer, and multiply back by N. Try for different values of N. You
    #       can also try preserving the 8 largest coefficients (out of the total of 8x8=64), and simply rounding them to
    #       the closest integer.
    #     Visualize the results after you invert the quantization and the DCT.
    # Repeat the above but instead of using the DCT, use the FFT (Fast Fourier Transform).
    # Repeat the above JPEG-type compression but don’t use any transform, simply perform quantization on the original
    #   image.
    # Do JPEG now for color images. In Matlab, use the rgb2ycbcr command to convert the Red-Green-Blue image to a Lumina
    #   and Chroma one; then perform the JPEG-style compression on each one of the three channels independently. After
    #   inverting the compression, invert the color transform and visualize the result. While keeping the compression
    #   ratio constant for the Y channel, increase the compression of the two chrominance channels and observe the results.
    # Compute the histogram of a given image and of its prediction errors. If the pixel being processed is at coordinate
    #   (0,0), consider
    #     predicting based on just the pixel at (-1,0);
    #     predicting based on just the pixel at (0,1);
    #     predicting based on the average of the pixels at (-1,0), (-1,1), and (0,1).
    # Compute the entropy for each one of the predictors in the previous exercise. Which predictor will compress better?
    week = 2


# helper functions
def split_image2blocks(image):
    # splits an image to 8x8 blocks and returns a list of blocks
    hsplit_size = image.shape[0] / 8
    vsplit_size = image.shape[1] / 8
    block_list = []
    h_blocks = np.array_split(image, hsplit_size, axis=0)
    for h_array in h_blocks:
        v_blocks = np.array_split(h_array, vsplit_size, axis=1)
        block_list.append(v_blocks)
    return block_list


def jpeg_implementation(image):
    # a basic implementation of JPEG:
    #   1. divide the image into non-overlapping 8x8 blocks
    #   2. compute the discrete cosine transform (DCT) of each block
    #   3. visualize the results after inverting the quantization and the DCT

    # build a list with all blocks
    blocks = split_image2blocks(image)

    # perform dct on blocks
    dct_blocks = []
    for block_list in blocks:
        for block in block_list:
            dct_block = scipy.fftpack.dct(block)
            dct_blocks.append(dct_block)


img_dir = '/Users/alexander_neumann/Desktop/bioinf/test/'
img_name = '199_C7_3_blue_red_green.jpg'
img = mahotas.imread(img_dir + img_name)
img_1ch = img[:,:,0]
jpeg_implementation(img_1ch)

#pylab.imshow(jpegged_img)
#pylab.show()
