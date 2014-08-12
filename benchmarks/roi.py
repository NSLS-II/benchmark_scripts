################################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven National       #
# Laboratory. All rights reserved.                                             #
#                                                                              #
# Redistribution and use in source and binary forms, with or without           #
# modification, are permitted provided that the following conditions are met:  #
#                                                                              #
# * Redistributions of source code must retain the above copyright notice,     #
#   this list of conditions and the following disclaimer.                      #
#                                                                              #
# * Redistributions in binary form must reproduce the above copyright notice,  #
#  this list of conditions and the following disclaimer in the documentation   #
#  and/or other materials provided with the distribution.                      #
#                                                                              #
# * Neither the name of the European Synchrotron Radiation Facility nor the    #
#   names of its contributors may be used to endorse or promote products       #
#   derived from this software without specific prior written permission.      #
#                                                                              #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"  #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE    #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE   #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE    #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR          #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS     #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN      #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)      #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                                  #
################################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging
logger = logging.getLogger(__name__)
import time
import matplotlib as mpl
from matplotlib import pyplot

"""
roi.py is a benchmark to evaluate the speed difference between:

Option 1:
1. Create binary mask where the roi is True and everything else is False
2. In-place multiply binary mask with data set.
3. Sum masked array

Option 2:
1. Generate list of roi coordinates
   a. 1-D [x1, x2, x3 ...]
   b. 2-D [(x1, y1), (x2, y2), ... ]
   c. 3-D [(x1, y1, z1), (x2, y2, z2), ... ]
2. Iterate over data sets and extract the relevant values at each coordinate
3. Sum coordinate values
"""
def get_bin_mask(dsize, roi_list):
    bin_mask = np.zeros(dsize)
    for (x,y) in roi_list:
        bin_mask[x][y] = 1
    return bin_mask

def option_1(data_list, roi_list, bin_mask, stat_func, make_bin_mask = True):
    """
    Option 1:
    1. Create binary mask where the roi is True and everything else is False
    2. In-place multiply binary mask with data set.
    3. Sum masked array

    Parameters
    ----------
    data : list
        ndarray list
    roi : list
        coordinate list. len(roi[0]) == data[0].ndims
    stat_func :
        sum, avg, stddev, etc...
    """
    if(make_bin_mask):
        bin_mask = get_bin_mask(data_list[0].shape, roi_list)
    roi = []
    for data in data_list:
        masked = np.multiply(data, bin_mask)
        roi.append(stat_func(masked))

    return roi


def option_2(data_list, roi_list, stat_func):
    """
    Option 2:
    1. Generate list of roi coordinates
       a. 1-D [x1, x2, x3 ...]
       b. 2-D [(x1, y1), (x2, y2), ... ]
       c. 3-D [(x1, y1, z1), (x2, y2, z2), ... ]
    2. Iterate over data sets and extract the relevant values at each coordinate
    3. Sum coordinate values

    Parameters
    ----------
    data : list
        ndarray list
    roi : list
        coordinate list. len(roi[0]) == data[0].ndims
    stat_func :
        sum, avg, stddev, etc...
    """
    roi = []
    for data_list in data_list:
        cur_val = 0
        for(x,y) in roi_list:
            cur_val += data_list[x][y]
        roi.append(cur_val)

    return roi


def datagen_2d(nx, ny, nz):
    data = []
    for _ in range(nz):
        data.append(np.ones((nx, ny)))

    return data


def get_2d_circle_coords(cx, cy, radius, nx, ny):
    min_x = cx - radius
    max_x = cx + radius

    min_y = cx - radius
    max_y = cx + radius

    if min_x < 0:
        min_x = 0
    if max_x > nx:
        max_x = nx

    if min_y < 0:
        min_y = 0
    if max_y > ny:
        max_y > ny

    coords_list = []
    for y in np.arange(min_y, max_y, 1):
        y_rel = y-cy
        for x in np.arange(min_x, max_x, 1):
            x_rel = x-cx
            len = np.sqrt(y_rel*y_rel + x_rel*x_rel)
            if len < radius:
                coords_list.append((x, y))

    return coords_list

if __name__ == "__main__":

    nx = 2048
    ny = 2048
    nz = 10
    cx = nx / 2
    cy = ny / 2
    radius = 25
    stat_func = np.sum
    data_list = datagen_2d(nx, ny, nz)
    roi_list = get_2d_circle_coords(cx, cy, radius, nx, ny)
    print("Approx area of circle: {0}".format(len(roi_list)))
    print("Computed area of circle: {0}".format(np.pi * radius * radius))

    radii = np.arange(5, 75, 5)

    cycles = 10
    opt_1_vals = []
    opt_1_err = []
    opt_2_vals = []
    opt_2_err = []
    opt_3_vals = []
    opt_3_err = []
    num_pixels = []
    roi_pixels = []
    for radius in radii:
        roi_list = get_2d_circle_coords(cx, cy, radius, nx, ny)
        # create the binary mask
        bin_mask = get_bin_mask(data_list[0].shape, roi_list)
        time_1 = []
        time_2 = []
        time_3 = []
        for _ in range(cycles):
            t1 = time.time()
            option_1(data_list=data_list, bin_mask=bin_mask,
                     roi_list=roi_list, stat_func=stat_func,
                     make_bin_mask=True)
            t2 = time.time()
            option_2(data_list=data_list, roi_list=roi_list,
                     stat_func=stat_func)
            t3 = time.time()
            option_1(data_list=data_list, bin_mask=bin_mask,
                     roi_list=roi_list, stat_func=stat_func,
                     make_bin_mask=False)
            t4 = time.time()
            time_1.append(t2-t1)
            time_2.append(t3-t2)
            time_3.append(t4-t3)

        roi_pixels.append(len(roi_list))
        opt_1_vals.append(np.average(time_1))
        opt_1_err.append(np.std(time_1))
        opt_2_vals.append(np.average(time_2))
        opt_2_err.append(np.std(time_2))
        opt_3_vals.append(np.average(time_3))
        opt_3_err.append(np.std(time_3))

    ax = pyplot.gca()
    ax.errorbar(roi_pixels, opt_1_vals, yerr=opt_1_err,
                label="construct binary mask on the fly and apply to image stack")
    ax.errorbar(roi_pixels, opt_3_vals, yerr=opt_3_err,
                label="apply a pre-defined binary mask to image stack")
    ax.errorbar(roi_pixels, opt_2_vals, yerr=opt_2_err,
                label="extract roi from coords list")
    ax.legend(loc='upper right')
    ax.set_xlabel("Number of pixels in ROI")
    ax.set_ylabel("Average time for {0} cycles (s)".format(cycles))

    pyplot.show()