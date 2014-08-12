# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import numpy as np
import logging

logger = logging.getLogger(__name__)
import time

from matplotlib import pyplot as plt
from collections import deque, defaultdict

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
    bin_mask = np.zeros(dsize, dtype=bool)
    for (x, y) in roi_list:
        bin_mask[x, y] = True
    return bin_mask


def option_1(data_list, roi_list, bin_mask, stat_func, make_bin_mask=True):
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
    if (make_bin_mask):
        bin_mask = get_bin_mask(data_list[0].shape, roi_list)
    roi = []
    for data in data_list:
        masked = np.multiply(data, bin_mask)
        roi.append(stat_func(masked))

    return roi


def option_1a(data_list, roi_list, bin_mask, stat_func, make_bin_mask=True):
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
    roi = deque()
    for data in data_list:
        roi.append(stat_func(data[bin_mask]))

    return np.array(roi)


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
        for (x, y) in roi_list:
            cur_val += data_list[x][y]
        roi.append(cur_val)

    return roi


def option_3(data_list, roi_list, stat_func):
    data = np.asarray(data_list)
    bin_mask = get_bin_mask(data.shape[1:], roi_list)
    return stat_func(data * bin_mask, axis=tuple(range(1, data.ndim)))


def option_4(data_list, roi_list, stat_func):
    data = np.asarray(data_list)
    bin_mask = get_bin_mask(data.shape[1:], roi_list)
    return stat_func(data[:, bin_mask], axis=1)


def datagen_2d(nx, ny, nz):
    return [np.ones((nx, ny)) for j in range(nz)]


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
        max_y = ny

    coords_list = []
    for y in np.arange(min_y, max_y, 1):
        y_rel = y - cy
        for x in np.arange(min_x, max_x, 1):
            x_rel = x - cx
            len = np.sqrt(y_rel * y_rel + x_rel * x_rel)
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

    radii = np.arange(75, 5, -5)

    cycles = 5
    test_functions = [
         {'color': 'r',
          'func': option_1,
          'kwargs': {'make_bin_mask': True},
          'label': 'op1_make',
          'ls': '--'},
         {'color': 'r',
          'func': option_1,
          'kwargs': {'make_bin_mask': False},
          'label': 'op1_pre',
          'ls': '-'},
         {'color': 'b',
          'func': option_1a,
          'kwargs': {'make_bin_mask': True},
          'label': 'op1a_make',
          'ls': '--'},
         {'color': 'b',
          'func': option_1a,
          'kwargs': {'make_bin_mask': False},
          'label': 'op1a_pre',
          'ls': '-'},
         {'color': 'k',
          'func': option_3,
          'kwargs': {},
          'label': 'op3',
          'ls': '-'},
         {'color': 'g',
          'func': option_4,
          'kwargs': {},
          'label': 'op4',
          'ls': '-'}]

    vals = defaultdict(list)
    errs = defaultdict(list)

    roi_pixels = []

    for radius in radii:
        roi_list = get_2d_circle_coords(cx, cy, radius, nx, ny)
        roi_pixels.append(len(roi_list))
        bin_mask = get_bin_mask(data_list[0].shape, roi_list)
        for data, label_post_fix in zip((data_list, np.asarray(data_list)),
                                    ('_list', '_array')):
            for test_dict in test_functions:
                # un-pack the useful stuff
                label = test_dict['label'] + label_post_fix
                t_kw = test_dict['kwargs']
                tf = test_dict['func']
                time_deque = deque()
                # special case option 1
                if 'make_bin_mask' in t_kw:
                    t_kw['bin_mask'] = bin_mask
                # loop over number of cycles
                for _ in range(cycles):
                    # get the time
                    t1 = time.time()
                    # run the function
                    res = tf(data_list=data,
                         roi_list=roi_list, stat_func=stat_func,
                         **t_kw)
                    # get the time after
                    t2 = time.time()
                    # record the delta
                    time_deque.append(t2 - t1)
                # compute the statistics
                vals[label].append(np.mean(time_deque))
                errs[label].append(np.std(time_deque))

    # do plotting
    fig, ax = plt.subplots(1, 1)
    for test_dict in test_functions:
        c = test_dict['color']
        ls = test_dict['ls']

        for lw, post_fix in zip((1, 4), ('_list', '_array')):
            label = test_dict['label'] + post_fix
            ax.errorbar(roi_pixels, vals[label],
                        yerr=errs[label],
                        label=label, color=c, linestyle=ls, lw=lw)

    ax.legend(loc='upper right')
    ax.set_xlabel("Number of pixels in ROI")
    ax.set_ylabel("Average time for {0} cycles (s)".format(cycles))

    plt.show()
