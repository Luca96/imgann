# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2018 Luca Anzalone
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
# -- Error: provide functions to analize the shape_predictor accuracy
# -----------------------------------------------------------------------------
import math


def __distance(p1, p2):
    '''returns the distance between two points'''
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def normalized_root_mean_square(truth, measured):
    '''returns the NRMSE across the ground truth and the measure points'''
    assert(len(truth) == len(measured))
    dist = 0

    # inter-ocular distance
    iod = __distance(truth[36], truth[45])

    for i in range(0, len(truth)):
        dist += __distance(truth[i], measured[i])

    return dist / iod


def point_to_point(truth, measured):
    '''returns the point-to-point error across truth and measured points'''
    assert(len(truth) == len(measured))

    Min = 2 ^ 31
    Max = -2 ^ 31
    Avg = 0
    iod = __distance(truth[36], truth[45])
    num = len(truth)

    for i in range(0, num):
        err = __distance(truth[i], measured[i]) / iod
        Avg += err

        if err > Max:
            Max = err

        if err < Min:
            Min = err

    return Min, Max, Avg / num
