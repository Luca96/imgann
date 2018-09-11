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
import utils

from utils import dlib
from utils import Colors


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


def test_shape_predictor(xml, model):
    '''wraps the dlib.test_shape_predictor method, to test
    the accuracy of the [model] on the labels described in [xml]'''
    error = dlib.test_shape_predictor(xml, model)
    print("model error: {} on {}".format(error, xml))


def of_dataset(folder="testset", model=None, view=False):
    '''measure the error across the given dataset,
    it compares the measured points with the annotated ground truth,
    optionally you can [view] the results'''
    assert(model)

    # load face and landmark detectors
    utils.load_shape_predictor(model)
    # utils.init_face_detector(True, 150)

    # init average-error
    err = 0
    num = 0

    for img, lmarks, path in utils.ibug_dataset(folder):
        # detections
        face = utils.prominent_face(utils.detect_faces(img, detector="dlib"))
        measured = utils.detect_landmarks(img, face)

        # get error
        num += 1
        err += normalized_root_mean_square(lmarks, measured)

        # results:
        if view is True:
            utils.draw_rect(img, face, color=Colors.yellow)
            utils.draw_points(img, lmarks, color=Colors.green)
            utils.draw_points(img, measured, color=Colors.red)
            utils.show_image(utils.show_properly(utils.crop_image(img, face)))

    print(err, num, err / num)
    print("average NRMS Error for {} is {}".format(folder, err / num))


def compare_models(folder="testset", m1=None, m2=None, view=False):
    '''compare the [m1] shape_predictor aganist the [m2] model,
    optionally you can [view] the results'''
    assert(m1 and m2)

    utils.init_face_detector(True, 150)

    # load models
    utils.load_shape_predictor(m2)
    sp_m2 = utils.shape_predictor

    utils.load_shape_predictor(m1)
    sp_m1 = utils.shape_predictor

    # init error
    err = 0
    num = 0

    for face, region in utils.faces_inside(folder):
        h, w = face.shape[:2]
        if h == 0 or w == 0:
            continue

        box = utils.Region(0, 0, region.width, region.height)
        # detect landmarks
        utils.shape_predictor = sp_m1
        lmarks_m1 = utils.detect_landmarks(face, box)

        utils.shape_predictor = sp_m2
        lmarks_m2 = utils.detect_landmarks(face, box)

        # update error:
        num += 1
        # err += normalized_root_mean_square(lmarks_m1, lmarks_m2)

        # results:
        if view is True:
            utils.draw_points(face, lmarks_m1, color=Colors.green)
            utils.draw_points(face, lmarks_m2, color=Colors.red)
            utils.show_image(utils.show_properly(face))

    if num != 0:
        err /= num

    print("the NRMSE of m1 aganist m2 is {}".format(err))
