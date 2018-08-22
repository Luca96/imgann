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
# -- Utils
# -----------------------------------------------------------------------------
import os
import cv2
import dlib
import json
import numpy as np
import argparse


from xml import Xml
from math import radians, sin, cos
from cv2.dnn import blobFromImage


# constants:
caffe_model = "caffe/res10_300x300_ssd_iter_140000.caffemodel"
caffe_proto = "caffe/deploy.prototxt"
cf_values = (104.0, 177.0, 123.0)
# cf_size = (300, 300)
# cf_size = (200, 200) # better
cf_size = (150, 150)  # best, detect even small faces
cf_scale = 1.0
confidence_threshold = 0.55

state_file = ".state"

# global variables:
caffeNet = None
shape_predictor = None


# -----------------------------------------------------------------------------
# -- CLASS UTILS
# -----------------------------------------------------------------------------
class Keys:
    '''opencv key constants'''
    S = 115
    R = 114
    Q = 113
    ESC = 27


class Colors:
    '''a set of predefined BGR colors for drawing functions'''
    white = (255, 255, 255)
    black = (0, 0, 0)
    cyan = (255, 255, 128)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    purple = (255, 64, 255)


class Annotation:
    '''face annotations'''

    def __init__(self, path, box=[], points=[]):
        self.path = path
        self.box = box
        self.points = points

    def save(self):
        '''save the annotation obj to a file'''
        with open(self.path, "w") as file:
            file.write(json.dumps({'box': self.box, 'points': self.points}))

        return self

    def load(self):
        '''load an annotation obj froma file'''
        print(self.path)
        with open(self.path, 'r') as file:
            data = json.loads(file.read())
            self.box = data["box"]
            self.points = data["points"]

        return self

    def inside(folder="."):
        '''iterate through all annotation inside the given folder'''
        for img, path in images_inside(folder):
            folder, file = os.path.split(path)

            # load the annotation relative to the current image
            ann_path = os.path.join(folder, file.split(".")[0] + ".ann")

            yield Annotation(ann_path).load(), path


# -----------------------------------------------------------------------------
def cli_arguments():
    '''build and parse command line arguments'''
    ap = argparse.ArgumentParser()

    s = """ generate an output file of annotations, it the file ends with .xml
        it generates an xml file ready to be used with dlib """

    # output file
    ap.add_argument("-o", "--out", required=True, help=s)

    # input directory
    ap.add_argument("-d", "--dir", required=True,
                    help="input directory with images")

    # (flag) append mode
    ap.add_argument("-a", "--append", action="store_const", const='a',
                    help="open the output file in append mode")

    # (flag) mirror points and images along x axis
    ap.add_argument("-m", "--mirror", action="store_true",
                    help="mirror points and images along x axis")

    # (flag) detect faces automatically
    ap.add_argument("--auto", action="store_true",
                    help="detect faces automatically")

    # (optional) detect landmarks
    ap.add_argument("-l", "--land", required=False,
                    help="automatically detect landmark")

    # (optional) train a model
    ap.add_argument("-t", "--train", required=False,
                    help="train a dlib shape-predictor model")

    return vars(ap.parse_args())


def load_state(flag):
    '''return the content of the state file about the last stopped execution'''
    resume, path = False, None

    if not os.path.exists(state_file):
        # create the checkpoint-file
        file = open(state_file, mode="w")
        file.close()
    else:
        # load the content of the file
        file = open(state_file, mode="r")
        path = file.read()
        resume = len(path) > 0

    return resume and flag, path


def save_state(path):
    '''update the state file with the new path'''
    file = open(state_file, mode="w")
    file.write(path)
    file.close()


def delete_state():
    '''remove the state file'''
    if os.path.isfile(state_file):
        os.remove(state_file)


def train_model(xml_path, model_path):
    '''tran a dlib shape-predictor model based on xml'''
    # model options
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 3
    options.nu = 0.1
    options.cascade_depth = 13
    options.be_verbose = True

    # train and save
    dlib.train_shape_predictor(xml_path, model_path, options)


def count_files_inside(directory="."):
    '''return the number of files (does not consider folders) in directory'''
    path, dirs, files = next(os.walk(directory))

    return len(files)


# -----------------------------------------------------------------------------
# -- FILE UTILS
# -----------------------------------------------------------------------------
def open_file(args):
    '''return an opened output file'''
    name = args["out"]
    mode = args["append"] or "w"

    if args["train"]:
        mode = "a"

    if name.endswith(".xml"):
        return Xml(name, mode=mode)
    else:
        return open(name, mode)


# -----------------------------------------------------------------------------
# -- IMAGE UTILS
# -----------------------------------------------------------------------------
def crop_image(image, roi, scale=1):
    '''returns the cropped image according to a region-of-interest'''
    t, l, r, b = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    if scale != 1:
        w = r - l
        h = b - t
        cx = r - w / 2
        cy = b - h / 2
        w = w * scale
        h = h * scale
        hw = w / 2
        hh = h / 2

        t, b = int(cy - hh), int(cy + hh)
        l, r = int(cx - hw), int(cx + hw)

    ih, iw = image.shape[:2]

    if t < 0:
        t = 0

    if b > ih:
        b = ih

    if l < 0:
        l = 0

    if r > iw:
        r = iw

    return image[l:b, t:r].copy(), (t, l, r, b)


def rotate_image(image, angle=0):
    '''rotate the given image'''
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)

    return cv2.warpAffine(image, M, (w, h))


def delete_image(path):
    '''delete the given image and the mirrored one if it exists'''
    folder, file = os.path.split(path)

    # delete mirror
    mirror = os.path.join(folder, file.replace(".", "_mirror."))

    if os.path.isfile(mirror):
        os.remove(mirror)

    os.remove(path)


def is_image(file):
    '''check if the given file is an image'''
    return (file.endswith(".jpg") or file.endswith(".jpeg") or
            file.endswith(".png"))


def is_mirrored(img_file):
    '''check if the given image is the mirror of another one'''
    return img_file.find("_mirror") > 0


def flip_image(image, axis=1):
    '''mirror the given image along x axis by default'''
    return cv2.flip(image, axis)


def images_inside(directory=""):
    '''generate all the images within the given directory'''

    for path, dirs, files in os.walk(directory):
        # scan every file in subfolders
        for file in files:
            # skip non-image file
            if not is_image(file):
                continue

            # load image
            img_path = os.path.join(path, file)

            yield cv2.imread(img_path), img_path


# -----------------------------------------------------------------------------
# -- FACE UTILS
# -----------------------------------------------------------------------------
def init_face_detector(flag=False, size=150, scale_factor=1.0):
    '''load the caffe-model if --auto flag is specified,
    typical values are: 224, 227, 299, 321'''
    global caffeNet, cf_size, cf_scale

    if flag is True:
        cf_size = (size, size)
        cf_scale = scale_factor
        caffeNet = cv2.dnn.readNetFromCaffe(caffe_proto, caffe_model)


def detect_faces(image):
    '''detect every face inside image.
       Returns a list of tuple\\rectangles: (top, left, right, bottom)'''
    assert(caffeNet)

    # get image dimension
    (h, w) = image.shape[:2]
    np_arr = np.array([w, h, w, h])

    if h <= 0 or w <= 0:
        return []

    # convert image to blob (that do some preprocessing..)
    blob = blobFromImage(cv2.resize(image, cf_size), cf_scale,
                         size=cf_size, mean=cf_values, swapRB=True)

    # obtain detections and predictions
    caffeNet.setInput(blob)
    detections = caffeNet.forward()

    # detected face-boxes
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:
            # compute the bounding box of the face
            box = detections[0, 0, i, 3:7] * np_arr
            boxes.append(box.astype("int"))

    return boxes


def faces_inside(directory="", scale_factor=1):
    '''generate all faces within a given directory'''

    for path, dirs, files in os.walk(directory):
        # scan every file in subfolders
        for file in files:
            # skip non-image file
            if not is_image(file):
                continue

            # load image
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)

            # detect faces within image
            regions = detect_faces(image)

            for region in regions:
                # crop the region
                face, new_region = crop_image(image, region, scale_factor)

                yield face, new_region


# -----------------------------------------------------------------------------
# -- LANDMARK UTILS
# -----------------------------------------------------------------------------
def load_shape_predictor(model_path):
    '''load the dlib shape predictor model'''
    global shape_predictor
    shape_predictor = dlib.shape_predictor(model_path)


def detect_landmarks(face, region):
    '''detect landmarks for the given face and face region,
    returns an array of tuples (x, y)'''
    t, l, r, b = region
    rect = dlib.rectangle(t, l, r, b)
    shape = shape_predictor(face, rect)
    points = []

    for i in range(0, shape.num_parts):
        point = shape.part(i)
        points.append((point.x, point.y))

    return points


def rotate_landmarks(points, center, angle=0):
    '''rotate the given points according to the specified angle'''
    rad = radians(-angle)
    siny = sin(rad)
    cosx = cos(rad)

    new_pts = []

    for p in points:
        x = p[0] - center[0]
        y = p[1] - center[1]

        px = int(x * cosx - y * siny + center[0])
        py = int(x * siny + y * cosx + center[1])

        new_pts.append((px, py))

    return new_pts


# -----------------------------------------------------------------------------
# -- DRAWING UTILS
# -----------------------------------------------------------------------------
def draw_rect(image, rect, color=(128, 0, 128), thickness=1):
    '''draw the given rectangle (top, left, right, bottom) on image'''
    top_left = (rect[0], rect[1])
    right_bm = (rect[2], rect[3])
    cv2.rectangle(image, top_left, right_bm, color, thickness)


def draw_point(image, x, y, radius=3, color=(0, 255, 255), thickness=-1):
    '''draw a circle on image at the given coordinate'''
    cv2.circle(image, (x, y), radius, color, thickness)


def draw_points(image, points, radius=3, color=(0, 255, 255), thickness=-1):
    '''draw a list of (x, y) point tuple'''
    for i, p in enumerate(points):
        cv2.circle(image, (p[0], p[1]), radius, color, thickness)
        cv2.putText(image, str(i), (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    radius / 10, Colors.white)
# -----------------------------------------------------------------------------
