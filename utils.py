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
face_det = None
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
    '''face annotations (.ann extension)'''

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
            ann_path = os.path.join(folder, file.split(".")[-1] + ".ann")

            yield Annotation(ann_path).load(), path

    def parse_ibug_annotation(file=None):
        '''returns an array of the points defined inside the given
        annotation file'''
        pts = []

        for line in file.readlines()[3:-2]:
            x, y = line.split()[:2]
            x = int(x.split(".")[0])
            y = int(y.split(".")[0])

            pts.append((x, y))

        return pts


class Region:
    '''class to express rectangular region easly'''

    def __init__(self, x, y, w, h):
        '''creates a new rect: requires top-left corner (x, y) and size'''
        assert(w > 0 and h > 0)

        # dimension:
        self.width = int(w)
        self.height = int(h)
        self.half_w = int(self.width / 2)
        self.half_h = int(self.height / 2)

        # position:
        self.left = int(x)
        self.top = int(y)
        self.right = int(x + w)
        self.bottom = int(y + h)

    def square(x, y, size):
        '''creates a square region'''
        return Region(x, y, size, size)

    def copy(self):
        '''returns a copy of the current region'''
        return Region(self.left, self.top, self.width, self.height)

    def dlib(rect):
        '''creates a Region from a dlib rect'''
        return Region(rect.left(), rect.top(), rect.width(), rect.height())

    def tuple(opencv_rect):
        '''creates a region from an opencv (left, top, right, bottom) tuple'''
        left, top, right, bottom = opencv_rect
        return Region(left, top, right - left, bottom - top)

    def ensure(self, width, height):
        '''edit the current region to respect the given dimension'''
        if self.left < 0:
            self.left = 0

        if self.top < 0:
            self.top = 0

        if self.width > width:
            self.width = int(width)
            self.half_w = int(width / 2)

        if self.height > height:
            self.height = int(height)
            self.half_h = int(height / 2)

        self.right = self.left + self.width
        self.bottom = self.top + self.height

        return self

    def center(self):
        '''return the center of the region as a tuple'''
        return (self.left + self.half_w, self.top + self.half_h)

    def move_center(self, pt):
        '''move the region in order to be centered at the given point'''
        xc, yc = pt

        self.left = xc - self.half_w
        self.right = xc + self.half_w
        self.top = yc - self.half_h
        self.bottom = yc + self.half_h

        return self

    def move_at(self, pt, axis=None):
        '''move the center of region in order to be at the given point,
        optionally axis can be locked'''
        if axis is None:
            return self.move_center(pt)

        elif axis == "x":
            x0 = pt[0]
            self.left = x0 - self.half_w
            self.right = x0 + self.half_w
            return self

        elif axis == "y":
            y0 = pt[1]
            self.top = y0 - self.half_h
            self.bottom = y0 + self.half_h
            return self

    def tl(self):
        '''return the top-left corner as a point-tuple (left, top)'''
        return (self.left, self.top)

    def br(self):
        '''return the bottom-right corner as a point-tuple (bottom, right)'''
        return (self.right, self.bottom)

    def scale(self, fx=1, fy=1):
        '''scale the the region by the specified factors, the scaled region
        will have the same center of the original one'''
        w = self.width * fx
        h = self.height * fy
        dw = (w - self.width) / 2
        dh = (h - self.height) / 2

        return Region(self.left - dw, self.top - dh, w, h)

    def origin(self, x=0, y=0):
        '''change the top-left corner of the region'''
        self.x = x
        self.y = y

    def area(self):
        '''returns the area of the region'''
        return self.width * self.height

    def unpack(self):
        '''returns left, top, right, bottom, width and height'''
        w = self.width
        h = self.height

        return self.left, self.top, self.right, self.bottom, w, h

    def as_dlib(self):
        '''returns a dlib.rectangle'''
        return dlib.rectangle(self.left, self.top, self.right, self.bottom)

    def as_tuple(self):
        '''returns a tuple (left, top, right, bottom) suitable for opencv'''
        return (self.left, self.top, self.right, self.bottom)

    def as_list(self):
        '''returns [left, top, right, bottom, width, height]'''
        w = self.width
        h = self.height
        return [self.left, self.top, self.right, self.bottom, w, h]


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


def void():
    '''a function that does nothing'''


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


def count_files_inside(directory="."):
    '''return the number of files (does not consider folders) in directory'''
    path, dirs, files = next(os.walk(directory))

    return len(files)


def get_file_with(extension, at_path):
    '''return the the file at the given path with the
    given extension (if provided and without the dot)'''
    folder, file = os.path.split(at_path)

    # find the right file
    if extension is not None:
        ext = file.split(".")[-1]
        file = file.replace(ext, extension)

    path = os.path.join(folder, file)

    return open(path, "r")


def get_associated_image(path):
    '''returns the path of the image file with the same name of the
    file at the given path'''
    folder, file = os.path.split(path)
    ext = file.split(".")[-1]
    exts = ["jpg", "png", "jpeg"]

    for e in exts:
        _file = file.replace(ext, e)
        _path = os.path.join(folder, _file)

        if os.path.isfile(_path):
            return _path


def file_iter(dir=".", ext=None):
    '''generate all files inside the given dir with the specified extension,
    if [ext] is None. It generates all files'''
    for path, dirs, files in os.walk(dir):
        # scan every file in subfolders
        for file in files:
            if (ext is None) or file.endswith(ext):
                yield os.path.join(path, file), file


# -----------------------------------------------------------------------------
# -- IMAGE UTILS
# -----------------------------------------------------------------------------
def crop_image(image, roi, ensure=False):
    '''returns the cropped image according to a region-of-interest, optionally
    force the region to be inside the image'''
    assert(type(roi) is Region)

    if ensure is True:
        h, w = image.shape[:2]
        roi = roi.ensure(w, h)

    l, t, r, b = roi.as_list()[:4]

    return image[t:b, l:r].copy()


def rotate_image(image, angle=0, center=None):
    '''rotate the given image, by default it rotates around the center'''
    h, w = image.shape[:2]

    if center is None:
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


def show_image(image, window="window", onSkip=void, onQuit=void):
    '''show image easly, support skip(ESC) and quit(Q) events with custom
    callbacks'''
    while True:
        cv2.imshow(window, image)
        key = cv2.waitKey(20) & 0Xff

        if key == Keys.ESC:
            return onSkip()

        elif key == Keys.Q:
            onQuit()
            cv2.destroyAllWindows()
            return exit()


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


def detect_faces(image, detector="cnn"):
    '''detect every face inside image. By default it uses the cnn detector,
       pass "dlib" to use the dlib frontal face detector.
       Returns a list of tuple\\rectangles: (top, left, right, bottom)'''

    # detect faces with dlib.frontal_face_detector
    if detector == "dlib":
        # load detector if needed
        global face_det

        if face_det is None:
            face_det = dlib.get_frontal_face_detector()

        dets = face_det(image, 1)
        boxes = []

        for d in dets:
            boxes.append(Region.dlib(d))

        return boxes

    # detect faces with opencv caffe cnn detector
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
            boxes.append(Region.tuple(box.astype("int")))

    return boxes


def faces_inside(directory="", scale_factor=1, remove_image=False):
    '''generate all faces within a given directory and optionally remove the
    images'''

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

            if remove_image is True:
                os.remove(os.path.join(path, file))


def is_face_aligned_with_landmarks(rect, points):
    '''decide if the face is aligned with the landmarks by comparing either
    boundin-boxes.'''
    print("warning [is_face_aligned_with_landmarks]: code changed!")
    assert(type(rect) is Region)

    region = points_region(points)
    # a, b, c, d = region
    # t, l, r, b = rect

    # get intersection rect
    # r1 = dlib.rectangle(l, t, r, b)
    # r2 = dlib.rectangle(b, a, c, d)
    r1 = rect.as_dlib()
    r2 = region.as_dlib()
    r3 = r1.intersect(r2)

    # area
    a1 = r1.area()
    a2 = r3.area()

    print(r1)
    print(r2)
    print(a1, r3, a2)

    if a1 == 0 or a2 == 0:
        return False
    print(a1 / a2)
    return (a1 / a2) >= 0.55


def ibug_dataset(folder="."):
    '''iterate through the ibug dataset-like generating: image, landmarks'''
    for fpath, fname in file_iter(folder, ext=".pts"):
        # retrive image
        ipath = get_associated_image(fpath)
        image = cv2.imread(ipath)

        # read landmarks
        marks = []
        with open(fpath, "r") as ann:
            lines = ann.readlines()[3:-2]

            for line in lines:
                x, y = line.split()[:2]
                x = int(x.split(".")[0])
                y = int(y.split(".")[0])
                marks.append((x, y))

        yield image, marks, fpath


def prominent_face(faces):
    '''returns the bigger (prominent) face (Region obj)'''
    max_area = -2 ^ 31
    max_face = None

    for face in faces:
        assert(type(face) is Region)
        area = face.area()

        if area > max_area:
            max_area = area
            max_face = face

    if max_face is None:
        print("face none")

    return max_face


# -----------------------------------------------------------------------------
# -- LANDMARK UTILS
# -----------------------------------------------------------------------------
def load_shape_predictor(model_path):
    '''load the dlib shape predictor model'''
    global shape_predictor
    shape_predictor = dlib.shape_predictor(model_path)


def detect_landmarks(face, region):
    '''detect landmarks for the given face and face Region,
    returns an array of tuples (x, y)'''
    assert(type(region) is Region)

    rect = region.as_dlib()
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


def points_region(pts):
    '''return a Region obj that contains all the given points'''
    assert(len(pts) > 1)

    left = min(pts, key=lambda p: p[0])[0]
    right = max(pts, key=lambda p: p[0])[0]
    top = min(pts, key=lambda p: p[1])[1]
    bottom = max(pts, key=lambda p: p[1])[1]

    return Region.tuple((left, top, right, bottom))


# -----------------------------------------------------------------------------
# -- DRAWING UTILS
# -----------------------------------------------------------------------------
def draw_rect(image, rect, color=(128, 0, 128), thickness=2):
    '''draw the given rectangle (Region obj) on image'''
    assert(type(rect) is Region)

    cv2.rectangle(image, rect.tl(), rect.br(), color, thickness)


def draw_rects(image, regions, color=(128, 0, 128), thickness=2):
    '''draws an array of Region obj'''
    for r in regions:
        cv2.rectangle(image, r.tl(), r.br(), color, thickness)


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
