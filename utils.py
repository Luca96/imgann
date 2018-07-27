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
import numpy as np
import argparse
from xml import Xml
from cv2.dnn import blobFromImage


# constants:
caffe_model = "caffe/res10_300x300_ssd_iter_140000.caffemodel"
caffe_proto = "caffe/deploy.prototxt"
cf_values = (104.0, 177.0, 123.0)
cf_size = (300, 300)
confidence_threshold = 0.65
state_file = ".state"

# global variables:
caffeNet = None


def mirror_image(image, path, axis=1):
    '''mirror image if no already mirrored image exists'''
    folder, file = os.path.split(path)

    if file.find("_mirror") > 0:
        # already mirrored
        return path
    else:
        # mirror image
        mirrored = cv2.flip(image, axis)

        # save image
        new_path = os.path.join(folder, file.replace(".", "_mirror."))
        cv2.imwrite(new_path, mirrored)
        return new_path


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


def is_image(file):
    '''check if the given file is an image'''
    return (file.endswith(".jpg") or file.endswith(".jpeg") or
            file.endswith(".png"))


def is_mirrored(img_file):
    '''check if the given image is the mirror of another one'''
    return img_file.find("_mirror") > 0


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


def init_face_detector(args):
    '''load the caffe-model if --auto flag is specified'''
    global caffeNet

    if args["auto"]:
        caffeNet = cv2.dnn.readNetFromCaffe(caffe_proto, caffe_model)


def detect_faces(image):
    '''detect every face inside image.
       Returns a list of tuple\\rectangles: (top, left, right, bottom)'''
    assert(caffeNet)

    # get image dimension
    (h, w) = image.shape[:2]
    np_arr = np.array([w, h, w, h])

    # convert image to blob (that do some preprocessing..)
    blob = blobFromImage(cv2.resize(image, cf_size), 1.0,
                         cf_size, cf_values)

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


def delete_image(path):
    '''delete the given image and the mirrored one if it exists'''
    folder, file = os.path.split(path)

    # delete mirror
    mirror = os.path.join(folder, file.replace(".", "_mirror."))

    if os.path.isfile(mirror):
        os.remove(mirror)

    os.remove(path)


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


def draw_rect(image, rect, color=(128, 0, 128), thickness=1):
    '''draw the given rectangle on image'''
    top_left = (rect[0], rect[1])
    right_bm = (rect[2], rect[3])
    cv2.rectangle(image, top_left, right_bm, color, thickness)
# -----------------------------------------------------------------------------
