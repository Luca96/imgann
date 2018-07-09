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
import argparse

from xml import Xml


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
    name = args["file"]
    mode = args["append"] or "w"

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


def get_arguments():
    '''build and parse command line arguments'''
    ap = argparse.ArgumentParser()

    s = """ generate an output file of annotations, it the file ends with .xml
        it generates an xml file ready to be used with dlib """

    # output file
    ap.add_argument("-f", "--file", required=True, help=s)

    # imput directory
    ap.add_argument("-d", "--dir", required=True,
                    help="input directory with images")

    # (flag) append mode
    ap.add_argument("-a", "--append", action="store_const", const='a',
                    help="open the output file in append mode")

    # (flag) mirror points and images along x axis
    ap.add_argument("-m", "--mirror", action="store_true",
                    help="mirror points and images along x axis")

    return vars(ap.parse_args())
# -----------------------------------------------------------------------------
