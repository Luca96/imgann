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
# -- Image Annotator tool - imgann
# -----------------------------------------------------------------------------
import os
import cv2
import utils
import numpy as np

from xml import Xml
from utils import is_image, is_mirrored

# global variables:
image = None
stack = []
boxes = []
points = []


def draw_circle(image, x, y):
    '''save the drawn point and backup image'''
    global stack, points

    stack.append(np.copy(image))
    points.append((x, y))

    cv2.circle(image, (x, y), radius=3, color=(0, 255, 255), thickness=-1)

    # point log
    print(f"\r> points: {len(points)}", end='\r')


def restore_image():
    '''return the last image available into the stack
    and remove the last inserted points'''
    global stack, points
    size = len(stack)

    if size > 0:
        points.pop()
        return stack.pop()
    else:
        return image


def mouse_callback(event, x, y, flags, param):
    global image

    if event == cv2.EVENT_LBUTTONUP:
        draw_circle(image, x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        image = restore_image()


def add_entry(out, path, boxes, points, mirror):
    '''add the image, points, boxes to the output file'''
    w = image.shape[1]

    # handle xml file
    if isinstance(out, Xml):
        out.append(path, boxes, points)

        if mirror:
            path = utils.mirror_image(stack[0], path)

            points = [(w - x, y) for x, y in points]
            out.append(path, boxes, points)
    else:
        out.write(f'{path}\n')

        for x, y in points:
            out.write(f'{x} {y}')
            out.write("\n")

        if mirror:
            path = utils.mirror_image(stack[0], path)

            out.write(f'{path}\n')

            for x, y in points:
                out.write(f'{w - x} {y}')
                out.write("\n")


def main():
    global image

    # getting arguments from cli
    args = utils.get_arguments()
    out = utils.open_file(args)
    img_dir = args["dir"]
    mirror_points = args["mirror"]

    for file in os.listdir(img_dir):
        # consider only images
        if not is_image(file):
            continue

        # avoid mirrored images
        if is_mirrored(file):
            continue

        # loading image:
        path = os.path.join(img_dir, file)
        image = cv2.imread(path)

        # clear: stack, boxes and points
        stack.clear()
        boxes.clear()
        points.clear()

        stack.append(image)

        # create a window with disabled menu when right-clicking with the mouse
        window = file
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

        # mouse callback to window
        cv2.setMouseCallback(window, mouse_callback)

        # showing image until esc is pressed
        while (1):
            cv2.imshow(window, image)

            if cv2.waitKey(20) & 0xFF == 27:
                break

        # clear point log
        print()

        # write annotations
        add_entry(out, path, boxes, points, mirror_points)

        # close window
        cv2.destroyAllWindows()

    out.close()


if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------
