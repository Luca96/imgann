# -----------------------------------------------------------------------------
# MIT License

# Copyright (c) 2018 Luca Anzalone

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

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
import numpy as np
import argparse

# global variables:
image = None
stack = []
points = []


def draw_circle(image, x, y):
    # save the drawn point and backup image
    global stack, points

    stack.append(np.copy(image))
    points.append((x, y))

    cv2.circle(image, (x, y), radius=3, color=(255, 255, 0))


def restore_image():
    # return the last image available into the stack
    # and remove the last inserted points
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


def get_arguments():
    # building and parsing command line arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--file", required=True,
                    help="output file for annotations")

    ap.add_argument("-d", "--dir", required=True,
                    help="image directory")

    ap.add_argument("-a", "--append", action="store_const", const='a',
                    help="open the output file in append mode")

    return vars(ap.parse_args())


def main():
    global image

    # getting arguments from cli
    args = get_arguments()
    out = open(args["file"], args["append"] or "w")
    img_dir = args["dir"]

    # creates a window with disabled menu when right-clicking with the mouse
    window = 'window'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    for file in os.listdir(img_dir):
        # loading image:
        path = os.path.join(img_dir, file)
        image = cv2.imread(path)

        # mouse callback to window
        cv2.setMouseCallback(window, mouse_callback)

        # showing image until esc is pressed
        while (1):
            cv2.imshow(window, image)

            if cv2.waitKey(20) & 0xFF == 27:
                break

        # write annotated points
        out.write(f'{path}\n')

        for x, y in points:
            out.write(f'{x} {y}')
            out.write("\n")

    cv2.destroyAllWindows()
    out.close()


if __name__ == '__main__':
    main()

# -----------------------------------------------------------------------------
