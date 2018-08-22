# -----------------------------------------------------------------------------
# -- Script to train, prepare and meaure error among data
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
import cv2
import dlib
import utils

from xml import Xml
from utils import Keys, Colors
from utils import Annotation


def region(face):
    '''return a region (0, 0, w, h) for the cropped face'''
    return (0, 0, face.shape[1], face.shape[0])


def debug_faces_founded(image):
    bbox = utils.detect_faces(image)

    for box in bbox:
        utils.draw_rect(image, box)


def show_properly(image, size=600):
    '''resize too large images'''
    h, w = image.shape[:2]
    ratio = h / w

    fx = 1 / (w * ratio / size)
    fy = 1 / (h / size)

    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)


def augment_data(image, region, landmarks):
    '''produce 2 type of augumentation: rotation and mirroring,
    returns an array of 3 tuple (image, landmarks)'''
    angle = 30
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # 30 degree rotation
    A = utils.rotate_image(image, angle)
    B = utils.rotate_landmarks(landmarks, center, angle)

    # -30 degree rotatation
    C = utils.rotate_image(image, -angle)
    D = utils.rotate_landmarks(landmarks, center, -angle)

    # mirroring
    E = utils.flip_image(image)
    F = utils.detect_landmarks(E, region)

    return [(A, B), (C, D), (E, F)]


def build_trainset(input_dir="data", output_dir="trainset", win_size=321):
    '''scan the input folder and put every image with annotations
    output folder'''
    utils.init_face_detector(True, win_size)
    utils.load_shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")

    count = int(utils.count_files_inside(output_dir) / 8)
    window = "window"
    cv2.namedWindow(window)

    for face, box in utils.faces_inside(input_dir, 1, True):
        face_copy = face.copy()
        face_rect = region(face)

        # detections
        points = utils.detect_landmarks(face, face_rect)
        utils.draw_points(face, points)

        # show face
        while (1):
            h, w = face.shape[:2]

            if h > 0 and w > 0:
                cv2.imshow(window, show_properly(face))
            else:
                break

            key = cv2.waitKey(20) & 0Xff

            if key == Keys.ESC:
                break  # skip current face

            elif key == Keys.S:
                path = f"{output_dir}/face{count}"

                # save image
                cv2.imwrite(f"{path}.jpg", face_copy)

                # save annotation relative to the current face
                Annotation(f"{path}.ann", face_rect, points).save()

                # generate and save augumetations
                array = augment_data(face_copy, face_rect, points)

                for i, x in enumerate(array):
                    # save image x[0]
                    cv2.imwrite(f"{path}_{i + 1}.jpg", x[0])

                    # save annotations
                    Annotation(f"{path}_{i + 1}.ann", face_rect, x[1]).save()

                count = count + 1
                break

            elif key == Keys.Q:
                # quit program
                return cv2.destroyAllWindows()

    cv2.destroyAllWindows()


def generate_training_xml(name, folder="trainset"):
    '''create an xml file suitable for training dlib shape predictor model'''
    xml = Xml(name)

    for annotation, path in Annotation.inside(folder):
        xml.append(path, [annotation.box], [annotation.points])

    xml.close()


def train_model(name, xml):
    '''train and return a dlib shape predictor'''
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 3
    options.nu = 0.1
    options.num_threads = 8
    options.cascade_depth = 10
    options.be_verbose = True

    dlib.train_shape_predictor(xml, name, options)


def test(folder="testset", model="dlib/shape_predictor_68_face_landmarks.dat"):
    utils.init_face_detector(True, 321)
    utils.load_shape_predictor(model)

    for img in utils.images_inside(folder):
        debug_faces_founded(img)

        while (1):
            cv2.imshow("window", img)
            key = cv2.waitKey(20) & 0Xff

            if key == Keys.ESC:
                break


def test_augment():
    utils.init_face_detector(True, 321)
    utils.load_shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")

    for img in utils.images_inside("trainset"):
        points = utils.detect_landmarks(img, region(img))

        angle = 30
        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        # 30 degree rotation
        rot1 = utils.rotate_image(img, angle)
        rot_pts1 = utils.rotate_landmarks(points, center, angle)

        # -30 degree rotatation
        rot2 = utils.rotate_image(img, -angle)
        rot_pts2 = utils.rotate_landmarks(points, center, -angle)

        # mirroring
        mir = utils.flip_image(img)
        mir_pts = utils.detect_landmarks(mir, region(mir))

        utils.draw_points(img, points)
        utils.draw_points(rot1, rot_pts1, color=Colors.cyan)
        utils.draw_points(rot2, rot_pts2, color=Colors.purple)
        utils.draw_points(mir, mir_pts, color=Colors.green)

        while True:
            cv2.imshow("image", img)
            cv2.imshow("mirrored", mir)
            cv2.imshow("rotated30", rot1)
            cv2.imshow("rotated-30", rot2)

            key = cv2.waitKey(20) & 0Xff

            if key == Keys.ESC:
                break
            elif key == Keys.Q:
                return cv2.destroyAllWindows()


if __name__ == '__main__':
    build_trainset()
    # generate_training_xml("xmls\\test.xml")
    # train_model("test.dat", "test.xml")
