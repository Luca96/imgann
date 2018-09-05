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
import os
import cv2
import dlib
import utils
import numpy as np

from xml import Xml
from utils import show_properly
from utils import Keys, Colors
from utils import Annotation, Region


def region(face):
    '''return a region (0, 0, w, h) for the cropped face'''
    return Region(0, 0, face.shape[1], face.shape[0])


def my_noise(image):
    '''add simulated camera noise to the given image'''
    out = image.copy()
    h, w = image.shape[:2]
    amount = int((h * w) * (3 / 100))

    for i in range(0, amount):
        row = np.random.randint(0, h)
        col = np.random.randint(0, w)
        b = np.random.randint(0, 128)
        g = np.random.randint(0, 180)
        r = np.random.randint(0, 256)
        out[row][col] = [b, g, r]

    return out


def augment_data(image, face, landmarks):
    '''produce 2 type of augumentation: rotation and noising,
    returns an array of 3 tuple (image, landmarks)'''
    assert(type(face) is Region)

    h, w = image.shape[:2]
    angle = 30
    pivot = face.center()

    # 30 degree rotation
    A = utils.rotate_image(image, angle, pivot)
    B = utils.rotate_landmarks(landmarks, pivot, angle)
    R1 = utils.points_region(B)

    # -30 degree rotatation
    C = utils.rotate_image(image, -angle, pivot)
    D = utils.rotate_landmarks(landmarks, pivot, -angle)
    R2 = utils.points_region(D)

    # mirroring
    E = utils.flip_image(image)
    F = utils.naive_flip_landmarks(landmarks, w)
    R3 = face.flip(width=w)

    return [(A, B, R1), (C, D, R2), (E, F, R3)]


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
    i = 0

    for annotation, path in Annotation.inside(folder):
        xml.append(path, [annotation.box], [annotation.points])
        i = i + 1
        print("{} writing: {}".format(i, path))

    xml.close()


def train_model(name, xml):
    '''train and return a dlib shape predictor.
    Training options are:
    cascade_depth:
        The number of cascades created to train the model with. The number of
        trees in the model is = cascade_depth * num_trees_per_cascade_level.
        > default: 10

    feature_pool_region_padding:
        Size of region within which to sample features for the feature pool,
        e.g a padding of 0.5 would cause the algorithm to sample pixels from
        a box that was 2x2 pixels
        > default: 0

    feature_pool_size:
        Number of pixels used to generate features for the random trees at
        each cascade. So in general larger settings of this parameter give
        better accuracy but make the algorithm run slower.
        > default: 400

    lambda_param:
        Controls how tight the feature sampling should be. Lower values enforce
        closer features. To decide how to split nodes in the regression tree
        the algorithm looks at pairs of pixels in the image. These pixel pairs
        are sampled randomly but with a preference for selecting pixels that
        are near each other.
        > default: 0.1

    nu:
        The regularization parameter.
        Larger values of this parameter will cause the algorithm to fit
        the training data better but may also cause overfitting.
        The value must be 0 < nu <= 1.
        > default: 0.1

    num_test_splits:
        Number of split features at each node to sample.
        Larger values of this parameter will usually give more accurate
        outputs but take longer to train.
        > default: 20

    num_trees_per_cascade_level:
        The number of trees created for each cascade.
        > default: 500

    oversampling_amount:
        The number of randomly selected initial starting points sampled
        for each training example. This parameter controls the number of
        randomly selected deformation applied to the training data.
        So the bigger this parameter the better (excepting that larger
        values make training take longer).
        > default: 20

    oversampling_translation_jitter:
        When generating the get_oversampling_amount() factor of extra training
        samples you can also jitter the bounding box by adding random small
        translational shifts. For instance, if you set it to 0.1 then it would
        randomly translate the bounding boxes by between 0% and 10% their
        width and height in the x and y directions respectively. Doing this is
        essentially equivalent to randomly jittering the bounding boxes in the
        training data. So doing this kind of jittering can help make the
        learned model more robust against slightly misplaced bounding boxes.
        > default: 0

    random_seed:
        The random seed used by the internal random number generator
        > default: ""

    tree_depth:
        The depth of the trees used in each cascade.
        There are pow(2, get_tree_depth()) leaves in each tree.
        > default: 4
    '''
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 3  # 4
    options.nu = 0.1
    options.num_threads = 8
    options.cascade_depth = 12  # 15
    options.be_verbose = True

    options.feature_pool_size = 400 + 20  # 400
    options.num_test_splits = 20 + 5
    options.oversampling_amount = 20
    # options.oversampling_translation_jitter = 0.1

    dlib.train_shape_predictor(xml, name, options)


def test(folder="testset", model="dlib/shape_predictor_68_face_landmarks.dat"):
    utils.init_face_detector(True, 150)
    utils.load_shape_predictor(model)
    my_sp = utils.shape_predictor
    dlib_sp = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")

    for face, r in utils.faces_inside(folder):
        box = region(face)

        utils.shape_predictor = my_sp
        lmarks0 = utils.detect_landmarks(face, box)

        utils.shape_predictor = dlib_spq
        lmarks1 = utils.detect_landmarks(face, box)

        # draw results
        utils.draw_points(face, lmarks1, color=Colors.green)
        utils.draw_points(face, lmarks0, color=Colors.red)
        utils.show_image(show_properly(face))


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


def another_test():
    utils.load_shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")

    for img, p in utils.images_inside("uffa"):
        fast = cv2.FastFeatureDetector_create()
        # kp = fast.detect(img, None)

        # draws:
        face = utils.detect_faces(img, detector="dlib")[0]
        # utils.draw_rect(img, face, color=Colors.green)

        pts = utils.detect_landmarks(img, face)
        pts = [pts[0], pts[3], pts[6], pts[10], pts[20], pts[22], pts[35]]
        # utils.draw_points(img, pts)

        # img = cv2.drawKeypoints(img, kp, None, color=Colors.cyan)

        keypoints = []

        for p in pts:
            roi = Region(p[0] - 10, p[1] - 10, 20, 20)
            patch = utils.crop_image(img, roi)
            keypoints.append(fast.detect(patch, None))

        # for kp in keypoints[2]:
        #     print(kp)
        #     # img = cv2.drawKeypoints(img, kp, None, color=Colors.cyan)

        for p in pts:
            for kp in keypoints:
                for k in kp:
                    x = int(k.pt[0] + p[0])
                    y = int(k.pt[1] + p[1])
                    utils.draw_point(img, x, y, radius=1)

        while True:
            cv2.imshow("window", show_properly(img))
            key = cv2.waitKey(20) & 0Xff

            if key == Keys.ESC:
                break


def adjust_landmarks(region, pts):
    '''adjust the position of the landmarks to be inside region'''
    x0, y0 = region.tl()
    new_pts = []

    for p in pts:
        x = p[0] - x0
        y = p[1] - y0
        new_pts.append((x, y))

    return new_pts


def build_trainset_auto(src="dataset", dst="trainset", debug=False):
    '''build a trainset automatically from an ibug-like dataset,
    the images are taken from [src] folder and saved to [dst] folder'''
    utils.init_face_detector(True, 150)
    qualiy = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

    # file count for naming
    count = int(utils.count_files_inside(dst) / 8)

    for img, lmarks, path in utils.ibug_dataset(src):
        h, w = img.shape[:2]

        # crop a bigger region around landmarks
        region = utils.points_region(lmarks)
        scaled = region.scale(1.8, 1.8).ensure(w, h)

        img = utils.crop_image(img, scaled)

        # detect faces
        face = utils.prominent_face(utils.detect_faces(img))

        # if cnn fails try with dlib
        if face is None:
            faces = utils.detect_faces(img, detector="dlib")

            # ..if dlib fails take the region around landmarks
            if face is None:
                face = region.copy()
            else:
                face = utils.prominent_face(faces)

        # edit landmarks according to scaled region
        lmarks = adjust_landmarks(scaled, lmarks)

        # augumentations
        i = 0
        for image, landmarks, box in augment_data(img, face, lmarks):
            i = i + 1

            if debug:
                utils.draw_rect(image, box, color=Colors.yellow)
                utils.draw_points(image, landmarks, color=Colors.purple)
                name = f"image{i}"
                utils.show_image(show_properly(image), window=name)
            else:
                # save annotation and image
                ipath = os.path.join(dst, f"face{count}_{i}.jpg")
                apath = os.path.join(dst, f"face{count}_{i}.ann")
                cv2.imwrite(ipath, image, qualiy)
                Annotation(apath, box.as_list()[:4], landmarks).save()

        if debug:
            utils.draw_rect(img, face, color=Colors.red)
            utils.draw_points(img, lmarks, color=Colors.green)
            utils.show_image(show_properly(img))
        else:
            # save image and annotation
            ipath = os.path.join(dst, f"face{count}.jpg")
            apath = os.path.join(dst, f"face{count}.ann")
            cv2.imwrite(ipath, img, qualiy)
            Annotation(apath, face.as_list()[:4], lmarks).save()

        count = count + 1

        # info
        print("{} processed: {}\r".format(count, ipath))


if __name__ == '__main__':
    # build_trainset()
    # test(folder="images", model="sp_68_fast.dat")
    # xml_file = "new_sp_68.xml"
    # generate_training_xml("boosted_ibug.xml")
    # train_model("sp_68_mini.dat", "labels_ibug_300W_train.xml")
    # another_test()
    build_trainset_auto(src="dataset", debug=False)
