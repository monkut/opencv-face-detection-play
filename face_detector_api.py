import os
import cv2
import dlib
import numpy as np
from math import sqrt
from hashlib import sha1
from collections import defaultdict
from apistar import App, Route, http

# define the path to the face detector
# pre-trained models available at:
# https://github.com/opencv/opencv/tree/master/data/haarcascades
base_path = os.path.abspath(os.path.dirname(__file__))
HAAR_FACE_CASCADE_MODEL_FILEPATH = f'{base_path}/models/cascades/haarcascade_frontalface_alt.xml'
LBP_FACE_CASCADE_MODEL_FILEPATH = f'{base_path}/models/cascades/lbpcascade_frontalface_improved.xml'
FACE_LANDMARK_SHAPE_PREDICTOR_FILEPATH = f'{base_path}/models/shapes/shape_predictor_68_face_landmarks.dat'

# Pre-trained facial landmark detection model at:
# https://github.com/spmallick/GSOC2017/blob/master/data/lbfmodel.yaml


def _resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    # from:
    # https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    h, w = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = width, int(h * r)

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def _detect_faces(binary_image_data: bytes) -> dict:
    """
    Convert raw image binary to image for detection and analysis of where the faces are in the image
    :param binary_fileobject:
    :return:
    """
    image_array = np.asarray(bytearray(binary_image_data), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the face cascade detector
    cascade = cv2.CascadeClassifier(HAAR_FACE_CASCADE_MODEL_FILEPATH)

    # detect faces in the image
    face_location_bboxes = cascade.detectMultiScale(image,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
    return {'faces_count': len(face_location_bboxes),
            'face_locations': face_location_bboxes}


def _detect_facial_landmarks(face_image):
    detector = dlib.get_frontal_face_detector() # Currently using openCV method
    predictor = dlib.shape_predictor(FACE_LANDMARK_SHAPE_PREDICTOR_FILEPATH)
    resized_image = _resize_image(face_image, width=500)
    #grayscale_resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    feature_indexes = (
        ('mouth', 48, 68),
        ('right_eyebrow', 17, 22),
        ('left_eyebrow', 22, 27),
        ('right_eye', 36, 42),
        ('left_eye', 42, 48),
        ('nose', 27, 36),
        ('jaw', 0, 17),
    )

    facial_landmarks = defaultdict(dict)
    for dlib_bbox_object in detector(resized_image, 1):
        # get the facial landmarks
        shape = predictor(resized_image, dlib_bbox_object)
        for idx, point in enumerate(shape.parts()):
            for location_id, start_idx, end_idx in feature_indexes:
                if start_idx <= idx < end_idx:
                    facial_landmarks[location_id][idx] = point.x, point.y
                    break
        break  # expect only 1 face
    return facial_landmarks


def _hash_facial_landmarks(facial_landmarks):
    """
    facial landmarks are extracted via dlib and results in a set of fixed label positions.

    Refer to:
    https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/

    :param facial_landmarks:
    :return:
    """
    key_measure = (
        ('nose', 27), ('nose', 33),
    )
    key_locationid_indexes = (
        # indexes are 0 start 0 - 35
        (('nose', 27), ('left_eye', 42)),
        (('nose', 27), ('right_eye', 39)),
        (('nose', 27), ('nose', 35)),
        (('nose', 27), ('nose', 31)),
        (('nose', 33), ('jaw', 8)),
    )
    x_1, y_1 = facial_landmarks['nose'][28]
    x_2, y_2 = facial_landmarks['nose'][34]
    key_length = sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    # get ratios
    hash = sha1()
    for (location_1, index_1), (location_2, index_2) in key_locationid_indexes:
        x_1, y_1 = facial_landmarks[location_1][index_1]
        x_2, y_2 = facial_landmarks[location_2][index_2]
        compare_length = sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
        ratio = int(round(compare_length/key_length, 5) * 10000)
        print(ratio)
        hash.update(str(ratio).encode('utf8'))
    return hash.hexdigest()
_hash_facial_landmarks.version = 0.1


def _extract_face(binary_image_data, bbox, pixel_buffer_multiplier: float=.80):
    """
    From the given image extract/crop out the face via the bbox
    :param image_binary_data:
    :param bbox:
    :return:
    """
    origin_x, origin_y, x, y = bbox

    # apply a buffer to the image in an attempt to include full facial features (chin, top of head)
    pixel_height = y - origin_y
    pixel_width = x - origin_x
    height_pixel_buffer = pixel_height * pixel_buffer_multiplier
    width_pixel_buffer = pixel_width * pixel_buffer_multiplier

    origin_x = int(origin_x - height_pixel_buffer)
    origin_y = int(origin_y - width_pixel_buffer)
    x = int(x + (width_pixel_buffer * 2))
    y = int(y + (height_pixel_buffer * 2))

    image_array = np.asarray(bytearray(binary_image_data), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cropped_image = image[origin_y: origin_y + y, origin_x: origin_x + x]
    return cropped_image


def detect_and_analyze_faces(request: http.Request) -> dict:
    """
    Detects faces in a POSTed image
    :param request:
    :return:
        {
            'face_count': NUMBER_OF_FACES_IN_IMAGE,
            'locations': [BBOX_OF_FACE, ...]
        }

    """
    image_binary_data = request.body.read()
    face_location_information = _detect_faces(image_binary_data)

    face_hashes = []
    for face_location_bbox in face_location_information['face_locations']:
        face_image = _extract_face(image_binary_data, face_location_bbox)
        facial_landmarks = _detect_facial_landmarks(face_image)
        face_hash = _hash_facial_landmarks(facial_landmarks)
        face_information = {
            'bbox': face_location_bbox,
            'facial_landmarks': facial_landmarks,
            'hash': face_hash,
            'version': _hash_facial_landmarks.version
        }
        face_hashes.append(face_information)

    return face_hashes


routes = [
    Route('/', method='POST', handler=detect_and_analyze_faces),
]

app = App(routes=routes)


if __name__ == '__main__':
    app.serve('127.0.0.1', 5000, debug=True)