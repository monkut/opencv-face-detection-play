import os
import cv2
import pytest
from pathlib import Path
from apistar.test import TestClient
from face_detector_api import _extract_face, _detect_facial_landmarks, _hash_facial_landmarks, _detect_faces, app

TEST_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'data')


@pytest.fixture
def data_directory(request):
    return Path(TEST_DATA_DIRECTORY)  # can use .joinpath('file')


def test__detect_faces(data_directory):
    sample_image = data_directory.joinpath('Charles_Taylor', 'Charles_Taylor_0002.jpg')
    result = None
    with open(sample_image, 'rb') as data:
        result = _detect_faces(data.read())
    assert result
    assert result['faces_count'] == 1
    assert len(result['face_locations']) == 1


def test__detect_facial_landmarks(data_directory):
    sample_image = data_directory.joinpath('Toshihiko_Fukui', 'Toshihiko_Fukui_0001.jpg')
    result = None
    with open(sample_image, 'rb') as data:
        binary_image_data = data.read()
        result = _detect_faces(binary_image_data)

    assert len(result['face_locations']) >= 1
    for bbox in result['face_locations']:
        cropped_face_image = _extract_face(binary_image_data, bbox)
        assert cropped_face_image.any()

        # get facial landmarks
        landmarks = _detect_facial_landmarks(cropped_face_image)
        assert landmarks
        expected_location_ids = (
            'mouth',
            'right_eyebrow',
            'left_eyebrow',
            'right_eye',
            'left_eye',
            'nose',
            'jaw',
        )
        for expected_location_id in expected_location_ids:
            assert expected_location_id in landmarks

        expected_indexes = list(range(68))
        for expected_index in expected_indexes:
            actual_indexes = []
            for location_id, positions in landmarks.items():
                for landmark_index in positions.keys():
                    actual_indexes.append(landmark_index)
            assert expected_index in actual_indexes


def test__hash_facial_landmarks(data_directory):
    for i in (1, 2, 3, 5, 6, 7, 8 , 9):
        sample_image = data_directory.joinpath('Charles_Taylor', f'Charles_Taylor_000{i}.jpg')
        result = None
        with open(sample_image, 'rb') as data:
            binary_image_data = data.read()
            result = _detect_faces(binary_image_data)

        assert len(result['face_locations']) >= 1
        for bbox in result['face_locations']:
            cropped_face_image = _extract_face(binary_image_data, bbox)
            assert cropped_face_image.any()

            # get facial landmarks
            landmarks = _detect_facial_landmarks(cropped_face_image)
            hash_result = _hash_facial_landmarks(landmarks)
            assert hash_result
            print(hash_result)
    assert False


def test__extract_face(data_directory):
    sample_image = data_directory.joinpath('Toshihiko_Fukui', 'Toshihiko_Fukui_0001.jpg')
    result = None
    with open(sample_image, 'rb') as data:
        binary_image_data = data.read()
        result = _detect_faces(binary_image_data)

    assert len(result['face_locations']) >= 1
    for bbox in result['face_locations']:
        cropped_face_image = _extract_face(binary_image_data, bbox)
        assert cropped_face_image.any()


def test_detect_and_analyze_faces():
    client = TestClient(app)
    response = client.get('http://localhost/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Welcome to API Star!'}