# opencv-face-detection-play README

This follows the article, [Creating a face detection API with Python and OpenCV (in just 5 minutes)](https://www.pyimagesearch.com/2015/05/11/creating-a-face-detection-api-with-python-and-opencv-in-just-5-minutes/),
and updates it to use [apistar](https://docs.apistar.com/#quickstart) in place of django.

In addition to face detection this attempts a simple ratio based hashing method in a (failed) attempt at finding a *simple*
person hash method to uniquely generate a hash to identify a unique person.

## dlib installation on macos

This follows the `dlib` installation steps defined at:
https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

> NOTE: this assumes you have [brew](https://brew.sh/)

0. Install `cmake` and `boost`:
    ```
    brew install cmake
    brew install boost
    brew install opencv
    brew install dlib
    ```

1. Clone the `dlib` repository:
    ```
    git clone https://github.com/davisking/dlib.git
    ```

2. Install the python extension:

    ```
    # from your virtualenv
    # from the cloned repository containing the `setup.py` file
    (venv)$ python setup.py install
    ```
