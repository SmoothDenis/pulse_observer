# Pulse Observer

## Description

Compute a person's pulse via webcam in real-time by tracking tiny changes in face coloration due to blood flow. For best results, try to minimize movement.
Uses Python, OpenCV, NumPy, SciPy, and Dlib.

## Requirements

-   Python 3.10.4
-   OpenCV-Python 4.5.5.64
-   NumPy 1.21.6
-   SciPy 1.8.0
-   Dlib 19.23.1
-   Download [Dlib's pre-trained predictor model for facial landmarks](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2) and put it in the same directory as _pulse_observer.py_.

## Compatibility

-   This program has **only** been tested on Windows 10. It may or may not work on Linux or Mac OS.
-   It will only correct work with Python 3+. It will **not** work with Python <3. This version of program adapt for use with IP-camera (RTSP) and more DEBUG function for learn how work this script.
-   Tested with the following webcams:
    -   Logitech C270
    -   Logitech C920
    -   Asus laptop
    -   Lenovo Laptop
    -   ESP32-CAM with OV2460

## Author

Kevin Perry
(kevinperry@gatech.edu)
