from imutils import face_utils
import numpy as np
import glob
import argparse
import imutils
import dlib
import cv2
from PIL import Image

import os
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", help="path to facial landmark predictor", default = "detect-face-parts/detect_face_parts.py")
ap.add_argument("-f", "--folder", required=True, help="path to folder of images")
ap.add_argument("-o", "--output", required=True, help="Output path")
args = vars(ap.parse_args())
print(args);

Image = []
Landmark = []
Emotion = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])




for filename in glob.glob(args["folder"]+"/*.png"):
    print(filename)
    image = cv2.imread(filename)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        Landmark.append((shape.flatten()))
        image = imutils.resize(image, width=224)
        Image.append(np.asarray(image))

Landmark = np.array(Landmark)
Image = np.array(Image)
import pdb
pdb.set_trace()
