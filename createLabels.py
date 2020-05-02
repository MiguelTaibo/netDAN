import pandas as pd
import numpy as np
import argparse
import cv2
import shutil, os
from PIL import Image
from imutils import face_utils
import matplotlib.pyplot as plt


def plot_img(img, lms, i, f):
    plt.imshow(img)
    plt.scatter(lms[i:f, 0], lms[i:f, 1], marker="x", color="red", s=20)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action='store_true')
    ap.add_argument("--dataroot", type=str, default='datasets/')
    ap.add_argument("--csv_file", type=str, default='training.csv')
    ap.add_argument("--out_path", type=str, default='results/')
    ap.add_argument("--longSize", type=int, default=128)

    args = vars(ap.parse_args())

    quiet = args['quiet']
    dataroot = args['dataroot']
    data = pd.read_csv(args['csv_file'])
    out_path = args['out_path']
    IMG_SIZE = args['longSize']

    columns = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']
    data = pd.read_csv('training.csv')

    for index, row in data.iterrows():
        ### LOAD IMAGE DATA
        img_ori = cv2.imread(dataroot+row['subDirectory_filePath'] )
        (width, height, _) = img_ori.shape

        ### LOAD LANDMARKS
        landmarks = []
        for lm in row['facial_landmarks'].split(';'):
            landmarks.append(float(lm))
        landmarks = np.array(landmarks).reshape((68,2))

        # plot_img(img_ori,landmarks,48,68)
        # plot_img(img_ori, landmarks, 60, 68)
        #
        # plot_img(img_ori, landmarks, 17, 22)
        # plot_img(img_ori, landmarks, 22, 27)
        # plot_img(img_ori, landmarks, 36, 42)
        # plot_img(img_ori, landmarks, 42, 48)
        #
        # plot_img(img_ori, landmarks, 27, 36)
        # plot_img(img_ori, landmarks, 0, 17)
        # exit()

        for i in range(68):
            landmarks[i, 0] = min(landmarks[i, 0] / width * IMG_SIZE,223)
            landmarks[i, 1] = min(landmarks[i, 1] / height * IMG_SIZE,233)


        ## CREATE LABEL
        background = np.zeros((IMG_SIZE,IMG_SIZE))
        colors = []
        [colors.append((255, 255, 255)) for roi in range(8)]
        output = face_utils.visualize_facial_landmarks(background, landmarks.astype('int64'), colors=colors, alpha=1)

        picture_name = row['subDirectory_filePath'].split("/")[-1]
        cv2.imwrite(os.path.join(out_path, picture_name), output)