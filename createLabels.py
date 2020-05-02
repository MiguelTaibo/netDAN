import pandas as pd
import numpy as np
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
    quiet = False
    dataroot = "datasets/"
    data = pd.read_csv('training.csv')
    out_path = "results/"

    columns = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']
    data = pd.read_csv('training.csv')

    for index, row in data.iterrows():
        ### LOAD IMAGE DATA
        img_ori = cv2.imread(dataroot+row['subDirectory_filePath'] )
        print(img_ori.shape)
        (width, height, _) = img_ori.shape
        print(width, height)

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
            landmarks[i, 0] = min(landmarks[i, 0] / width * 224,223)
            landmarks[i, 1] = min(landmarks[i, 1] / height * 224,233)


        ## CREATE LABEL
        background = np.zeros((224,224))
        colors = []
        [colors.append((255, 255, 255)) for roi in range(8)]
        output = face_utils.visualize_facial_landmarks(background, landmarks.astype('int64'), colors=colors, alpha=1)

        picture_name = row['subDirectory_filePath'].split("/")[-1]
        cv2.imwrite(os.path.join(out_path, picture_name), output)