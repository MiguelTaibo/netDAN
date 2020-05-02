import pandas as pd
import numpy as np
import argparse
import cv2
import shutil, os
from PIL import Image
from imutils import face_utils
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    out_path = args['out_path']
    IMG_SIZE = args['longSize']

    columns = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks',
               'expression', 'valence', 'arousal']

    data = pd.read_csv(args['csv_file'])
    errors = pd.DataFrame(None, columns=columns)
    data1 = data.loc[data['expression'] == 1]
    data2 = data.loc[data['expression'] == 2]
    datares = pd.DataFrame(None, columns=columns)

    num_imgs = 25000
    i = 0
    for index, row in tqdm(data1.iterrows()):
        if (i==num_imgs):
            break
        try:
            ### LOAD IMAGE DATA
            img_ori = cv2.imread(dataroot + row['subDirectory_filePath'])
            (width, height, _) = img_ori.shape

            ### LOAD LANDMARKS
            landmarks = []
            for lm in row['facial_landmarks'].split(';'):
                landmarks.append(float(lm))
            landmarks = np.array(landmarks).reshape((68, 2))

            for i in range(68):
                landmarks[i, 0] = min(landmarks[i, 0] / width * IMG_SIZE, 223)
                landmarks[i, 1] = min(landmarks[i, 1] / height * IMG_SIZE, 233)

            ## CREATE LABEL
            background = np.zeros((IMG_SIZE, IMG_SIZE))
            colors = []
            [colors.append((255, 255, 255)) for roi in range(8)]
            output = face_utils.visualize_facial_landmarks(background, landmarks.astype('int64'), colors=colors,
                                                           alpha=1)

            cv2.imwrite(os.path.join(out_path, row['subDirectory_filePath']), output)
            datares = datares.append(row)
        except:
            import pdb
            pdb.set_trace()
            print("Error en", row['subDirectory_filePath'])
            errors = errors.append(row)

    i = 0
    for index, row in tqdm(data2.iterrows()):
        if (i==num_imgs):
            break
        try:
            ### LOAD IMAGE DATA
            img_ori = cv2.imread(dataroot + row['subDirectory_filePath'])
            (width, height, _) = img_ori.shape

            ### LOAD LANDMARKS
            landmarks = []
            for lm in row['facial_landmarks'].split(';'):
                landmarks.append(float(lm))
                landmarks = np.array(landmarks).reshape((68, 2))

            for i in range(68):
                landmarks[i, 0] = min(landmarks[i, 0] / width * IMG_SIZE, 223)
                landmarks[i, 1] = min(landmarks[i, 1] / height * IMG_SIZE, 233)

            ## CREATE LABEL
            background = np.zeros((IMG_SIZE, IMG_SIZE))
            colors = []
            [colors.append((255, 255, 255)) for roi in range(8)]
            output = face_utils.visualize_facial_landmarks(background, landmarks.astype('int64'), colors=colors,
                                                           alpha=1)

            cv2.imwrite(os.path.join(out_path, row['subDirectory_filePath']), output)
            datares = datares.append(row)
        except:
            print("Error en", row['subDirectory_filePath'])
            errors = errors.append(row)

    print(len(errors),"errores")
    errors.to_csv('errors.csv', columns=columns)
    datares.to_csv('data.csv', columns=columns)