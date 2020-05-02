import pandas as pd
import numpy as np
import shutil, os
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #TODO archivo arguments
    quiet = False
    dataroot = "datasets/"
    train_ouput = "results/AffectnetTrain_7.npz"
    validation_output = "results/AffectnetVal_7.npz"
    nb_emotions = 2
    proporcion_train_val = 0.1
    proporcion = int(1 //proporcion_train_val)
    data = pd.read_csv('training.csv')

    Images = []
    Landmarks_temp = []
    Landmarks = []
    Emotions = []

    columns = ['subDirectory_filePath','face_x','face_y','face_width','face_height','facial_landmarks','expression','valence','arousal']



    for landmarks in data['facial_landmarks']:
        try:
            s = landmarks.split(';')
            lm = []
            for str in s:
                lm.append(float(str))
            Landmarks_temp.append(lm)
        except:
            print("ERROR")
            pass

    for index, row in data.iterrows():
        ### IMAGENES
        img_ori = Image.open(dataroot+row['subDirectory_filePath'])
        width, height = img_ori.size
        img_ori = img_ori.convert('1')
        img = img_ori.resize((224,224))
        img = np.asarray(img).reshape((224,224,1))
        Images.append(img)


        ### LANDMARKS
        landmarks = np.array(Landmarks_temp[index]).reshape((68,2))

        img_ori = Image.open(dataroot + row['subDirectory_filePath'])


        landmarks[:, 0] = landmarks[:, 0] / width * 224
        landmarks[:, 1] = landmarks[:, 1] / height * 224
        Landmarks.append(landmarks.flatten())
        #TODO some landmarks overstands images (cause there is face outside the image)
        # we must check it this is a problem, and if it is correct it

        ### EMOCIONES
        emocion = np.zeros(nb_emotions)
        emocion[int(row['expression'])-1]=1
        Emotions.append(emocion)


    num_data = len(Landmarks)

    Landmark_train = []
    Landmark_validation = []
    Image_train = []
    Image_validation = []
    Emotion_train = []
    Emorion_validation = []

    for i in range(num_data):
        if (i % proporcion)==0:
            Landmark_validation.append(Landmarks[i])
            Image_validation.append(Images[i])
            Emorion_validation.append(Emotions[i])
        else:
            Landmark_train.append(Landmarks[i])
            Image_train.append(Images[i])
            Emotion_train.append(Emotions[i])

    Landmark_train = np.array(Landmark_train)
    Landmark_validation = np.array(Landmark_validation)
    Image_train = np.array(Image_train)
    Image_validation = np.array(Image_validation)
    Emotion_train = np.array(Emotion_train)
    Emorion_validation = np.array(Emorion_validation)

    np.savez(train_ouput, Image=Image_train, Landmark=Landmark_train, Emotion=Emotion_train)
    np.savez(validation_output, Image=Image_validation, Landmark=Landmark_validation, Emotion=Emorion_validation)

    if not quiet:
        print(Landmark_train.shape,Landmark_validation.shape)
        print(Image_train.shape,Image_validation.shape)
        print(Emotion_train.shape,Emorion_validation.shape)
