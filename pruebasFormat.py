import pandas as pd
import numpy as np
import shutil, os

dataroot = "/mnt/pgth04b/Data_Miguel/AutoEncoders/datasets/AFFECTNET"

Image = []
Landmark = []
Emotion = []

columns = ['subDirectory_filePath','face_x','face_y','face_width','face_height','facial_landmarks','expression','valence','arousal']

data = pd.read_csv('training.csv');

for landmarks in data['facial_landmarks']:
    try:
        s = landmarks.split(';')
        lm = []
        for str in s:
            lm.append(float(str))
        Landmark.append(lm)
    except:
        print("ERROR")
        pass

for filename in data['subDirectory_filePath']:
    n = filename.find('/')
    os.mkdir('datasets/'+filename[0:n])
    shutil.copy(dataroot+'/'+filename,'datasets/'+filename)

Landmark = np.array(Landmark)
#import pdb
#pdb.set_trace()
print(Landmark)
