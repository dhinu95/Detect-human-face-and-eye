import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

#fer2013 dataset

csv_file = pd.read_csv("/content/gdrive/My Drive/fer2013.csv")
len(csv_file.loc[csv_file['pixels'] != 1])
csv_file = csv_file.loc[csv_file['emotion'] != 1]
faces = csv_file['pixels'].tolist()
emotions = csv_file['emotion'].tolist()
for i in range(len(emotions)):
  if emotions[i] != 0:
    emotions[i] -= 1
for i in range(len(faces)):
  faces[i] = np.array(faces[i].split()).reshape(48, 48).astype('float32') 
faces = np.array(faces)
emotions = np.array(emotions)
len(faces)
len(emotions)

#own dataset

mypath1 = "gdrive/My Drive/owndata"
onlyfiles1 = np.array([f for f in listdir(mypath1) if isfile(join(mypath1, f))])
np.random.shuffle(onlyfiles1)
for i in onlyfiles1:
    img = cv2.imread(mypath1 + '/' + i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (160, 160)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    X.append(img)
    y.append(int(i[0]) - 1)
dataset = np.array(X)
y = np.array(y)
len(X)


