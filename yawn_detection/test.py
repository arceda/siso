import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)

from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from matplotlib import cm

import math
import os
import sys
import cv2
import numpy as np

from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import load_model

from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from numpy import argmax

import glob

current_dir = os.path.dirname(os.path.abspath(__file__))


model  = load_model(current_dir + "/models/model_epoch=50.h5")

#cap = cv2.VideoCapture('/home/vicente/datasets/NTHU/testing/videoplayback.mp4')
#cap = cv2.VideoCapture('/home/vicente/datasets/NTHU/testing/020_glasses_yawning.avi')

# for video 003_noglasses_mix at 2:48 there is yawning 
#cap = cv2.VideoCapture('/home/vicente/datasets/NTHU/testing/003_noglasses_mix.mp4')
#min = 2; seg = 48
#frame_yawning = min*60*30 + seg*30 # minuto 2:48
#cap.set(1,frame_yawning)

# for video 018_noglasses_mix at 2:48 there is yawning 
#cap = cv2.VideoCapture('/home/vicente/datasets/NTHU/testing/018_noglasses_mix.mp4')

path_video = sys.argv[1]
minute = sys.argv[2]
second = sys.argv[3]

#cap = cv2.VideoCapture(path_video)
cap = cv2.VideoCapture(0)

# python3 test.py /home/vicente/datasets/NTHU/testing/020_glasses_yawning.mp4 0 1
# python3 test.py /home/vicente/datasets/NTHU/testing/003_noglasses_mix.mp4 2 48
# python3 test.py /home/vicente/datasets/NTHU/testing/018_noglasses_mix.mp4 2 30
# python3 test.py /home/vicente/datasets/NTHU/testing/videoplayback.mp4 0 1

min = int(minute); seg = int(second)
frame_yawning = min*60*30 + seg*30 # minuto 2:30
cap.set(1,frame_yawning)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#out = cv2.VideoWriter(current_dir + '/results/output.avi', fourcc, 60, (frame_width, frame_height))
#print(total_frames)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        img = cv2.resize(frame, (64,64))

        X = np.array([img])
        label =  model.predict(X)[0][0]
        print(label)
        if label > 0.6:
            cv2.putText(frame, "YAWNING ALERT!", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4) 
        else:
            cv2.putText(frame, "NON YAWNING", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)    

        cv2.imshow('frame',frame)
        #out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

#cap.release()
#out.release()
#cv2.destroyAllWindows()

#2:48