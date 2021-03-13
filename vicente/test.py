# este scriot obtiene las métricas de las bases de datos entrenadas
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import sys
from argparse import ArgumentParser
import cv2
import os
import glob

##################################################################
# arguments
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="path_model", help="model drowsy path", metavar="DIR")
parser.add_argument("-d", "--dataset", dest="path_dataset", help="dataset TEST path", metavar="DIR")  
args = parser.parse_args()
##################################################################

print(args.path_model)
print(args.path_dataset)

current_dir = os.path.dirname(os.path.abspath(__file__))

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return 1-preds[0][0]


model = keras.models.load_model(args.path_model)

files_dormido = glob.glob(args.path_dataset + "/dormido/*") 
files_despierto = glob.glob(args.path_dataset + "/despierto/*") 

y_true = np.hstack(( np.ones(len(files_dormido)), np.zeros(len(files_despierto)) )) 

y_pre = []
for file in files_dormido:
    img = cv2.imread(file)
    img = cv2.resize(img, (224,224))    
    y_pre.append(predict(model, img))

for file in files_despierto:
    img = cv2.imread(file)
    img = cv2.resize(img, (224,224))    
    y_pre.append(predict(model, img))

print(y_true)
print(y_pre)

acc = accuracy_score(y_true, y_pred, normalize=False)
matrix = confusion_matrix(y_true, y_pred, labels=["somnoliento", "no somnoliento"])

print(acc)
print(matrix)
# python3 test.py -m ../models/modelSiso_y_NTHU_Inception3.h5 -d /home/vicente/datasets/SISO/TEST/