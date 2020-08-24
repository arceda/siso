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

#from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from numpy import argmax

import glob

current_dir = os.path.dirname(os.path.abspath(__file__))

# params 
#########################################################################################################
#########################################################################################################
path_db = "/home/vicente/datasets/NTHU_IMG/yawn/"
path_db = sys.argv[1]
epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])

#python3 yawn_detection/train.py "/home/vicente/datasets/NTHU_IMG/yawn/" 2  64

#########################################################################################################
#########################################################################################################

# read dataset
#########################################################################################################
#########################################################################################################
files_yawning = np.array(glob.glob(path_db + "/yawning/*"))
files_non_yawning = np.array(glob.glob(path_db + "/non_yawning/*"))

############################################################
#files_yawning = files_yawning[0:100]
#files_non_yawning = files_yawning[0:100]
############################################################

files_X = np.hstack((files_yawning, files_non_yawning))

X = []
for file in files_X:
    img = cv2.imread(file)
    img = cv2.resize(img, (64,64))
    X.append(img)

X = np.array(X)

# create y vector
y_yawning = []
for file in files_yawning:
    #y_yawning.append("yawning")
    y_yawning.append(1)

y_non_yawning = []
for file in files_yawning:
    #y_non_yawning.append("non_yawning")
    y_non_yawning.append(0)

y_yawning = np.array(y_yawning)
y_non_yawning = np.array(y_non_yawning)
y = np.hstack((y_yawning, y_non_yawning))

labels = set(y)

print("X.shape: ", X.shape)
print("y.shape: ", y.shape)
print("labels: ", labels)


samples_num, img_rows, img_cols, img_channels = X.shape

#########################################################################################################
#########################################################################################################

# model
#########################################################################################################
#########################################################################################################

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (img_rows, img_cols, img_channels), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fourth convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
history = classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#########################################################################################################
#########################################################################################################

# train
history = classifier.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

classifier.save(current_dir + '/results/model_epoch='+ str(epochs) +'.h5')

print("finish")