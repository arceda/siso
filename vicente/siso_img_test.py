import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil
import time
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
import sys

path_model = sys.argv[1]
path_image = sys.argv[2]
# python3 siso_img.py ../models/model_inception_siso_with_nthu_10k_with_2class.h5 /home/vicente/datasets/SISO/IMG/TEST/dormido/glasses_corregido8292.jpg

def predict(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities 
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def plot_preds(img, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        preds: list of predicted labels and their probabilities
    """
    labels = ("dormido", "despierto")
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.figure(figsize=(8,8))
    plt.subplot(gs[0])
    plt.imshow(np.asarray(img))
    plt.subplot(gs[1])
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


#tamanio de nuestras imágenes
WIDTH = 128 
HEIGHT = 128

model = keras.models.load_model(path_model)

img = image.load_img(path_image, target_size=(HEIGHT, WIDTH))
preds = predict(model, img)
print(preds)
plot_preds(np.asarray(img), preds)
