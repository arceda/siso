import tensorflow as tf
import os
import skimage
import matplotlib.pyplot as plt 
from skimage import transform 
from skimage.color import rgb2gray
import numpy as np
import random

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/vicente/datasets/"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

# ver la distribucion de cada clase
#plt.hist(labels, 62)
#plt.show()

# para ver algunas images
##########################################################################################
'''
traffic_signs = [300, 2250, 3650, 4000]
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))
'''


# PARA VER UNA IMAGES DE CLASE
##########################################################################################
'''
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
plt.show()
'''


images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)


'''
traffic_signs = [300, 2250, 3650, 4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images28[traffic_signs[i]], cmap="gray")
    plt.subplots_adjust(wspace=0.5)

plt.show()
'''
images_flatten = np.array([image.flatten() for image in images28])

print(images28.shape)
print(images_flatten.shape)


# NEURAL NETWORK
###################################################################################################
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 784]) # 784 despues de flatten 28x28
y = tf.placeholder(dtype = tf.float32, shape = [None, 62])

# capa 1
w_1 = tf.Variable(tf.truncated_normal(shape=[784, 512], stddev = 0.2))
b_1 = tf.Variable(tf.zeros(shape=[512]))

# capa 2
w_2 = tf.Variable(tf.truncated_normal(shape=[512, 62], stddev = 0.2))
b_2 = tf.Variable(tf.zeros(shape=[62]))

z_1 = tf.matmul(x, w_1) + b_1
a_1 = tf.nn.relu(z_1) # relu como funcion de activacion para esta capa

z_2 = tf.matmul(z_1, w_2) + b_2
y_  = z_2

# funcion costo
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y))

# predicciones
train_pred = tf.nn.softmax(y_)

# optimizador
opt = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# one hot transformation
a = np.array(labels)
labels_one_hot = np.zeros((a.size, a.max()+1))
labels_one_hot[np.arange(a.size),a] = 1
print(labels_one_hot.shape)

# running
###################################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def precision(predicciones, etiquetas):
    return (100.0 * np.sum(np.argmax(predicciones, 1) == np.argmax(etiquetas, 1))
          / predicciones.shape[0])


for i in range(2000):
        #print('EPOCH', i)
        _,costo,predicciones = sess.run([opt, cross_entropy, train_pred], feed_dict={x: images_flatten, y: labels_one_hot})
        if i % 10 == 0:
            print("Costo del minibatch hasta el paso %d: %f" % (i, costo))
            print("Precision en el conjunto de entrenamiento: %.1f%%" % precision(predicciones, labels_one_hot))            
            print("\n")
        #print('DONE WITH EPOCH')

'''
%%time 

pasos = 5000

print("Entrenamiento:")
for i in range(pasos):
    batch = mnist.train.next_batch(100)
    _,costo,predicciones =  sess.run([opt, cross_entropy, train_pred],  feed_dict={x: batch[0], y: batch[1]})
    
    if (i % 500 == 0):
        print("Costo del minibatch hasta el paso %d: %f" % (i, costo))
        print("Precision en el conjunto de entrenamiento: %.1f%%" % precision(predicciones, batch[1]))
        print("Precision en el conjunto de validacion: %.1f%%" % precision(
        valid_pred.eval(session=sess), mnist.validation.labels))
        print("\n")

'''