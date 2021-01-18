import numpy as np
import os
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def read_tensor_from_image_file(file_name,
                input_height=224,
                input_width=224,
                input_mean=0,
                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def extractor(image_path):

	with open('output_graph.pb', 'rb') as graph_file:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(graph_file.read())
		tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		# for n in tf.get_default_graph().as_graph_def().node:
		# 	print(n.name)
		# for i in sess.graph.get_operations():
		# 	print(i.name)
		# pooling_tensor = tf.get_default_graph().get_tensor_by_name('module_apply_default/InceptionV3/Logits/GlobalPool:0')
		pooling_tensor = tf.get_default_graph().get_tensor_by_name('module_apply_default/MobilenetV1/Logits/AvgPool_1a/AvgPool:0')
		# image_data = tf.gfile.FastGFile(image_path, 'rb').read()
		image_data = read_tensor_from_image_file(image_path)
		pooling_features = sess.run(pooling_tensor, \
			{'Placeholder:0': image_data})
		print(pooling_features.shape)
		pooling_features = pooling_features[0]
		#print(pooling_features)

	return pooling_features

def extract_features():
	with open('data/data_file.csv','r') as f:
		reader = csv.reader(f)
		for videos in reader:
			path = os.path.join('data', 'sequences', videos[2] + '-' + str(26) + \
'-features.npy')
		
			path_frames = os.path.join('data', videos[0], videos[1])
			filename = videos[2]
			frames = sorted(glob.glob(os.path.join(path_frames, filename + '/*jpg')))
		
			sequence = []
			for image in frames:
				with tf.Graph().as_default():
					features = extractor(image)
					print('Appending sequence of image:',image,' of the video:',videos)				
				  
                   
              		             																																																																																																																																			
					sequence.append(features)

			np.save(path,sequence)
			print('Sequences saved successfully')						

extract_features()					
