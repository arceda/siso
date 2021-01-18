import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import csv
import glob
import tensorflow as tf
import h5py as h5py
import time
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
# import winsound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Dataset():
    def __init__(self, data_name, seq_length=10, class_limit=2, image_shape=(56, 24, 3)):
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('sequences')
        # Get the data.
        self.data = data_name

    def get_all_sequences_in_memory(self):
        X = []
        sequence = self.get_extracted_sequence()
        X.append(sequence)
        return np.array(X)

    def get_extracted_sequence(self):
        """Get the saved extracted features."""
        filename = self.data
        path = os.path.join(self.sequence_path, str(filename) + '-' + str(10) +
                            '-' + 'features' + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

def load(video):
    data = Dataset(
        seq_length=10,
        class_limit=2,
        data_name=video
    )

    X_test = data.get_all_sequences_in_memory()

    print ("##################################################")
    X_test = np.ravel(X_test)
    X_test = X_test.reshape(1, 10, -1)
    print("X_test.shape", X_test.shape)
    print("##################################################")

    start_time = time.time()
    model = load_model('my_model.h5')
    elapsed_time = time.time() - start_time
    print("load time: ", elapsed_time)

    predictions = model.predict(X_test)

    for j in predictions:
        if j[0] > j[1]:
            print("Driver is alert with the confidence of", (j[0]*100), "%")
        else:
            print("Driver is drowsy with the confidence of", (j[1]*100), "%")
            print("Sounding the alarm now....")
            duration = 10  # second
            freq = 440  # Hz
            os.system(
                'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
            # winsound.Beep(440, 1000)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_alt2.xml')
eye_cascade_left = cv2.CascadeClassifier('Haar/haarcascade_mcs_eyepair_big.xml')
eye_cascade_right = cv2.CascadeClassifier('Haar/haarcascade_mcs_eyepair_big.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

# def hsv_histogram_for_window(frame, window):
#     # set up the ROI (region of interest) for tracking
#     c,r,w,h = window
#     roi = frame[r:r+h, c:c+w]
#     hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
#     roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
#     cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#     return roi_hist


# def resample(weights):
#     n = len(weights)
#     indices = []
#     C = [0.] + [sum(weights[:i+1]) for i in range(n)]
#     u0, j = np.random.random(), 0
#     for u in [(u0+i)/n for i in range(n)]:
#       while u > C[j]:
#           j+=1
#       indices.append(j-1)
#     return indices

def read_tensor_from_image_file(file_name,
                input_height=96,
                input_width=96,
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

def skeleton_tracker2(v):
    # load model
    # global model
    # model = load_model('my_model.h5')

    #create folder that save features(.npy)
    createFolder('./sequences')

    # Open output file
    #output_name = sys.argv[3] + file_name
    output_name = 'testing.txt'
    output = open(output_name,"w")

    frameCounter = 0
    frameCounter2 = -1
    countFolder = 1
    createFolder('./'+sys.argv[4]+'/'+str(countFolder)+'/')
    # read first frame
    ret ,frame = v.read()
    if frame is None:
        print("end")
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame) # x, y, width, height
    pt = (0,c+w/2,r+h/2)
    # Write track point for first frame
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # # set the initial tracking window
    # track_window = (c,r,w,h)

    # # calculate the HSV histogram in the window
    # # NOTE: you do not need this in the Kalman, Particle or OF trackers
    # roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    # term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    # add
    x = c-int(w/2)
    y = r-int(h/2)
    w = 2*w
    h = 2*h
    ey = None
    ey1 = None

    with open('output_graph_v2_50_96.pb', 'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        pooling_tensor = tf.get_default_graph().get_tensor_by_name('module_apply_default/MobilenetV2/Logits/AvgPool:0')

        while(1):
            ret ,frame = v.read() # read another frame
            # if ret == False:
            #     break

            # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            # dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # x,y,w,h = track_window

            # add
            # x,y,w,h = detect_one_face(frame)
            # x = x + 5

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # write the result to the output file
            pt = (frameCounter,x+w/2,y+h/2)
            output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y

            # h = int(h/2)+10

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #image, top-left, bottom-right, RGB color, width
            cv2.imshow('img',frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes_left = eye_cascade_left.detectMultiScale(roi_gray)
            eyes_right = eye_cascade_right.detectMultiScale(roi_gray)
            for (ex1,ey1,ew1,eh1) in eyes_left:
                break
            for (ex,ey,ew,eh) in eyes_right:
                break

            if (ey != None and ey1 != None):
                if (ey < ey1):
                    cy = ey
                    ch = ey1 - ey + eh
                else:
                    cy = ey1
                    ch = ey - ey1 + eh1

                #????
                frame[ey+eh:ey, ex:ex1+ew1]

                # cv2.rectangle(roi_color,(ex-5,ey-5),(ex+(ex1-ex)+ew1+5,ey+eh+5),(0,255,0),2)
                cv2.rectangle(roi_color,(ex-5,cy-10),(ex1+ew1+15,cy+ch+5),(0,255,0),2)

                # croppedImg = roi_color[ey1:ey1+eh1, ex:ex1+ew1]
                croppedImg = roi_color[cy-5:cy+ch, ex:ex1+ew1+10]

                if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=0):
                    frameCounter = frameCounter + 1
                    continue

                print (croppedImg.shape)
                #print(frameCounter)
                frameCounter2 = frameCounter2+1
                #print(frameCounter2)
                if(frameCounter2 == 10):
                    start_time = time.time()
                    with tf.Graph().as_default():
                        frameCounter2 = 0
                        video = countFolder
                        print(video)
                        # extract_features(video)
                        path = os.path.join('sequences', str(video) + '-' + str(10) + '-features.npy')
                        path_frames = os.path.join('Output')
                        filename = video
                        frames = sorted(glob.glob(os.path.join(path_frames, str(filename) + '/*jpg')))
                        
                        sequence = []
                        
                        for image_path in frames:
                            # print('Appending sequence of image:',image_path,' of the video:',video)
                            image_data = read_tensor_from_image_file(image_path)
                            pooling_features = sess.run(pooling_tensor, \
                                {'Placeholder:0': image_data})
                            pooling_features = pooling_features[0]
                            sequence.append(pooling_features)

                        np.save(path,sequence)
                        print('Sequences saved successfully')

                    elapsed_time = time.time() - start_time
                    print('used time:', elapsed_time)

                    load(video)

                    countFolder = countFolder+1
                    createFolder('./'+sys.argv[4]+'/'+str(countFolder)+'/')

                cv2.imshow('img1',croppedImg)
                output_name = "./"+sys.argv[4]+"/"+str(countFolder)+"/"+str(frameCounter)+".jpg"
                cv2.imwrite(output_name, croppedImg)

            k = cv2.waitKey(1) & 0xff
            if k == ord("q"):
                break
            frameCounter = frameCounter + 1

    output.close()


if __name__ == '__main__':
    question_number = -1

    question_number = int(sys.argv[1])
    if (question_number > 4 or question_number < 1):
        print("Input parameters out of bound ...")
        sys.exit()

    # read video file
    v = cv2.VideoCapture(sys.argv[2])
    # v = cv2.VideoCapture(0)
    skeleton_tracker2(v)
    v.release()
    cv2.destroyAllWindows()

    #if (question_number == 1):
     #   skeleton_tracker1(video, "output_camshift.txt")

