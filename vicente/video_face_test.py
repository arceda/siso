import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras

import sys
from argparse import ArgumentParser
import datetime;
import tensorflow as tf

import cv2
import os

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0][0]
    #return preds

def detect_faces(model_face, frame):
    current_faces = []
    (h, w) = frame.shape[:2]
    #############################################################
    # detect if ther is faces 
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model_face.setInput(blob)
    detections = model_face.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]                    
        if confidence > 0.90: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")   
            current_faces.append( [startX, startY, endX, endY] )   

    return current_faces


def draw_box(frame, x, y, x_end, y_end, prop_drowsy):
    if prop_drowsy > 0.70:
        cv2.putText(frame, "ALERT ", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4) 

        # para el cuadro transparente
        blk = np.zeros(frame.shape, np.uint8)
        cv2.rectangle(blk, (x, y), (x_end, y_end), (0, 0, 255), cv2.FILLED)
        frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)

    cv2.rectangle(frame, (x, y), (x_end, y_end), (0, 0, 255), 2)
    return frame 


def write_log(drowsy_prop, yawn_prop, frame, file_log):
    now = datetime.datetime.now()
    img_file_name = now.strftime("img_%Y-%m-%d_%H:%M:%S.%f.jpg")
    cv2.imwrite(args.path_output + "/" + img_file_name, frame)
    print("drowsy detected: ", drowsy_prop, "\tfile: ", img_file_name) 

    with open(args.path_output + "/" + log_file_name, "a+") as file_log:
        print("\n Starting SISO analysis...\n")
        file_log.write("drowsy detected: " + str(drowsy_prop) + "\tfile: " + img_file_name + "\n")    
        file_log.flush()      

##################################################################
# arguments
parser = ArgumentParser()
parser.add_argument("-m1", "--model_1", dest="path_model_1", help="model drowsy path", metavar="FILE")
parser.add_argument("-m2", "--model_2", dest="path_model_2", help="model yawn path", metavar="FILE")  
parser.add_argument("-v", "--video", dest="path_video", help="path video", metavar="FILE")
parser.add_argument("-o", "--output", dest="path_output", help="path output", metavar="DIR")
parser.add_argument("-w", "--window", dest="window", help="show window")
args = parser.parse_args()
##################################################################

print(args.path_model_1)
print(args.path_model_2)
print(args.path_video)
print(args.path_output)

current_dir = os.path.dirname(os.path.abspath(__file__))

# python3 video_face_test.py -m1 ../models/modelSiso_y_NTHU_Inception3.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/003_noglasses_mix.mp4 -o /home/vicente/projects/siso/output/ -w 1
# python3 video_face_test.py -m1 ../models/modelSiso_y_NTHU_Inception3.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/016_noglasses_mix.mp4 -o /home/vicente/projects/siso/output/ -w 1
# python3 video_face_test.py -m1 ../models/modelSiso_y_NTHU_Inception3.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/018_noglasses_mix.mp4 -o /home/vicente/projects/siso/output/ -w 1
# python3 video_face_test.py -m1 ../models/modelSiso_y_NTHU_Inception3.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/videoplayback.mp4 -o /home/vicente/projects/siso/output/ -w 1


#tamanio de nuestras imagenes
WIDTH = 128 
HEIGHT = 128
STEP_FRAME_RATE = 15 # cada 10 frames hacemos el procesamiento

model = keras.models.load_model(args.path_model_1)
print("model paul loaded")

#model_yawn = keras.models.load_model(args.path_model_2)
#print("model yawn loaded")

model_face = cv2.dnn.readNetFromCaffe(current_dir + "/../models/deploy.prototxt.txt", current_dir + "/../models/res10_300x300_ssd_iter_140000.caffemodel")
print("model face detection loaded")

try:
    if args.path_video == None:
        cap = cv2.VideoCapture(0)    
    else:
        cap = cv2.VideoCapture(args.path_video)
except:
    print("Cannot open video")

frame_count = 1
drowsy_prop = 0
non_drowsy_prop = 0
yawn_prop = 0

now = datetime.datetime.now()
log_file_name = now.strftime("log_%Y-%m-%d_%H:%M:%S.txt")
with open(args.path_output + "/" + log_file_name, "a+") as file_log:
    print("\n Starting SISO analysis...\n ")
    file_log.write("\n Starting SISO analysis...\n ")
    file_log.flush()

if int(args.window) == 1:
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('frame', 1200,800)

    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('face', 400,400)


current_faces_rec = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:    
        if frame_count % STEP_FRAME_RATE == 0:            
            current_faces_rec = detect_faces(model_face, frame)  # detect faces 

            for face_rec in current_faces_rec:    
                face = frame[face_rec[1]-100:face_rec[3]+100, face_rec[0]-100:face_rec[2]+100] 
                face = cv2.resize(face, (224,224))
                cv2.imshow('face',face)
                        
                drowsy_prop = round(predict(model, face), 3)  

                if drowsy_prop > 0.70:
                    write_log(drowsy_prop, yawn_prop, frame, file_log)
                frame = draw_box(frame, face_rec[0], face_rec[1], face_rec[2], face_rec[3], drowsy_prop)                      
                    
        else:
            for face_rec in current_faces_rec:    
                frame = draw_box(frame, face_rec[0], face_rec[1], face_rec[2], face_rec[3], drowsy_prop) 
       
        cv2.rectangle(frame, (0,0), (180, 60), (255,255,255), -1)
        cv2.putText(frame, "drowsiness: " + str(drowsy_prop), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
        #cv2.putText(frame, "no-drowsy: " + str(non_drowsy_prop), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)    
        #cv2.putText(frame, "yawn:      " + str(yawn_prop), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)    

        if int(args.window) == 1:
            cv2.imshow('frame',frame)
        frame_count += 1
        #out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

file_log.close() 
cap.release()
cv2.destroyAllWindows()


################################################################################
'''
img = cv2.imread("../media/test5.jpg")
img = cv2.resize(img, (224, 224))
drowsy_prop = predict(model, img) 
print( "cwecwec", drowsy_prop )

(h, w) = img.shape[:2]
#blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
model_face.setInput(blob)
detections = model_face.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]                    
    if confidence > 0.90: 
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")                   

        face = img[startY-100:startY-100+endY+100, startX-100:endX+100] 
        face = img[startY:startY+endY, startX:endX] 
        img = cv2.resize(img, (224, 224))

        drowsy_prop = predict(model, face) 
        print( drowsy_prop )

        cv2.imshow("tst", face)
        cv2.waitKey(0)
'''
################################################################################