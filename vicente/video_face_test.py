import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras
import dlib

import sys
from argparse import ArgumentParser
import datetime;
import tensorflow as tf

from scipy.spatial import distance as dist
#from imutils import face_utils

import cv2
import os

def eye_aspect_ratio(eye):	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])	
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / C
	return ear

def mouth_aspect_ratio(mouth):	
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    D = dist.euclidean(mouth[3], mouth[5])
    C = dist.euclidean(mouth[0], mouth[4])	
    mar = (A + B + D) / C
    return mar

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return 1-preds[0][0]
    #return preds

def detect_faces(frame):
    current_faces = []
    (h, w) = frame.shape[:2]
    #############################################################
    # detect if ther is faces 
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model_face.setInput(blob)
    detections = model_face.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]                    
        if confidence > 0.80: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")   
            current_faces.append( [startX, startY, endX, endY] )   

    return current_faces

# return the biggest face in frame using dlib
def get_face_caffe(frame):
    current_faces = []
    areas = []
    h, w, c = frame.shape
    #############################################################
    # detect if ther is faces 
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model_face.setInput(blob)
    detections = model_face.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]                    
        if confidence > 0.50: 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")   
            current_faces.append( [startX, startY, endX, endY] )   
            areas.append( (endX - startX)*(endY - startY) )
    
    #faces_rec = np.array(faces_rec)
    if len(areas) > 0:
        index_max = np.argmax(areas)
        face_rec = current_faces[index_max]        
    else:
        return False, None, None, None
        
    # con el rostro obtenido, lo extraemos del frame y pintamos
    face_width = face_rec[2] - face_rec[0]
    face_height = face_rec[3] - face_rec[1]
    pad = int(0.25*face_width)

    y1 = face_rec[1] - pad if face_rec[1] - pad > 0 else 0
    y2 = face_rec[3] + pad if face_rec[3] + pad < h else h
    x1 = face_rec[0] - pad if face_rec[0] - pad > 0 else 0
    x2 = face_rec[2] + pad if face_rec[2] + pad < w else w          

    face_img = frame[y1:y2, x1:x2].copy()
    face_img_copy = face_img.copy()
    # dlib ###########################################
    if args.dlib:
        faces_rec_dlib = detector(face_img, 1)
        for face_rec_dlib in faces_rec_dlib:
            shape = predictor(face_img, face_rec_dlib) #Get coordinates
            face_img_copy = draw_landmarks(face_img_copy, shape)
            #for i in range(1,68): #There are 68 landmark points on each face
            #    cv2.circle(face_img_copy, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 
        if len(faces_rec_dlib) == 0:
            face_img_copy = cv2.resize( face_img_copy, (WIDTH_WIN_FACE, HEIGHT_WIN_FACE) )
            img_h, img_w, ch = face_img_copy.shape    
            pad = int(HEIGHT_WIN_FACE/2)
            img_debug = np.zeros(( img_h, img_w + pad, 3), dtype = "uint8")  
            img_debug[ 0:img_h, 0:img_w ] = face_img_copy
            face_img_copy = img_debug

    return True, face_img, face_img_copy, face_rec

def draw_landmarks(img, shape):
    global TOTAL_BLINKS
    global TOTAL_YAWNS

    leftEye = []
    for i in range(36, 42):
        leftEye.append( [shape.part(i).x, shape.part(i).y] )
    rightEye = []
    for i in range(42, 48):
        rightEye.append( [shape.part(i).x, shape.part(i).y] )
    mouth = []
    for i in range(60, 68):
        mouth.append( [shape.part(i).x, shape.part(i).y] )
    
    leftEye = np.array(leftEye)
    rightEye = np.array(rightEye)
    mouth = np.array(mouth)

    #shape = face_utils.shape_to_np(shape)
    #(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    #(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    #leftEye = shape[lStart:lEnd] #lista de puntos del ojo izq
    #rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    MAR = mouth_aspect_ratio(mouth)

    ear = (leftEAR + rightEAR) / 2.0
    EAR_HISTORY.append(ear)
    MAR_HISTORY.append(MAR)

    if ear < EYE_AR_THRESH:
        TOTAL_BLINKS += 1
    if MAR > MOUTH_AR_THRESH:
        TOTAL_YAWNS += 1

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(img, [leftEyeHull], -1, (0, 0, 255), 2)
    cv2.drawContours(img, [rightEyeHull], -1, (0, 0, 255), 2)
    cv2.drawContours(img, [mouth], -1, (0, 0, 255), 2)

    img = cv2.resize( img, (WIDTH_WIN_FACE, HEIGHT_WIN_FACE) )
    img_h, img_w, ch = img.shape    
    pad = int(HEIGHT_WIN_FACE/2)
    img_debug = np.zeros(( img_h, img_w + pad, 3), dtype = "uint8")  
    img_debug[ 0:img_h, 0:img_w ] = img
    
    #cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 50),	cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 4)
    #cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 50),    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2)
    cv2.putText(img_debug, "EAR: {:.2f}".format(ear), (img_w + 30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(img_debug, "MAR: {:.2f}".format(MAR), (img_w + 30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(img_debug, "BLINKS: " + str(TOTAL_BLINKS), (img_w + 30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(img_debug, "YAWNS: " + str(TOTAL_YAWNS), (img_w + 30, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    return img_debug
    #for i in range(1,68): #There are 68 landmark points on each face
    #    cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 

# return the biggest face in frame using dlib
def get_face(frame):
    h, w, c = frame.shape
    faces_rec = detector(frame, 1) # detect face with dlib

    # obtenemos el rostro mas grande
    areas = []
    for rec in faces_rec:
        areas.append( (rec.bottom() - rec.top())*(rec.right() - rec.left()) )
    
    #faces_rec = np.array(faces_rec)
    if len(areas) > 0:
        index_max = np.argmax(areas)
        face_rec = faces_rec[index_max]        
    else:
        return False, None, None, None
        
    # con el rostro obtenido, lo extraemos del frame y pintamos
    face_width = face_rec.right() - face_rec.left()
    face_height = face_rec.bottom() - face_rec.top()
    pad = int(0.25*face_width)

    y1 = face_rec.top() - pad if face_rec.top() - pad > 0 else 0
    y2 = face_rec.bottom() + pad if face_rec.bottom() + pad < h else h
    x1 = face_rec.left() - pad if face_rec.left() - pad > 0 else 0
    x2 = face_rec.right() + pad if face_rec.right() + pad < w else w
    
    frame_copy = frame.copy()

    shape = predictor(frame, face_rec) #Get coordinates
    frame_copy = draw_landmarks(frame_copy, shape)
    #for i in range(1,68): #There are 68 landmark points on each face
    #    cv2.circle(frame_copy, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 

    face_img = frame[y1:y2, x1:x2].copy()
    face_img_painted = frame_copy[y1:y2, x1:x2]

    return True, face_img, face_img_painted, (face_rec.left(), face_rec.top(), face_rec.right(), face_rec.bottom())

def draw_box(frame, rec, prop_drowsy):
    h, w, c, = frame.shape
    x = rec[0]
    y = rec[1]
    x_end = rec[2]
    y_end = rec[3]

    pad = int(0.3*(x_end-x))

    y1 = y - pad if y - pad > 0 else 0
    y2 = y_end + pad if y_end + pad < h else h
    x1 = x - pad if x - pad > 0 else 0
    x2 = x_end + pad if x_end + pad < w else w
    
    
    if prop_drowsy > 0.70:
        cv2.putText(frame, "A L E R T ", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10) 

        # para el cuadro transparente
        blk = np.zeros(frame.shape, np.uint8)
        cv2.rectangle(blk, (x1, y1), (x2, y2), (0, 0, 255), cv2.FILLED)
        frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)

        color = [0, 0, 255]
    else:
        color = [0, 255, 0]

    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    left = int(x1)
    top = int(y1)
    right = int(x2)
    bottom = int(y2)
    width_face = right - left
    height_face = bottom - top
    w = int(width_face / 3)
    h = int(height_face / 3)

     
    

    cv2.line(frame, (left, top), (left + w, top), color, thickness=10)
    cv2.line(frame, (left, top), (left, top + h), color, thickness=10)
    cv2.line(frame, (right, top), (right - w, top), color, thickness=10)
    cv2.line(frame, (right, top), (right, top + h), color, thickness=10)

    cv2.line(frame, (left, bottom), (left, bottom - h), color, thickness=10)
    cv2.line(frame, (left, bottom), (left + w, bottom), color, thickness=10)
    cv2.line(frame, (right, bottom), (right, bottom - h), color, thickness=10)
    cv2.line(frame, (right, bottom), (right - w, bottom), color, thickness=10)
    #cv2.putText(frame, "ALERT ", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2, 1)


    return frame 

def draw_face(img):
    h, w, c = img.shape
    detections = detector(img, 1) #Detect the faces in the image

    shape = 0
    for k,rect in enumerate(detections): 
        #print("face at pos:", rect.left(), rect.top(), rect.right(), rect.bottom())
        shape = predictor(img, rect) #Get coordinates
        for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)

        face_width = rect.right() - rect.left()
        face_height = rect.bottom() - rect.top()
        pad = int(0.25*face_width)

        y1 = rect.top() - pad if rect.top() - pad > 0 else 0
        y2 = rect.bottom() + pad if rect.bottom() + pad < h else h
        x1 = rect.left() - pad if rect.left() - pad > 0 else 0
        x2 = rect.right() + pad if rect.right() + pad < w else w

        #face_img = img[ rect.top() - pad : rect.bottom() + pad, rect.left() - pad:rect.right() + pad ]
        face_img = img[ y1:y2, x1:x2 ]

    if len(detections) > 0:
        cv2.imshow("face", face_img)

def draw_pads(frame, drowsy_prop, yawn_prop, frame_count):
    fh, fw, ch = frame.shape
    upper_pad = int(fh*0.1) 
    lower_pad = int(fh*0.05)
    
    frame_debug = np.zeros(( fh + upper_pad + lower_pad, fw, 3), dtype = "uint8")       
    #print(upper_pad, lower_pad, frame_debug.shape)
    frame_debug[ upper_pad:fh + upper_pad , 0:fw] = frame

    font_size =1
    title = "S I S O"
    cv2.putText(frame_debug, title, ( 20 , 70), cv2.FONT_HERSHEY_TRIPLEX, font_size*2.3, (0, 255, 0), 4) 
    cv2.putText(frame_debug, "D R O W S Y  " + str(drowsy_prop), (int(fw*0.6), 40), cv2.FONT_HERSHEY_TRIPLEX, font_size, (0, 255, 0), 2) 
    #cv2.putText(frame_debug, "Y A W N  " + str(yawn_prop), (int(fw*0.6), 80), cv2.FONT_HERSHEY_TRIPLEX, font_size, (0, 255, 0), 2) 
    
    cv2.putText(frame_debug, "F R A M E    " + str(frame_count), (20, fh + upper_pad + 35), cv2.FONT_HERSHEY_TRIPLEX, font_size, (0, 255, 0), 2)

    return frame_debug
        

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
parser.add_argument("-o", "--output", dest="path_output", help="path output", metavar="DIR", default='')
parser.add_argument("-w", "--window", dest="window", help="show window", default=True)
parser.add_argument("-dlib", "--dlib", dest="dlib", help="set dlib", default=True)
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

#python3 video_face_test.py -m1 ../models/modelSiso_y_NTHU_Inception3.h5 -v /home/vicente/datasets/NTHU/testing/016_noglasses_mix.mp4
#python3 video_face_test.py -m1 ../models/modelSiso_y_NTHU_Inception3.h5 -v /home/vicente/datasets/SISO/normal.mp4



STEP_FRAME_RATE = 15 # cada 10 frames hacemos el procesamiento
EAR_HISTORY = []
MAR_HISTORY = []
EYE_AR_THRESH = 0.3
MOUTH_AR_THRESH = 0.6
EYE_AR_CONSEC_FRAMES = 3
TOTAL_BLINKS = 0
TOTAL_YAWNS = 0

WIDTH_WIN_FACE = 400
HEIGHT_WIN_FACE = 600

model = keras.models.load_model(args.path_model_1)
print("model paul loaded")

#model_yawn = keras.models.load_model(args.path_model_2)
#print("model yawn loaded")

model_face = cv2.dnn.readNetFromCaffe(current_dir + "/../models/deploy.prototxt.txt", current_dir + "/../models/res10_300x300_ssd_iter_140000.caffemodel")
print("model face detection loaded")

if args.dlib:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../models/data_shape_predictor_68_face_landmarks.dat") #Or set this to whatever 
    print("dlib loaded")

try:
    if args.path_video == None:
        cap = cv2.VideoCapture(0)    
    else:
        cap = cv2.VideoCapture(args.path_video)
        if not cap.isOpened():
            print("cannot open video")
except:
    print("Cannot open video")

frame_count = 1
drowsy_prop = 0
non_drowsy_prop = 0
yawn_prop = 0

if args.path_output != '':
    now = datetime.datetime.now()
    log_file_name = now.strftime("log_%Y-%m-%d_%H:%M:%S.txt")
    with open(args.path_output + "/" + log_file_name, "a+") as file_log:
        print("\n Starting SISO analysis...\n ")
        file_log.write("\n Starting SISO analysis...\n ")
        file_log.flush()

if int(args.window) == 1:
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1200,800)
    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('face', 400,400)

current_faces_rec = []
res = False

faces_caffe = 0
faces_dlib = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:  
        
        if frame_count % STEP_FRAME_RATE == 0: 
            res, face_img, face_img_painted, face_rec = get_face_caffe(frame) # caffe es mejor qdlib
            if res:
                # predict drowsiness
                face_img = cv2.resize(face_img, (224,224))                        
                drowsy_prop = round(predict(model, face_img), 3)
                if drowsy_prop > 0.70 and args.path_output != '':
                    write_log(drowsy_prop, yawn_prop, frame, file_log)                 

                frame = draw_box(frame, face_rec, drowsy_prop)                
                if args.window:
                    cv2.imshow("face", face_img_painted)
        else:
            if res:
                frame = draw_box(frame, face_rec, drowsy_prop)
        
        frame_debug = draw_pads(frame, drowsy_prop, drowsy_prop, frame_count)        
        if args.window:
            cv2.imshow('frame',frame_debug)
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

if args.path_output != '':
    file_log.close() 
cap.release()
cv2.destroyAllWindows()

