# algoritmo simple dee deteccion de somnolencia basado en la duracion de los ojos cerrados (es lo mismo que blink_retrieval)


from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../models/data_shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def detect_drowsiness(frame, detector, predictor):
    global COUNTER

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) != 1:
        print("Too many faces or no faces: ", len(rects))
        cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    

    else: # solo un rostro
        face = rects[0]
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
    
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1               
            if COUNTER >= EYE_AR_CONSEC_FRAMES:                                           
        
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)           
        
        else:
            COUNTER = 0

        draw(frame, leftEye, rightEye, ear)

def draw(frame, leftEye, rightEye, ear):
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    print len(channels)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)




cap = cv2.VideoCapture('/home/vicente/datasets/SISO/Drowsy/H264_20191230_233115_CH2.AVI_ROTATE.AVI')
#cap = cv2.VideoCapture('/home/vicente/datasets/SISO/Drowsy/H264_20191230_065315_CH1.AVI_ROTATE.AVI')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('outpy.avi', fourcc, 30, (frame_width, frame_height))

index = 0
cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1200,900)
while True:
    ret, frame = cap.read()

    if ret:
        frame = imutils.resize(frame, width=850)
        frame_copy = frame.copy()
        #frame_equalized = hisEqulColor(frame) # es casi lo mismo   
        
        #detect_drowsiness(frame, detector, predictor)

        # show the frame
        cv2.imshow("Frame", frame)
        #cv2.imshow("Frame equalizaded", frame_equalized)
        #out.write(frame)

        #####################################################################################################
        # testing gamma correction 
        #frame = imutils.resize(frame, width=650)
                
        all_images = frame
        for gamma in np.arange(0.3, 1.0, 0.3):   
            adjusted = adjust_gamma(frame_copy, gamma=gamma)

            #detect_drowsiness(adjusted, detector, predictor)

            cv2.putText(adjusted, str(gamma), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            all_images = np.hstack([all_images, adjusted])

        cv2.imshow("Gamma correction", all_images)
        #frame_equalized = hisEqulColor(all_images) # es casi lo mismo
        #cv2.imshow("Gamma correction equalized", frame_equalized)
        
        #####################################################################################################

        key = cv2.waitKey(1) & 0xFF    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
