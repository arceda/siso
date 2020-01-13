# algorithmo propuesto por : A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection


from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	#ear = (A + B) / (2.0 * C)
	ear = (A + B) / C

	# return the eye aspect ratio
	return ear

def blink_retrieval_alg(x, epsilon):
    print("processing blink_retrieval_alg")
    print(len(x))
    print(x)

    plt.plot(x)
    plt.ylabel('EAR')
    plt.axis([0, 15, 0, 1.2])
    plt.show()


def draw(frame, leftEye, rightEye):
    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    
    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 50),	cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 4)
    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 50),    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 110),    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 4)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 110),    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2)



####################################### PARAMS #################################################################
################################################################################################################
M = 15 # se procesa M frames consecutivos
EYE_AR_THRESH = 0.5
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
################################################################################################################

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture('/home/vicente/datasets/somnolencia/Fold3_part2/36/10.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('outpy.avi', fourcc, 30, (frame_width, frame_height))

index = 0
x = []
while True:
    ret, frame = cap.read()

    if ret == True: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        if len(rects) != 1:
            print("Too many faces or no faces")

        else: # solo un rostro
            face = rects[0]
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd] #lista de puntos del ojo izq
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0
            x.append(ear)
            index += 1

            draw(frame, leftEye, rightEye)

            if (index % M) == 0:
                ### process
                blinks = blink_retrieval_alg(x, 0.01)  
                x = []      
            

        cv2.putText(frame, str(index), (frame_width - 100, 50),	cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 4)
        cv2.imshow("Frame", frame)
        out.write(frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
    else:
		break


cv2.destroyAllWindows()
