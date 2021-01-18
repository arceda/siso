import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    # convierte a escala de grises para mejorar la eficiencia de detección
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # void detectMultiScale(
	# const Mat& image,                 # la imagen a detectar es generalmente una imagen en escala de grises para acelerar la velocidad de detección
	# CV_OUT vector<Rect>& objects,     # grupo de vectores de marco rectangular del objeto detectado；
	# double scaleFactor = 1.1,         # indica el factor de escala de la ventana de búsqueda en dos exploraciones sucesivas
    #                                   # el valor predeterminado es 1.1, lo que significa que la ventana de búsqueda se expandirá un 10% cada vez
	# int minNeighbors = 3,             # Representa el número de vecinos que constituyen el objetivo de detección, y cada ventana es un mínimo de fusión de ventanas mín.
	# int flags = 0,                    # flags--usa el valor prestablecido，cualquiera de los usos: CV_HAAR_DO_CANNY_PRUNING，si se establece en: CV_HAAR_DO_CANNY_PRUNING，
    #                                   # la función usará la detección de bordes Canny para excluir áreas con demasiado o muy poco borde, por lo que estas áreas generalmente no están donde está la cara;
	# Size minSize = Size(),            
	# Size maxSize = Size()             # minSize y maxSize se utilizan para limitar el rango del área de destino
    # );

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI (region of interest) for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

# Camshift

def skeleton_tracker1(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # # detect face in first frame
    # c,r,w,h = detect_one_face(frame) # x, y, width, height
    # pt = (0,c+w/2,r+h/2)
    # # Write track point for first frame
    # output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    # frameCounter = frameCounter + 1

    # # set the initial tracking window
    # track_window = (c,r,w,h)

    # # calculate the HSV histogram in the window
    # # NOTE: you do not need this in the Kalman, Particle or OF trackers
    # roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    # term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    # add
    # width = vcap.get(3)   #float
    # height = vcap.get(4)  #float
    # fps = vcap.get(5)
    x = int(v.get(3)/5)
    y = 0
    w = int(v.get(3))-2*x
    h = int(v.get(4))

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # x,y,w,h = track_window

        # add
        # x,y,w,h = detect_one_face(frame)
        # x = x + 5

        # write the result to the output file
        pt = (frameCounter,x+w/2,y+h/2)
        output.write("%d,%d,%d\n" % pt) # Write as frame_index,pt_x,pt_y

        # h = int(h/2)+10

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #image, top-left, bottom-right, RGB color, width

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes_left = eye_cascade_left.detectMultiScale(roi_gray)
        eyes_right = eye_cascade_right.detectMultiScale(roi_gray)

        for (ex1,ey1,ew1,eh1) in eyes_left:
            break
        for (ex,ey,ew,eh) in eyes_right:
            break

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
        cv2.imshow('img',frame)
        cv2.imshow('img1',croppedImg)
        output_name = "./"+sys.argv[4]+str(frameCounter)+".jpg"
        cv2.imwrite(output_name, croppedImg)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        frameCounter = frameCounter + 1
        if int(str(frameCounter)) >= 1000:
            break
    output.close()


if __name__ == '__main__':
    question_number = -1

    question_number = int(sys.argv[1])
    if (question_number > 4 or question_number < 1):
        print("Input parameters out of bound ...")
        sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2])
    if (question_number == 1):
        skeleton_tracker1(video, "output_camshift.txt")

