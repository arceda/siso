import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras
import cv2
import sys
from argparse import ArgumentParser
import datetime;
import tensorflow as tf

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

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

#path_model_1 = sys.argv[1]
#path_model_2 = sys.argv[2]
#video_webcam = int(sys.argv[3]) #0 -> webcam, 1-> video
#path_video = sys.argv[4]

# python3 video_test.py -m1 ../models/model_inception_siso_with_nthu_10k_with_2class.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/003_noglasses_mix.mp4 -o /home/vicente/projects/siso/output/ -w 1
# python3 video_test.py -m1 ../models/model_inception_siso_with_nthu_10k_with_2class.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/018_noglasses_mix.mp4 -o /home/vicente/projects/siso/output/ -w 1
# python3 video_test.py -m1 ../models/model_inception_siso_with_nthu_10k_with_2class.h5 -m2 ../models/model_yawn.h5 -v /home/vicente/datasets/NTHU/testing/videoplayback.mp4 -o /home/vicente/projects/siso/output/ -w 1


#tamanio de nuestras imagenes
WIDTH = 128 
HEIGHT = 128
STEP_FRAME_RATE = 15 # cada 10 frames hacemos el procesamiento

model = keras.models.load_model(args.path_model_1)
print("model paul loaded")

model_yawn = keras.models.load_model(args.path_model_2)
print("model yawn loaded")

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
file_log = open(args.path_output + "/" + log_file_name, "a+")

if int(args.window) == 1:
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow('frame', 1200,800)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if frame_count % STEP_FRAME_RATE == 0:
            #############################################################
            # detect drowsy, paul model            
            img = cv2.resize(frame, (WIDTH,HEIGHT))
            (drowsy_prop, non_drowsy_prop) = predict(model, img) 
            #############################################################

            #############################################################
            # detect yawn            
            img_yawn = cv2.resize(frame, (64,64))
            X = np.array([img_yawn])
            yawn_prop =  model_yawn.predict(X)[0][0]
            #############################################################

            drowsy_prop = round(drowsy_prop, 2)
            non_drowsy_prop = round(non_drowsy_prop, 2)
            yawn_prop = round(yawn_prop, 2)

            if yawn_prop > 0.6 or drowsy_prop > 0.54:
                # perfom action to store img of drowsy
                print("drowsy yawn detected!!!")
                print(drowsy_prop, non_drowsy_prop, yawn_prop)             
                now = datetime.datetime.now()
                img_file_name = now.strftime("img_%Y-%m-%d_%H:%M:%S.%f.jpg")
                file_log.write("\nimg_file_name:\t\t" + img_file_name )
                file_log.write("\ndrowsy_prop:\t\t" + str(drowsy_prop))
                file_log.write("\nnon_drowsy_prop:\t" + str(non_drowsy_prop))
                file_log.write("\nyawn_prop:\t\t" + str(yawn_prop) + "\n")               
                cv2.imwrite(args.path_output + "/" + img_file_name, frame)
            

        cv2.rectangle(frame, (0,0), (180, 60), (255,255,255), -1)
        cv2.putText(frame, "drowsy:     " + str(drowsy_prop), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
        cv2.putText(frame, "no-drowsy: " + str(non_drowsy_prop), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)    
        cv2.putText(frame, "yawn:       " + str(yawn_prop), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)    

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


