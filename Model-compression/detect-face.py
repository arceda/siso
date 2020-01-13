import cv2
from mtcnn import MTCNN


def draw_image_with_boxes(img, faces):	
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),6)

        for key, value in face['keypoints'].items():
            cv2.circle(img, value, 2, (0,0,255), 2) 

def get_eye_mouth(img, face):
    mouth_left_point = face['keypoints']['mouth_left']       
    mouth_right_point = face['keypoints']['mouth_right']       
    
    mouth_width = mouth_right_point[0] - mouth_left_point[0]
    mouth_height = int(mouth_width*0.75)

    mouth_box_x = int(mouth_left_point[0])
    mouth_box_y = int(mouth_left_point[1] - mouth_height/2)

    print(mouth_width, mouth_height, mouth_box_x, mouth_box_y)

    mouth = img[ mouth_box_y:mouth_box_y+mouth_height, mouth_box_x:mouth_box_x+mouth_width]

    print(mouth.shape)
    cv2.imshow("mouth", mouth)
    cv2.waitKey(30) 


img = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
faces = detector.detect_faces(img)

#for face in faces:
#    get_eye_mouth(img, face)

draw_image_with_boxes(img, faces)
cv2.imshow("Frame", img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 


cap = cv2.VideoCapture('../Eye-blink-detection/blink_detection_demo.mp4')

while True:
    ret, frame = cap.read()
    if ret == True: 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)

        draw_image_with_boxes(img, faces)
        for face in faces:
            get_eye_mouth(img, face)

        cv2.imshow("Frame", img)
        key = cv2.waitKey(30) 

        if key == ord("q"):
            break
    else:
        break