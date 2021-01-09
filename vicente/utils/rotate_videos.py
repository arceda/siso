import glob
import sys
import cv2

path = sys.argv[1]
show = sys.argv[2]
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

files = glob.glob(path + "/*.AVI")
for file in files:
    print("Processing video "  + file  + "...")

    cap = cv2.VideoCapture(file)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(file + "_ROTATE.AVI", fourcc, 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if ret:
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_180)
            out.write(frame_rotated)
            if show == "1":
                cv2.imshow("Frame", frame)
                cv2.imshow("Frame rotated", frame_rotated)
                

                key = cv2.waitKey(30) & 0xFF   
                if key == ord("q"):
                    break
        else:
            break

cv2.destroyAllWindows()
print("finish")