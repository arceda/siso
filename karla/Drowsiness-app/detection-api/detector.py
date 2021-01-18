import dlib
import cv2
import numpy
from imutils import face_utils
from scipy.spatial import distance as dist
import dotenv
import os

class DrowsinessDetector:
    def __init__(self):
        dotenv.load_dotenv()
        self._consecutiveDrowsyFrames = 0
        self._maxDrowsyFramesBeforeSignal = int(os.getenv("FRAMES_BEFORE_DROWSINESS_CONFIRMED"))
        self._minimumEyeAspectRatioBeforeCloseAssumed = float(os.getenv("MINIMUM_EYE_ASPECT_RATIO_BEFORE_ASSUMED_CLOSED"))


    def areEyesClosed(self, img):
        landmarkDetector = dlib.get_frontal_face_detector()
        shapePredictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        dets = landmarkDetector(img, 1)
        print("num faces: ", len(dets))

        if not dets:
            return False

        facialLandmarks = shapePredictor(img, dets[0])
        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        shape = face_utils.shape_to_np(facialLandmarks)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.getEyeAspectRatio(leftEye)
        rightEAR = self.getEyeAspectRatio(rightEye)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
        ear = (leftEAR + rightEAR) / 2
        print(ear)

        if ear < self._getMinimumEyeAspectRatio():
            self.incrementNumberConsecutiveDrowsyFrames()
            return True

        return False

    def getEyeAspectRatio(self, eye):
        # compute the euclidean distances between vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between horizontal eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)

        return ear

    def isDrowsy(self):
        return self.getNumberConsecutiveDrowsyFrames() > \
               self.getMaxDrowsyFramesBeforeSignal()

    def _getMinimumEyeAspectRatio(self):
        return self._minimumEyeAspectRatioBeforeCloseAssumed

    def getMaxDrowsyFramesBeforeSignal(self):
        return self._maxDrowsyFramesBeforeSignal

    def getNumberConsecutiveDrowsyFrames(self):
        return self._consecutiveDrowsyFrames

    def incrementNumberConsecutiveDrowsyFrames(self):
        self._consecutiveDrowsyFrames += 1

    def resetNumberConsecutiveDrowsyFrames(self):
        self._consecutiveDrowsyFrames = 0

if __name__ == "__main__":
    d = DrowsinessDetector()
    img = dlib.load_grayscale_image("testImage.png")
    print(d.areEyesClosed(img))
