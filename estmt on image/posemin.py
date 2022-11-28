import cv2
import numpy as np
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

pTime = 0

cap = cv2.VideoCapture(r'posevideos\11.mp4')

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while cap.isOpened():
    rect, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    if (results.pose_landmarks):
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    for id,lm in enumerate(results.pose_landmarks.landmark):
        h,w,c = frame.shape
        print(id,lm)
        cx,cy = int(lm.x*w),int(lm.y*h )
        cv2.circle(frame,(cx,cy),10,(255,0,0),cv2.FILLED)

    frame = rescale_frame(frame, percent=35)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow('image', frame)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()