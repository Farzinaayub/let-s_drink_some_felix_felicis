import cv2
import time
import posemodule as pm
import mediapipe as mp 

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while cap.isOpened():
    #time.sleep(2)
    rect, frame = cap.read()
    lst_frm_prd = detector.findPose(frame)
    frame = lst_frm_prd[0]
    pred = lst_frm_prd[1]
    lmlist = detector.findPostition(frame)
        #if len(lmlist)!=0:
            #print(lmlist)

    frame = rescale_frame(frame, percent=100)

    cTime=time.time()
    #fps=1/120
    fps=1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow('image', frame)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()